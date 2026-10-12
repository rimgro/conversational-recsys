"""Matched query adapter, item tower, preprocessing and 256-dimensional HNSW."""
from pathlib import Path
import json
import shutil
import tempfile

import numpy as np
import torch
from transformers import AutoTokenizer

from recsys.track_tower import TrackPreprocessor, TrackTransformerTower
from .config import TrainConfig
from .data import atomic_json, file_hash
from .features import HistoryHasher, padded_histories, recent_history
from .model import QwenHistoryTower, load_backbone


def export_bundle(trainer, tokenizer):
    import hnswlib
    best_step = trainer.load_best()
    trainer.check_budget()
    base = trainer.run_dir / "exports"
    base.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=base))
    try:
        model, data = trainer.model, trainer.data
        vectors = trainer.item_vectors().cpu().numpy()
        np.save(stage / "track_vectors.npy", vectors)
        torch.save(model.query_tower.trainable_state(), stage / "query_trainable.pt")
        torch.save({k: v.cpu() for k, v in model.item_tower.state_dict().items()}, stage / "item_tower.pt")
        data.preprocessor.save(stage / "preprocessing.json")
        np.savez_compressed(stage / "history_features.npz", indices=data.hash_indices,
            weights=data.hash_weights, audio=data.audio, numbers=data.history_numbers)
        atomic_json(stage / "catalog.json", [{"m4a_id": str(r["m4a_id"]),
            "spotify_id": data.scorer.keys[i].removeprefix("spotify:") if data.scorer.keys[i].startswith("spotify:") else "",
            "artist": data.scorer.artists[i]} for i, r in enumerate(data.catalog)])
        item_args = json.loads((data.cache_dir / "item_args.json").read_text())
        atomic_json(stage / "config.json", {"training": trainer.config.to_dict(), "item_args": item_args,
                                           "hasher": HistoryHasher(trainer.config.hash_buckets, trainer.config.seed).to_dict()})
        tokenizer.save_pretrained(stage / "tokenizer")
        trainer.check_budget()
        index = hnswlib.Index(space="cosine", dim=trainer.config.embedding_dim)
        index.init_index(max_elements=len(vectors), ef_construction=200, M=32, random_seed=trainer.config.seed)
        index.add_items(vectors, np.arange(len(vectors)), num_threads=1)
        index.set_ef(200)
        index.save_index(str(stage / "tracks.hnsw"))
        checksum = file_hash(stage / "query_trainable.pt")[:12]
        bundle_id = f"{trainer.signature[:16]}-step-{best_step}-{checksum}"
        files = {str(p.relative_to(stage)): file_hash(p) for p in sorted(stage.rglob("*")) if p.is_file()}
        atomic_json(stage / "manifest.json", {"schema_version": 1, "bundle_id": bundle_id,
            "space": "cosine", "dimension": trainer.config.embedding_dim, "tracks": len(vectors),
            "best_step": best_step, "model": trainer.config.qwen_model,
            "revision": trainer.config.qwen_revision, "sha256": files})
        destination = base / bundle_id
        if destination.exists():
            existing = json.loads((destination / "manifest.json").read_text())
            if existing != json.loads((stage / "manifest.json").read_text()):
                raise ValueError("Existing bundle differs; refusing to overwrite")
            shutil.rmtree(stage)
        else:
            stage.rename(destination)
        atomic_json(trainer.run_dir / "latest_bundle.json", {"path": str(destination), "bundle_id": bundle_id})
        return destination
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


class QwenRetriever:
    """Training-free inference from a matched exported bundle.

    encode_query() returns the 256-vector accepted by a matching vector API.
    search() uses the included HNSW; API/index deployment is a separate action.
    """
    def __init__(self, bundle_dir, device="cpu", cache_dir=None, verify=True):
        import hnswlib
        self.root, self.device = Path(bundle_dir), torch.device(device)
        manifest = json.loads((self.root / "manifest.json").read_text())
        if manifest["schema_version"] != 1:
            raise ValueError("Unknown bundle schema")
        if verify:
            for name, checksum in manifest["sha256"].items():
                if file_hash(self.root / name) != checksum:
                    raise ValueError("Bundle integrity failure: " + name)
        settings = json.loads((self.root / "config.json").read_text())
        self.config = TrainConfig(**settings["training"])
        self.config.gradient_checkpointing = False
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if self.device.type == "cuda" else torch.float32
        self.dtype = dtype
        backbone, _ = load_backbone(self.config, dtype=dtype, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.root / "tokenizer", trust_remote_code=False)
        self.tokenizer.padding_side = "right"
        self.tower = QwenHistoryTower(backbone, self.config.hash_buckets, self.config.embedding_dim).to(self.device)
        state = torch.load(self.root / "query_trainable.pt", map_location="cpu", weights_only=True)
        self.tower.load_trainable_state(state)
        self.tower.eval().requires_grad_(False)
        self.item_tower = TrackTransformerTower(**settings["item_args"]).to(self.device)
        self.item_tower.load_state_dict(torch.load(self.root / "item_tower.pt", map_location="cpu", weights_only=True))
        self.item_tower.eval().requires_grad_(False)
        self.preprocessor = TrackPreprocessor.from_dict(json.loads((self.root / "preprocessing.json").read_text()))
        self.catalog = json.loads((self.root / "catalog.json").read_text())
        self.ids = [r["m4a_id"] for r in self.catalog]
        self.lookup = {t: i for i, t in enumerate(self.ids)}
        feature = np.load(self.root / "history_features.npz")
        self.features = {key: torch.tensor(feature[key], device=self.device)
                         for key in ("indices", "weights", "audio", "numbers")}
        self.index = hnswlib.Index(space="cosine", dim=manifest["dimension"])
        self.index.load_index(str(self.root / "tracks.hnsw"))
        self.index.set_ef(200)
        self.manifest = manifest

    @torch.inference_mode()
    def encode_query(self, query, history=()):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Nonempty English query required")
        selected = recent_history(history, self.lookup, self.config.history_events)
        histories, mask = padded_histories([selected])
        h = torch.tensor(histories, device=self.device)
        text = self.tokenizer([query], padding=True, truncation=True,
            max_length=self.config.query_tokens, add_special_tokens=False, return_tensors="pt")
        inputs = {"hash_indices": self.features["indices"][h], "hash_weights": self.features["weights"][h],
            "audio": self.features["audio"][h], "numbers": self.features["numbers"][h],
            "history_mask": torch.tensor(mask, device=self.device),
            "query_ids": text["input_ids"].to(self.device), "query_mask": text["attention_mask"].to(self.device)}
        if self.device.type == "cuda":
            with torch.autocast("cuda", dtype=self.dtype):
                vector = self.tower(**inputs)
        else:
            vector = self.tower(**inputs)
        return vector[0].cpu().numpy()

    @torch.inference_mode()
    def encode_item_features(self, packed_features):
        """New tracks use the saved preprocessing and the same fixed MiniLM/MuQ encoders."""
        features = torch.tensor(np.asarray(packed_features, dtype=np.float32), device=self.device)
        return self.item_tower(features).float().cpu().numpy()

    def search(self, query, history=(), k=20):
        vector = self.encode_query(query, history)
        labels, distances = self.index.knn_query(vector[None, :], k=min(k, len(self.ids)))
        return [{"m4a_id": self.ids[i], "score": float(1 - d)} for i, d in zip(labels[0], distances[0])]
