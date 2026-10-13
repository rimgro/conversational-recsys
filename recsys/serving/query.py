"""Only the left tower: no item-tower weights, vectors or HNSW are loaded."""
from pathlib import Path
import json

import numpy as np
import torch
from transformers import AutoTokenizer

from recsys.two_tower_qwen.bundle import QwenRetriever
from recsys.two_tower_qwen.config import TrainConfig
from recsys.two_tower_qwen.model import QwenHistoryTower, load_backbone
from .artifacts import load_subset


class QwenQueryEncoder:
    def __init__(self, bundle_dir, device="cpu", cache_dir=None):
        self.root, self.device = Path(bundle_dir), torch.device(device)
        self.serving_manifest, self.manifest = load_subset(self.root, "left")
        settings = json.loads((self.root / "config.json").read_text())
        self.config = TrainConfig(**settings["training"])
        self.config.gradient_checkpointing = False
        self.dtype = ((torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
            if self.device.type == "cuda" else torch.float32)
        backbone, _ = load_backbone(self.config, dtype=self.dtype, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.root / "tokenizer", trust_remote_code=False)
        self.tokenizer.padding_side = "right"
        self.tower = QwenHistoryTower(backbone, self.config.hash_buckets, self.config.embedding_dim).to(self.device)
        state = torch.load(self.root / "query_trainable.pt", map_location="cpu", weights_only=True)
        self.tower.load_trainable_state(state)
        self.tower.eval().requires_grad_(False)
        self.catalog = json.loads((self.root / "catalog.json").read_text())
        self.ids = [str(r["m4a_id"]) for r in self.catalog]
        self.lookup = {t: i for i, t in enumerate(self.ids)}
        if len(self.lookup) != self.manifest["tracks"] or len(self.ids) != len(self.lookup):
            raise ValueError("Left catalog identity mismatch")
        with np.load(self.root / "history_features.npz", allow_pickle=False) as features:
            if any(len(features[key]) != len(self.ids) for key in ("indices", "weights", "audio", "numbers")):
                raise ValueError("History feature count mismatch")
            if any(not np.isfinite(features[key]).all() for key in ("weights", "audio", "numbers")):
                raise ValueError("Nonfinite history features")
            self.features = {key: torch.tensor(features[key], device=self.device)
                for key in ("indices", "weights", "audio", "numbers")}

    # Reuse the frozen training/export inference recipe, including recent 256 events.
    # All filtering still receives the FULL history at the remote index.
    encode_query = QwenRetriever.encode_query
