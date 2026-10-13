"""CPU-only search over vectors produced by the trained right tower."""
from pathlib import Path
import json
import math

import hnswlib
import numpy as np

from .artifacts import load_subset


def text(value):
    if value is None or isinstance(value, float) and not math.isfinite(value):
        return ""
    value = " ".join(str(value).split())
    return "" if value.casefold() in {"", "nan", "none", "null", "unknown", "n/a"} else value


class CandidateIndex:
    def __init__(self, root, ef=1024):
        root = Path(root)
        self.manifest, self.source_manifest = load_subset(root, "right")
        self.catalog = json.loads((root / "catalog.json").read_text())
        self.ids = [str(r["m4a_id"]) for r in self.catalog]
        self.lookup = {tid: i for i, tid in enumerate(self.ids)}
        self.keys = ["spotify:" + text(r.get("spotify_id")) if text(r.get("spotify_id"))
            else "m4a:" + str(r["m4a_id"]) for r in self.catalog]
        self.artists = [(text(r.get("m4a_artist")) or text(r.get("artist"))).casefold()
            for r in self.catalog]
        self.vectors = np.load(root / "track_vectors.npy", mmap_mode="r", allow_pickle=False)
        n = self.manifest["tracks"]
        if len(self.ids) != n or len(self.lookup) != n or self.vectors.shape != (n, 256):
            raise ValueError("Catalog/vector shape mismatch")
        if self.vectors.dtype != np.float32 or not np.isfinite(self.vectors).all():
            raise ValueError("Invalid track vectors")
        if not np.allclose(np.linalg.norm(self.vectors, axis=1), 1, atol=1e-5, rtol=0):
            raise ValueError("Track vectors must be normalized")
        self.index = hnswlib.Index(space="cosine", dim=256)
        self.index.load_index(str(root / "tracks.hnsw"))
        if self.index.get_current_count() != n or set(self.index.get_ids_list()) != set(range(n)):
            raise ValueError("Index/catalog identity mismatch")
        selected = np.random.default_rng(42).choice(n, min(n, 64), replace=False)
        if not np.allclose(self.index.get_items(selected), self.vectors[selected], atol=3e-6, rtol=1e-5):
            raise ValueError("Index vectors differ from exported vectors")
        self.ef = min(n, max(1024, ef))
        self.index.set_ef(self.ef)  # Fixed after startup; no races with concurrent requests.

    def health(self):
        return {"status": "ok", "api_version": 1, "bundle_id": self.manifest["bundle_id"],
            "release_id": self.manifest["release_id"], "dimension": 256, "n_items": len(self.ids),
            "device": "cpu", "ef": self.ef}

    def search(self, vector, k=200, mode="hnsw", history=(), discovery=False,
               query_type="exact", exclude_m4a_id="", exclude_artist="", exclude_ids=()):
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (256,) or not np.isfinite(vector).all():
            raise ValueError("Expected 256 finite vector values")
        norm = float(np.linalg.norm(vector))
        if not .995 <= norm <= 1.005:
            raise ValueError("Query vector must be L2-normalized")
        if not 1 <= k <= 200 or mode not in {"exact", "hnsw"}:
            raise ValueError("Invalid search options")
        vector = vector / norm
        banned = {self.keys[self.lookup[str(t)]] for t in exclude_ids if str(t) in self.lookup}
        if discovery:
            banned.update(self.keys[self.lookup[str(t)]] for t in history if str(t) in self.lookup)
        artist = text(exclude_artist).casefold() if query_type == "similar_to" else ""
        if query_type == "similar_to" and text(exclude_m4a_id) in self.lookup:
            banned.add(self.keys[self.lookup[text(exclude_m4a_id)]])

        def clean(labels, scores):
            seen, items = set(), []
            for label, score in zip(labels, scores):
                i = int(label)
                key = self.keys[i]
                if key in seen:
                    continue
                seen.add(key)
                if key in banned or artist and self.artists[i] == artist:
                    continue
                items.append({"m4a_id": self.ids[i], "score": float(score)})
                if len(items) == k:
                    break
            return items

        if mode == "exact":
            scores = self.vectors @ vector
            labels = np.argsort(-scores, kind="stable")
            items = clean(labels, scores[labels])
        else:
            count = min(len(self.ids), max(self.ef, 5 * k))
            while True:
                labels, distances = self.index.knn_query(vector[None, :], k=count, num_threads=1)
                items = clean(labels[0], 1 - distances[0])
                if len(items) >= k or count == len(self.ids):
                    break
                count = min(len(self.ids), count * 2)
        return {"bundle_id": self.manifest["bundle_id"], "release_id": self.manifest["release_id"],
            "mode": mode, "returned": len(items), "candidates": items,
            "ids": [x["m4a_id"] for x in items], "scores": [x["score"] for x in items]}
