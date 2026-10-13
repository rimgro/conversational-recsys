"""Small synthetic export; never contains production weights or user histories."""
from pathlib import Path
import json
import numpy as np
import hnswlib
from recsys.serving.artifacts import sha256


def write_manifest(root):
    manifest = {"schema_version": 1, "bundle_id": "synthetic-step-1-test", "space": "cosine",
        "dimension": 256, "tracks": len(json.loads((root / "catalog.json").read_text())),
        "best_step": 1, "model": "synthetic", "revision": "synthetic-pinned",
        "sha256": {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "manifest.json"}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def make_export(root, n=2100):
    root = Path(root)
    root.mkdir(parents=True)
    catalog = [{"m4a_id": str(i), "spotify_id": "alias" + str(i // 2),
        "artist": "artist" + str(i // 2 % 5)} for i in range(n)]
    (root / "catalog.json").write_text(json.dumps(catalog))
    vectors = np.zeros((n, 256), dtype=np.float32)
    vectors[:, 0] = np.linspace(.99, -.99, n)
    vectors[:, 1] = np.sqrt(1 - vectors[:, 0] ** 2)
    np.save(root / "track_vectors.npy", vectors)
    index = hnswlib.Index(space="cosine", dim=256)
    index.init_index(max_elements=n, ef_construction=100, M=16, random_seed=42)
    index.add_items(vectors, np.arange(n), num_threads=1)
    index.save_index(str(root / "tracks.hnsw"))
    (root / "config.json").write_text('{}')
    (root / "item_tower.pt").write_bytes(b"synthetic unused item weights")
    (root / "query_trainable.pt").write_bytes(b"synthetic unused query weights")
    np.savez_compressed(root / "history_features.npz", indices=np.zeros((n, 2), np.int64),
        weights=np.zeros((n, 2), np.float32), audio=np.zeros((n, 128), np.float32),
        numbers=np.zeros((n, 14), np.float32))
    (root / "tokenizer").mkdir()
    (root / "tokenizer/tokenizer_config.json").write_text('{}')
    write_manifest(root)
    return root


if __name__ == "__main__":
    import argparse
    from recsys.serving.artifacts import split_bundle
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    split_bundle(make_export(args.output / "source"), args.output / "split")
