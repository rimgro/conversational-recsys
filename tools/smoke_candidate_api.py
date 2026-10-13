"""Check a running API against its local matched vectors; no Qwen/GPU required."""
from pathlib import Path
import argparse
import json
import os
import sys
import time

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--right", type=Path, required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--api-key-file", type=Path)
    p.add_argument("--queries", type=int, default=3)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if not 1 <= args.queries <= 64:
        p.error("queries must be between 1 and 64")
    from recsys.serving.artifacts import load_subset
    from recsys.serving.client import CandidateAPIClient
    from recsys.serving.index import text
    serving, _ = load_subset(args.right, "right")
    key = args.api_key_file.read_text().strip() if args.api_key_file else os.environ.get("QWEN_API_KEY")
    vectors = np.load(args.right / "track_vectors.npy", mmap_mode="r", allow_pickle=False)
    catalog = json.loads((args.right / "catalog.json").read_text())
    ids = [str(r["m4a_id"]) for r in catalog]
    canonical = ["spotify:" + text(r.get("spotify_id")) if text(r.get("spotify_id"))
        else "m4a:" + str(r["m4a_id"]) for r in catalog]
    lookup = dict(zip(ids, canonical))
    samples = np.random.default_rng(42).choice(len(ids), min(args.queries, len(ids)), replace=False)
    latencies = []
    with CandidateAPIClient(args.url, key, serving["bundle_id"], serving["release_id"], timeout=30.) as api:
        api.check_health()
        for label in samples:
            vector = np.asarray(vectors[label])
            scores = vectors @ (vector / np.linalg.norm(vector))
            ranking = np.argsort(-scores, kind="stable")
            seen, expected = set(), []
            for i in ranking:
                if canonical[i] not in seen:
                    expected.append(ids[i]);seen.add(canonical[i])
                if len(expected) == 200: break
            for mode in ("exact", "hnsw"):
                started = time.perf_counter()
                result = api.search(vector.tolist(), mode=mode)
                latencies.append((time.perf_counter() - started) * 1000)
                assert len({lookup[t] for t in result["ids"]}) == result["returned"]
                assert result["returned"] == min(200, len(set(canonical)))
                if mode == "exact":
                    assert result["ids"] == expected, "Exact ranking differs from local export"
        history = ids[:min(500, len(ids))]
        result = api.search(vectors[samples[0]].tolist(), history=history, discovery=True)
        assert {lookup[t] for t in result["ids"]}.isdisjoint({lookup[t] for t in history})
    report = {"status": "verified", "bundle_id": serving["bundle_id"],
        "release_id": serving["release_id"], "tracks": len(ids), "dimension": 256,
        "http_requests_verified": len(latencies) + 2, "exact_rankings_match_local_export": True,
        "canonical_deduplication_verified": True, "discovery_verified": True,
        "mean_roundtrip_ms": float(np.mean(latencies)), "max_roundtrip_ms": max(latencies)}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
