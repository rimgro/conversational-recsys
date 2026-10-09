"""Evaluate prefilter variants with cached query vectors and query tags.

Reuses cached track tags (`--tags`), precomputes query embeddings and query tags
ONCE, then scores: vector, tags-only, hybrid_and, hybrid_or, hybrid_soft.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ndcg20(rank):
    import math

    return 1.0 / math.log2(rank + 2) if rank is not None and rank < 20 else 0.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--lancedb", required=True)
    p.add_argument("--tags", required=True, help="JSONL with m4a_id + extracted_tags")
    p.add_argument("--out", required=True)
    p.add_argument("--gemma-module", default="recsys/retrieval/gemma_lancedb.py")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--max-queries", type=int, default=5000)
    p.add_argument("--query-batch", type=int, default=64)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--tag-model", default="dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1")
    p.add_argument("--approaches", default="vector,tags,hybrid_and,hybrid_or,hybrid_soft")
    args = p.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    pre = load_module("crs_prefilter.py", "crs_prefilter")
    gemma = load_module(args.gemma_module, "gemma_mod")
    import lancedb
    import numpy as np

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- tagged DB ---------------------------------------------------------
    tags_by_id = {}
    for line in Path(args.tags).read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            tags_by_id[o["m4a_id"]] = o["extracted_tags"]
    db_path = out / "lancedb"
    if (db_path / "tracks.lance").exists():
        table = lancedb.connect(str(db_path)).open_table("tracks")
    else:
        table = pre.add_tags_column(args.lancedb, db_path, tags_by_id)
    print(f"[eval] db rows: {table.count_rows()}", flush=True)

    queries, qrels = pre.load_queries_qrels(
        f"{args.dataset}/train_queries.parquet",
        f"{args.dataset}/train_qrels.parquet",
        max_queries=args.max_queries,
    )
    print(f"[eval] queries: {len(queries)}", flush=True)
    embedder = gemma.GemmaEmbedder(device=args.device, dtype=args.dtype)
    metadata_df = pre.load_metadata(table)

    # ---- precompute query vectors -----------------------------------------
    t0 = time.time()
    vecs = []
    for s in range(0, len(queries), args.query_batch):
        batch = [q["query"] for q in queries[s : s + args.query_batch]]
        v = embedder.encode([f"task: search result | query: {t}" for t in batch], batch_size=args.query_batch)
        vecs.extend(v)
    print(f"[eval] embedded {len(vecs)} queries in {time.time()-t0:.0f}s", flush=True)

    # ---- precompute query tags (cache) ------------------------------------
    qtags_path = out / "query_tags.jsonl"
    if qtags_path.exists():
        qtags = [json.loads(l) for l in qtags_path.read_text(encoding="utf-8").splitlines()]
    else:
        tagger = pre.DiffusionTagClassifier(args.tag_model, device=args.device, dtype=args.dtype, max_length=args.max_length)
        qtags = []
        t1 = time.time()
        for s in range(0, len(queries), 64):
            batch = queries[s : s + 64]
            for q in batch:
                inc, exc = pre.extract_query_tags(q["query"], tagger, threshold=args.threshold)
                qtags.append({"include": inc, "exclude": exc})
            if s % 640 == 0:
                print(f"[eval] query tags {s}/{len(queries)}", flush=True)
        qtags_path.write_text("\n".join(json.dumps(x) for x in qtags), encoding="utf-8")
        print(f"[eval] query tags in {time.time()-t1:.0f}s", flush=True)

    approaches = [a.strip() for a in args.approaches.split(",") if a.strip()]
    results = {}
    for approach in approaches:
        lat = []
        recs = []
        for i, q in enumerate(queries):
            v = vecs[i]
            inc = qtags[i]["include"]
            exc = qtags[i]["exclude"]
            t2 = time.perf_counter()
            if approach == "vector":
                rows = pre.search_vector(table, v, 100)
            elif approach == "tags":
                rows = pre.search_tags_only(metadata_df, inc, exc, 100)
            elif approach == "hybrid_and":
                rows = pre.search_hybrid_v2(table, v, inc, exc, 100, mode="and")
            elif approach == "hybrid_or":
                rows = pre.search_hybrid_v2(table, v, inc, exc, 100, mode="or")
            elif approach == "hybrid_soft":
                rows = pre.search_hybrid_v2(table, v, inc, exc, 100, mode="soft")
            else:
                raise ValueError(approach)
            lat.append(time.perf_counter() - t2)
            qrel = qrels[q["query_id"]]
            rows = pre._filter_similar_to(rows, qrel)
            ranked = [r["id"] for r in rows]
            gt = qrel["target_m4a_id"]
            rank = ranked.index(gt) if gt in ranked else None
            recs.append((q["query_type"], rank))
        n = max(1, len(recs))
        by = {}
        for qt, r in recs:
            by.setdefault(qt, []).append(r)
        results[approach] = {
            "n": len(recs),
            "ndcg@20": sum(ndcg20(r) for _, r in recs) / n,
            "recall@20": sum(1 for _, r in recs if r is not None and r < 20) / n,
            "recall@100": sum(1 for _, r in recs if r is not None) / n,
            "mrr": sum(1.0 / (r + 1) for _, r in recs if r is not None) / n,
            "search_ms_mean": 1000.0 * sum(lat) / n,
            "by_query_type": {
                qt: {
                    "n": len(rs),
                    "ndcg@20": sum(ndcg20(r) for r in rs) / max(1, len(rs)),
                    "recall@20": sum(1 for r in rs if r is not None and r < 20) / max(1, len(rs)),
                    "recall@100": sum(1 for r in rs if r is not None) / max(1, len(rs)),
                }
                for qt, rs in sorted(by.items())
            },
        }
        m = results[approach]
        print(f"[eval] {approach:12s} nDCG@20={m['ndcg@20']:.4f} R@20={m['recall@20']:.4f} "
              f"R@100={m['recall@100']:.4f} MRR={m['mrr']:.4f} search={m['search_ms_mean']:.1f}ms", flush=True)
        (out / f"results_en_{approach}.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary_variants.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
