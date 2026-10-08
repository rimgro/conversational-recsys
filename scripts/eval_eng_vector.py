"""Minimal English-set evaluation: vector-only (no tagger, no diffusion).

Loads the existing LanceDB index, embeds `train_queries` with EmbeddingGemma-2
and reports nDCG@20 / Recall / MRR + end-to-end RPS.
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--lancedb", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gemma-module", default="recsys/retrieval/gemma_lancedb.py")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--max-queries", type=int, default=5000)
    p.add_argument("--query-batch", type=int, default=64)
    args = p.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    pre = load_module("crs_prefilter.py", "crs_prefilter")
    gemma = load_module(args.gemma_module, "gemma_mod")
    import lancedb

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = lancedb.connect(args.lancedb).open_table("tracks")
    queries, qrels = pre.load_queries_qrels(
        f"{args.dataset}/train_queries.parquet",
        f"{args.dataset}/train_qrels.parquet",
        max_queries=args.max_queries,
    )
    print(f"[eval] queries: {len(queries)} | db rows: {table.count_rows()}", flush=True)
    t0 = time.time()
    embedder = gemma.GemmaEmbedder(device=args.device, dtype=args.dtype)
    metrics = pre.evaluate_approach(
        table, queries, qrels, approach="vector", embedder=embedder,
        k=20, query_batch=args.query_batch,
    )
    metrics["seconds"] = time.time() - t0
    e2e = pre.measure_rps(table, queries, approach="vector", embedder=embedder, sample=200, k=20)
    metrics["rps_e2e"] = e2e["rps"]
    metrics["latency_e2e_ms"] = e2e["latency_ms_mean"]
    (out / "results_en_vector.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("n", "ndcg@20", "recall@20", "recall@100", "mrr", "rps_e2e", "seconds")}, indent=2), flush=True)
    print("by_query_type:", json.dumps(metrics["by_query_type"], ensure_ascii=False)[:800], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
