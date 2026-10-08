"""End-to-end pre-filtering run: tag extraction -> LanceDB -> 3-way eval + RPS.

Usage (after mounting the datasets on Kaggle):

    python scripts/run_prefilter_eval.py \
        --dataset /kaggle/input/datasets/<user>/music4all-crs-en \
        --lancedb /kaggle/input/datasets/<user>/music4all-crs-lancedb \
        --out /kaggle/working/prefilter \
        --tagger diffusion --device cuda --threshold 0.8 --max-queries 5000

Steps:
  1. extract 63 tags per track (diffusion LM for mood/theme/vocal + metadata),
  2. rebuild LanceDB with the ``extracted_tags`` column,
  3. evaluate vector / tags / hybrid with nDCG@20 + RPS,
  4. write results to ``--out``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
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
    p = argparse.ArgumentParser(description="pre-filtering end-to-end run")
    p.add_argument("--dataset", required=True, help="dir with tracks_meta.parquet and *_queries/qrels.parquet")
    p.add_argument("--lancedb", required=True, help="dir containing tracks.lance (the existing vector index)")
    p.add_argument("--out", required=True)
    p.add_argument("--prefilter-module", default="crs_prefilter.py")
    p.add_argument("--gemma-module", default="recsys/retrieval/gemma_lancedb.py")
    p.add_argument("--tagger", choices=["diffusion", "lexicon"], default="diffusion")
    p.add_argument("--tag-model", default="dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--query-threshold", type=float, default=0.6)
    p.add_argument("--max-queries", type=int, default=5000)
    p.add_argument("--query-batch", type=int, default=64)
    p.add_argument("--tag-batch", type=int, default=128)
    p.add_argument("--limit-tracks", type=int, default=None)
    p.add_argument("--approaches", default="vector,tags,hybrid")
    args = p.parse_args()

    t0 = time.time()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    pre = load_module(args.prefilter_module, "crs_prefilter")
    gemma = load_module(args.gemma_module, "gemma_mod")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"args": vars(args)}

    import pyarrow.parquet as pq

    # ---- 1. tags ----------------------------------------------------------- #
    tags_path = out / "tags.jsonl"
    columns = [
        "m4a_id", "title", "m4a_song", "artist", "m4a_artist", "release",
        "m4a_album", "release_year", "lang", "m4a_genres_full", "m4a_genres",
        "m4a_tags_full", "m4a_tags", "artist_genres", "album_genres",
        "spotify_genres", "tags", "pseudo_caption", "lyrics", "tempo", "energy",
        "is_instrumental",
    ]
    records = pq.read_table(f"{args.dataset}/tracks_meta.parquet", columns=columns).to_pylist()
    if args.limit_tracks:
        records = records[: args.limit_tracks]
    print(f"[run] tracks: {len(records)}", flush=True)
    if tags_path.exists():
        print("[run] reusing existing tags.jsonl", flush=True)
        tags_by_id = {}
        for line in tags_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                o = json.loads(line)
                tags_by_id[o["m4a_id"]] = o["extracted_tags"]
    else:
        if args.tagger == "diffusion":
            tagger = pre.DiffusionTagClassifier(args.tag_model, device=args.device, dtype=args.dtype)
        else:
            tagger = pre.LexiconTagger()
        tags_by_id = pre.extract_track_tags(
            records, tagger, threshold=args.threshold, batch_size=args.tag_batch
        )
        tags_path.write_text(
            "\n".join(json.dumps({"m4a_id": k, "extracted_tags": v}) for k, v in tags_by_id.items()),
            encoding="utf-8",
        )
    summary["n_tracks"] = len(records)
    summary["tags_seconds"] = time.time() - t0
    print(f"[run] tags done in {summary['tags_seconds']:.0f}s", flush=True)

    # ---- 2. rebuild LanceDB with tags ------------------------------------- #
    db_path = out / "lancedb"
    if not (db_path / "tracks.lance").exists():
        shutil.rmtree(db_path, ignore_errors=True)
        os.makedirs(db_path, exist_ok=True)
        table = pre.add_tags_column(args.lancedb, db_path, tags_by_id)
    else:
        import lancedb

        table = lancedb.connect(str(db_path)).open_table("tracks")
    summary["db_rows"] = table.count_rows()
    print(f"[run] lancedb rows: {summary['db_rows']}", flush=True)

    # ---- 3. queries + embedder -------------------------------------------- #
    queries, qrels = pre.load_queries_qrels(
        f"{args.dataset}/train_queries.parquet",
        f"{args.dataset}/train_qrels.parquet",
        max_queries=args.max_queries,
    )
    print(f"[run] queries: {len(queries)}", flush=True)
    embedder = gemma.GemmaEmbedder(device=args.device, dtype=args.dtype)
    metadata_df = pre.load_metadata(table)
    if args.tagger == "diffusion":
        query_tagger = pre.DiffusionTagClassifier(args.tag_model, device=args.device, dtype=args.dtype)
    else:
        query_tagger = pre.LexiconTagger()

    # ---- 4. evaluate approaches ------------------------------------------- #
    results = {}
    for approach in [a.strip() for a in args.approaches.split(",") if a.strip()]:
        t = time.time()
        metrics = pre.evaluate_approach(
            table, queries, qrels,
            approach=approach, embedder=embedder, tagger=query_tagger,
            metadata_df=metadata_df, k=20, query_batch=args.query_batch,
            threshold=args.query_threshold,
        )
        metrics["seconds"] = time.time() - t
        try:
            e2e = pre.measure_rps(
                table, queries, approach=approach, embedder=embedder,
                tagger=query_tagger, metadata_df=metadata_df,
                sample=200, k=20, threshold=args.query_threshold,
            )
            metrics["rps_e2e"] = e2e["rps"]
            metrics["latency_e2e_ms"] = e2e["latency_ms_mean"]
        except Exception as exc:  # pragma: no cover
            metrics["rps_e2e"] = None
            print(f"[run] e2e rps failed for {approach}: {exc}", flush=True)
        results[approach] = metrics
        (out / f"results_en_{approach}.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[run] {approach:7s} nDCG@20={metrics['ndcg@20']:.4f} "
            f"R@20={metrics['recall@20']:.4f} R@100={metrics['recall@100']:.4f} "
            f"RPS={metrics['rps']:.1f} ({metrics['seconds']:.0f}s)",
            flush=True,
        )

    summary["results"] = results
    summary["total_seconds"] = time.time() - t0
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[run] done in {summary['total_seconds']:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
