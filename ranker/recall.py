"""Потолок ранкера: recall@k каждого источника и пула после RRF, по query_family и is_new.

    python -m ranker.recall --data-dir ../conversational-recsys/data --bm25-index-dir path/to/indexes \
        [--split test_public] [--n-queries 5000] [--hnsw module:function]
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ranker import candidates, data
from ranker.run import log, make_sources, sample_queries


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--bm25-index-dir")
    ap.add_argument("--bm25-url")
    ap.add_argument("--bm25-api-key")
    ap.add_argument("--hnsw")
    ap.add_argument("--split", default="test_public")
    ap.add_argument("--n-queries", type=int, default=5000)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--out", default="recall.csv")
    args = ap.parse_args(argv)

    tracks = data.load_tracks(args.data_dir)
    split = data.load_split(args.data_dir, args.split, tracks)
    q = sample_queries(split.queries, args.n_queries).copy()
    q["words"] = candidates.parse_words(q, candidates.rule_words_fn(args.data_dir))
    sources = make_sources(SimpleNamespace(**vars(args)))
    tid = tracks["track_id"].to_numpy()
    in_hist = set(zip(split.history["user_id"], split.history["track_idx"]))

    rows = []
    for i, r in enumerate(q.itertuples(index=False)):
        target = tid[r.target_idx]
        lists = {s.name: [t for t, _ in s.search(r.query, list(r.words), args.top_k)] for s in sources}
        fused = [t for t, _, _ in candidates.rrf(lists, top_n=args.top_n)]
        row = {"query_family": r.query_family, "is_new": r.is_new, "no_words": not r.words,
               "in_history": (r.user_id, r.target_idx) in in_hist, f"pool@{args.top_n}": target in fused}
        row.update({f"{n}@{args.top_k}": target in set(ids) for n, ids in lists.items()})
        rows.append(row)
        if (i + 1) % 1000 == 0:
            log(f"{i + 1}/{len(q)}")

    df = pd.DataFrame(rows)
    metrics = [c for c in df.columns if "@" in c] + ["in_history", "no_words"]
    report = pd.concat([df[metrics].mean().to_frame("all").T,
                        df.groupby("query_family")[metrics].mean(),
                        df.groupby("is_new")[metrics].mean().rename(index=lambda x: f"is_new={x}")])
    report["n"] = [len(df)] + df.groupby("query_family").size().tolist() + df.groupby("is_new").size().tolist()
    report.round(4).to_csv(args.out)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(report.round(4))


if __name__ == "__main__":
    main()
