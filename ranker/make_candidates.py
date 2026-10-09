"""Кандидаты для новых данных: BM25 (индексы dev/bm25) [+ hnsw] -> по top_k от каждого -> RRF -> top_n.

    python -m ranker.make_candidates --data-dir ../new_ds_v1/music4all_crs --split train \
        --queries ranker_train_queries.csv --bm25-index-dir ../bm25_indexes_new --bm25 genres,tags,all \
        [--hnsw-candidates hnsw_train.parquet] --out candidates_train.parquet

Запросы английские: слова для BM25 — токены текста запроса. hnsw подключается готовым файлом
(query_id, m4a_id, score — top_k на запрос) или функцией --hnsw module:function(query, k).
Выход: query_id, m4a_id, rrf_score, n_sources (это и есть колонки источников для ранкера).
"""
from __future__ import annotations

import argparse
import importlib
import re
import time
from pathlib import Path

import polars as pl

from ranker.candidates import BM25IndexSource, FunctionSource, rrf

_TOKEN = re.compile(r"[^\w']+")


def tokens(text: str):
    return [t for t in _TOKEN.split((text or "").lower()) if t]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--queries", help="csv/parquet с query_id; по умолчанию все запросы сплита")
    ap.add_argument("--bm25-index-dir", required=True)
    ap.add_argument("--bm25", default="genres,tags,all", help="имена индексов через запятую")
    ap.add_argument("--hnsw", help="module:function, function(query, k) -> [(m4a_id, score)]")
    ap.add_argument("--hnsw-candidates", help="готовый parquet: query_id, m4a_id, score")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    q = pl.read_parquet(Path(args.data_dir) / f"{args.split}_queries.parquet", columns=["query_id", "query"])
    if args.queries:
        ids = pl.read_csv(args.queries) if args.queries.endswith(".csv") else pl.read_parquet(args.queries)
        q = q.join(ids.select("query_id"), on="query_id", how="semi")
    sources = [BM25IndexSource(f"bm25_{n}", Path(args.bm25_index_dir) / n) for n in args.bm25.split(",") if n]
    if args.hnsw:
        mod, fn = args.hnsw.split(":")
        sources.append(FunctionSource("hnsw", getattr(importlib.import_module(mod), fn)))
    hnsw = {}
    if args.hnsw_candidates:
        h = (pl.read_parquet(args.hnsw_candidates).sort(["query_id", "score"], descending=[False, True])
             .group_by("query_id", maintain_order=True).head(args.top_k))
        hnsw = {k[0]: v["m4a_id"].to_list() for k, v in h.group_by("query_id", maintain_order=True)}
    print(f"запросов {q.height}, источники: {[s.name for s in sources] + (['hnsw(file)'] if hnsw else [])}",
          flush=True)

    qid_out, mid_out, score_out, n_out = [], [], [], []
    t0 = time.time()
    for i, (qid, text) in enumerate(zip(q["query_id"], q["query"])):
        words = tokens(text)
        lists = {s.name: [t for t, _ in s.search(text, words, args.top_k)] for s in sources}
        if hnsw:
            lists["hnsw"] = hnsw.get(qid, [])
        for tid, score, n in rrf(lists, k=args.rrf_k, top_n=args.top_n):
            qid_out.append(qid), mid_out.append(tid), score_out.append(score), n_out.append(n)
        if (i + 1) % 10_000 == 0:
            print(f"  {i + 1}/{q.height} ({time.time() - t0:.0f} с)", flush=True)
    out = pl.DataFrame({"query_id": qid_out, "m4a_id": mid_out,
                        "rrf_score": pl.Series(score_out, dtype=pl.Float32),
                        "n_sources": pl.Series(n_out, dtype=pl.Int8)})
    out.write_parquet(args.out)
    print(f"готово: {args.out}, строк {out.height}, {time.time() - t0:.0f} с", flush=True)




# ---------------------------------------------------------------- готовые топы источников из файлов

def fuse_list_files(files: dict, query_ids: pl.DataFrame, top_k: int = 100, top_n: int = 200,
                    rrf_k: int = 60) -> pl.DataFrame:
    """files: {источник: parquet со строкой на запрос: query_id, list[m4a_id], list[score]} (топы по убыванию).
    -> query_id, m4a_id, rrf_score, n_sources, score_<источник>. Порядок внутри списка = ранг."""
    parts = []
    for name, path in files.items():
        schema = pl.read_parquet_schema(path)
        ids_col = next(c for c, t in schema.items() if t == pl.List(pl.Utf8))
        sc_col = next((c for c, t in schema.items() if isinstance(t, pl.List) and t.inner.is_float()), None)
        cols = [pl.col(ids_col).list.head(top_k).alias("m4a_id")]
        if sc_col:
            cols.append(pl.col(sc_col).list.head(top_k).cast(pl.List(pl.Float32)).alias("score"))
        lf = (pl.scan_parquet(path).join(query_ids.lazy(), on="query_id", how="semi")
              .select("query_id", *cols).explode(["m4a_id"] + (["score"] if sc_col else []))
              .with_columns((pl.int_range(pl.len()).over("query_id") + 1).alias("rank"))
              .unique(["query_id", "m4a_id"], keep="first")
              .with_columns(pl.lit(name).alias("source")))
        if not sc_col:
            lf = lf.with_columns(pl.lit(None, dtype=pl.Float32).alias("score"))
        parts.append(lf.select("query_id", "m4a_id", "rank", "score", "source").collect())
    long = pl.concat(parts)
    fused = (long.group_by("query_id", "m4a_id")
             .agg((1.0 / (rrf_k + pl.col("rank"))).sum().cast(pl.Float32).alias("rrf_score"),
                  pl.len().cast(pl.Int8).alias("n_sources"))
             .sort(["query_id", "rrf_score", "m4a_id"], descending=[False, True, False])
             .group_by("query_id", maintain_order=True).head(top_n))
    wide = long.pivot(on="source", index=["query_id", "m4a_id"], values="score")
    wide = wide.rename({c: f"score_{c}" for c in wide.columns if c not in ("query_id", "m4a_id")})
    return fused.join(wide, on=["query_id", "m4a_id"], how="left")


def main_files(argv=None) -> None:
    """python -m ranker.make_candidates files --queries q.csv --file hnsw=top200_gemma.parquet --file bm25=top200.parquet --out c.parquet"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="csv/parquet с query_id")
    ap.add_argument("--file", action="append", required=True, help="имя=путь к parquet с топами")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    ids = pl.read_csv(args.queries) if args.queries.endswith(".csv") else pl.read_parquet(args.queries)
    t0 = time.time()
    out = fuse_list_files(dict(f.split("=", 1) for f in args.file), ids.select("query_id").unique(),
                          args.top_k, args.top_n, args.rrf_k)
    out.write_parquet(args.out)
    print(f"готово: {args.out}, строк {out.height}, запросов {out['query_id'].n_unique()}, {time.time() - t0:.0f} с",
          flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "files":
        main_files(sys.argv[2:])
    else:
        main()
