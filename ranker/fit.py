"""Обучение и оценка ранкера на признаках из ranker.pit (новый формат: query_id, label).

    python -m ranker.fit --train-features features_train.parquet --queries ranker_train_queries.csv \
        --test-features features_test_public.parquet --data-dir ../new_ds_v1/music4all_crs \
        --model-dir model [--use-query-type]

train/valid — по колонке part из списка запросов (valid = отдельные пользователи).
Признаки = ranker.pit.FEATURES + числовые колонки источников из файла (rrf_score, n_sources, ...).
Отчёт: nDCG@20 ранкера и порядка RRF по query_type и is_new; запросы без таргета в пуле дают 0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import polars as pl

from ranker import evaluate, train as train_mod
from ranker.pit import AGE_BUCKETS, CATEGORICAL, FEATURES, GENDERS, log
from ranker.ranker import LGBMRanker

META_COLS = {"query_id", "m4a_id", "query_type", "label"}


def source_columns(schema: Dict[str, pl.DataType], allowed: List[str]) -> List[str]:
    """Колонки источников из файла признаков; по умолчанию только rrf_score и n_sources."""
    return [c for c, t in schema.items() if c not in META_COLS and c not in FEATURES and t.is_numeric()
            and ("all" in allowed or c in allowed)]


def to_pandas(df: pl.DataFrame, features: List[str], categorical: Dict[str, List[str]]) -> pd.DataFrame:
    keep = [c for c in ["query_id", "label"] + features if c in df.columns]
    out = df.select(keep).to_pandas()
    for col, cats in categorical.items():
        vals = out[col].where(out[col].isin(cats), "other" if "other" in cats else None)
        out[col] = pd.Categorical(vals, categories=cats)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-features", required=True)
    ap.add_argument("--queries", required=True, help="ranker_train_queries.csv (query_id, part)")
    ap.add_argument("--test-features")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--test-split", default="test_public")
    ap.add_argument("--eval-queries", help="csv с query_id: на каких запросах тестового сплита считать метрику "
                                           "(по умолчанию — все запросы сплита)")
    ap.add_argument("--model-dir", default="model")
    ap.add_argument("--use-query-type", action="store_true", help="query_type как категориальный признак")
    ap.add_argument("--source-features", default="rrf_score,n_sources",
                    help="какие колонки источников брать признаками (через запятую; all — все числовые)")
    ap.add_argument("--params", default="{}", help='JSON поверх train.PARAMS, например {"min_data_in_leaf": 50}')
    ap.add_argument("--chunk-rows", type=int, default=5_000_000)
    args = ap.parse_args(argv)
    data_dir = Path(args.data_dir)

    log("признаки train")
    tr = pl.read_parquet(args.train_features)
    parts = pl.read_csv(args.queries).select("query_id", "part")
    tr = tr.join(parts, on="query_id", how="inner")
    sources = source_columns(dict(tr.schema), args.source_features.split(","))
    categorical = {"age_bucket": AGE_BUCKETS, "gender": GENDERS,
                   "top_genre": sorted(set(tr["top_genre"].unique().to_list()) | {"other", "unknown"})}
    features = sources + FEATURES
    if args.use_query_type:
        categorical["query_type"] = sorted(tr["query_type"].unique().to_list())
        features = features + ["query_type"]
    assert set(CATEGORICAL) <= set(categorical)
    log(f"строк {tr.height}, запросов {tr['query_id'].n_unique()}, признаков {len(features)}: {features}")

    train_df = to_pandas(tr.filter(pl.col("part") == "train"), features, categorical)
    valid_df = to_pandas(tr.filter(pl.col("part") == "valid"), features, categorical)
    del tr
    log("обучение")
    booster, info = train_mod.train(train_df, valid_df, params=json.loads(args.params), features=features,
                                    categorical=categorical, group="query_id")
    info["sources"] = sources
    train_mod.save(booster, args.model_dir, info, features=features, categorical=categorical)
    log(f"модель: {info}")
    print(train_mod.feature_importance(booster).to_string(index=False))
    if not args.test_features:
        return

    log(f"скоринг {args.test_split}")
    ranker = LGBMRanker(args.model_dir)
    lf = pl.scan_parquet(args.test_features)
    n = lf.select(pl.len()).collect().item()
    scored = []
    for off in range(0, n, args.chunk_rows):
        chunk = lf.slice(off, args.chunk_rows).collect()
        pdf = to_pandas(chunk, features, categorical)
        keep = chunk.select(["query_id", "label"] + [c for c in ("rrf_score",) if c in chunk.columns]).to_pandas()
        scored.append(keep.assign(ranker=ranker.score(pdf)))
    scored = pd.concat(scored, ignore_index=True)

    queries = (pl.read_parquet(data_dir / f"{args.test_split}_queries.parquet", columns=["query_id", "query_type"])
               .join(pl.read_parquet(data_dir / f"{args.test_split}_qrels.parquet", columns=["query_id", "is_new"]),
                     on="query_id"))
    if args.eval_queries:
        queries = queries.join(pl.read_csv(args.eval_queries).select("query_id"), on="query_id", how="semi")
    report = evaluate.compare_labels(scored, queries.to_pandas(), ["ranker"] + (["rrf_score"] if "rrf_score" in scored else []))
    out = Path(args.model_dir) / f"report_{args.test_split}.csv"
    report.to_csv(out)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(report)
    log(f"отчёт: {out}")


if __name__ == "__main__":
    main()
