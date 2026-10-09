"""Весь цикл: данные -> A/B -> приоры -> пулы -> признаки -> обучение -> оценка на test_public.

    python -m ranker.run --data-dir ../conversational-recsys/data --work-dir artifacts \
        --bm25-index-dir ../conversational-recsys/bm25/indexes [--hnsw my_pkg.hnsw:search] [--max-queries 5000]

Пулы и слова кэшируются в work-dir (повторный запуск их не пересчитывает; --force — пересчитать).
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ranker import candidates, data, evaluate, features, priors as priors_mod, train as train_mod
from ranker.ranker import LGBMRanker

FRAC_A = 0.65       # доля train-пользователей под приоры (CTR артиста)
FRAC_VALID = 0.10   # доля пользователей части B под early stopping


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_sources(args) -> List:
    srcs: List = []
    if args.hnsw:
        mod, fn = args.hnsw.split(":")
        srcs.append(candidates.FunctionSource("hnsw", getattr(importlib.import_module(mod), fn)))
    for name in args.bm25_indexes.split(","):
        if args.bm25_url:
            srcs.append(candidates.BM25HttpSource(f"bm25_{name}", args.bm25_url, name, args.bm25_api_key))
        else:
            srcs.append(candidates.BM25IndexSource(f"bm25_{name}", Path(args.bm25_index_dir) / name))
    log(f"источники: {[s.name for s in srcs]}")
    return srcs


def sample_queries(q: pd.DataFrame, n) -> pd.DataFrame:
    if not n or len(q) <= n:
        return q
    keep = data.hash_split(q["qid"], n / len(q), salt="debug-sample")
    return q[keep].reset_index(drop=True)


def cached_pools(name: str, q: pd.DataFrame, args, sources, tracks, words_fn) -> pd.DataFrame:
    work = Path(args.work_dir)
    wpath, ppath = work / f"words_{name}.parquet", work / f"pools_{name}.parquet"
    if wpath.exists() and not args.force:
        w = pd.read_parquet(wpath).set_index("qid")["words"]
        q["words"] = [list(x) for x in w.reindex(q["qid"])]
    else:
        q["words"] = candidates.parse_words(q, words_fn)
        q[["qid", "words"]].to_parquet(wpath)
    if ppath.exists() and not args.force:
        pools = pd.read_parquet(ppath)
        return pools[pools["qid"].isin(set(q["qid"]))]
    log(f"пулы {name}: {len(q)} запросов")
    pools = candidates.build_pools(q, sources, tracks, top_k=args.top_k, top_n=args.top_n,
                                   progress_every=5000)
    pools.to_parquet(ppath)
    return pools


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", default="artifacts")
    ap.add_argument("--bm25-index-dir", help="папка с индексами genres/ и tags/ (формат dev/bm25)")
    ap.add_argument("--bm25-indexes", default="genres,tags,all",
                    help="какие индексы BM25 берём источниками (all = artist+song+genres+tags)")
    ap.add_argument("--bm25-url", help="вместо локальных индексов: http://host:8000/bm25/search")
    ap.add_argument("--bm25-api-key")
    ap.add_argument("--hnsw", help="module:function, function(query, k) -> [(m4a_id, score)]")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=20.0, help="сглаживание CTR")
    ap.add_argument("--max-queries", type=int, default=0, help="сэмпл запросов на сплит (для отладки)")
    ap.add_argument("--eval-split", default="test_public")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if not (args.bm25_index_dir or args.bm25_url):
        ap.error("нужен --bm25-index-dir или --bm25-url")
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    log("каталог")
    tracks = data.load_tracks(args.data_dir)
    codes = features.Codes(tracks)
    log(f"треков {len(tracks)}, тегов {len(codes.vocab)}")

    log("train")
    tr = data.load_split(args.data_dir, "train", tracks)
    users = tr.users["user_id"].to_numpy()
    in_a = data.hash_split(users, FRAC_A, salt="priors")
    part_a, part_b = data.subset(tr, users[in_a]), data.subset(tr, users[~in_a])
    log(f"пользователей {len(users)}: A={in_a.sum()} B={(~in_a).sum()}; запросов B={len(part_b.queries)}")

    pri_a = priors_mod.compute_priors(part_a, tracks, alpha=args.alpha)
    pri_a.save(work / "priors_A")
    sources = make_sources(args)
    words_fn = candidates.rule_words_fn(args.data_dir)

    qb = sample_queries(part_b.queries, args.max_queries)
    pools_b = cached_pools("train_B", qb, args, sources, tracks, words_fn)
    stats_b = features.UserStats(part_b, codes)
    log("признаки B")
    fb = features.build_features(pools_b, qb, stats_b, codes, pri_a)
    b_users = qb["user_id"].unique()
    is_valid = pd.Series(data.hash_split(b_users, FRAC_VALID, salt="valid"), index=b_users)
    vmask = is_valid.reindex(qb.set_index("qid")["user_id"].reindex(fb["qid"]).to_numpy()).to_numpy()
    recall_b = fb.groupby("qid")["label"].max().reindex(qb["qid"]).fillna(0).mean()
    log(f"строк {len(fb)}, recall@{args.top_n} пула на B: {recall_b:.4f}")

    log("обучение")
    booster, info = train_mod.train(fb[~vmask], fb[vmask])
    info["recall_pool_B"] = float(recall_b)
    train_mod.save(booster, work / "model", info)
    log(f"модель: {info}")
    del fb

    log(args.eval_split)
    pri_all = priors_mod.compute_priors(tr, tracks, alpha=args.alpha)
    pri_all.save(work / "priors_all")
    te = data.load_split(args.data_dir, args.eval_split, tracks)
    qt = sample_queries(te.queries, args.max_queries)
    pools_t = cached_pools(args.eval_split, qt, args, sources, tracks, words_fn)
    ft = features.build_features(pools_t, qt, features.UserStats(te, codes), codes, pri_all)
    scores = LGBMRanker(work / "model").score(ft)
    report = evaluate.compare(ft, scores, qt, k=20)
    report.to_csv(work / f"report_{args.eval_split}.csv")
    (work / "summary.json").write_text(json.dumps(info, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(report)


if __name__ == "__main__":
    main()
