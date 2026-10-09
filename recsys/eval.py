"""Офлайн-оценка как в датасете: nDCG@20 с одной целью (1/log2(rank+1), если цель в топ-20), по query_type;
плюс hit / mrr и recall каждого источника кандидатов. Для similar_to перед подсчётом из выдачи убираются
сам трек-образец (exclude_ids) и все треки его артиста (exclude_artist).
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from recsys.config import deep_update
from recsys.data.catalog import Catalog
from recsys.llm import BaseLLM
from recsys.pipeline import Pipeline
from recsys.schemas import Request


def hit_rate_at_k(rec: Sequence[str], targets: Iterable[str], k: int) -> float:
    t = set(targets)
    return float(any(r in t for r in rec[:k])) if t else 0.0


def recall_at_k(rec: Sequence[str], targets: Iterable[str], k: int) -> float:
    t = set(targets)
    return len(t & set(rec[:k])) / len(t) if t else 0.0


def ndcg_at_k(rec: Sequence[str], targets: Iterable[str], k: int) -> float:
    t = set(targets)
    if not t:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(rec[:k]) if r in t)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(t), k)))
    return dcg / idcg


def mrr_at_k(rec: Sequence[str], targets: Iterable[str], k: int) -> float:
    t = set(targets)
    for i, r in enumerate(rec[:k]):
        if r in t:
            return 1.0 / (i + 1)
    return 0.0


def scored_ids(track_ids: Sequence[str], request: Request, catalog: Catalog) -> List[str]:
    """Выдача так, как её видит метрика: без exclude_ids и треков exclude_artist (similar_to)."""
    banned = set(request.meta.get("exclude_ids") or [])
    artist = (request.meta.get("exclude_artist") or "").lower()
    return [t for t in track_ids if t not in banned and not (artist and catalog.artist(t).lower() == artist)]


ID_COLUMNS = ["request_id", "query_type", "is_new"]
OUTPUT_COLUMNS = ["top", "response"]


def evaluate(pipeline: Pipeline, requests: List[Request], k: int = 20, verbose: bool = True,
             outputs: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """-> (метрики по запросам, средние). Берутся только запросы с target_ids. Нужен ranker.top_k >= k.
    outputs=True — ещё колонки top (выдача, top-k id) и response (текст ответа): для файла сабмита."""
    rows: List[Dict[str, Any]] = []
    reqs = [r for r in requests if r.target_ids]
    t0 = time.time()
    for i, req in enumerate(reqs):
        resp = pipeline.run(req, debug=True)
        rec = scored_ids(resp.track_ids, req, pipeline.catalog)
        row: Dict[str, Any] = {
            "request_id": req.request_id,
            "query_type": req.meta.get("query_type"),
            "is_new": req.meta.get("is_new"),
            f"ndcg@{k}": ndcg_at_k(rec, req.target_ids, k),
            f"hit@{k}": hit_rate_at_k(rec, req.target_ids, k),
            f"mrr@{k}": mrr_at_k(rec, req.target_ids, k),
        }
        fused_ids = [f.track_id for f in resp.debug["fused"]]
        row["recall@fused"] = recall_at_k(fused_ids, req.target_ids, len(fused_ids))
        for src, cands in resp.debug["candidates"].items():
            ids = [c.track_id for c in cands]
            row[f"recall@{src}"] = recall_at_k(ids, req.target_ids, len(ids))
            # место цели в списке источника (NaN — не нашёл) и сколько кандидатов он отдал
            row[f"rank:{src}"] = next((c.rank for c in cands if c.track_id in req.target_ids), float("nan"))
            row[f"n:{src}"] = len(cands)
        if outputs:
            row.update(top=resp.track_ids[:k], response=resp.text)
        rows.append(row)
        if verbose and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(reqs)}  {time.time() - t0:.1f} c")
    df = pd.DataFrame(rows)
    metrics = df[[c for c in df.columns if "@" in c]] if len(df) else df
    return df, metrics.mean(numeric_only=True) if len(df) else pd.Series(dtype=float)


STAGES = ["rrf", "filtered", "fused"]   # все источники вместе: до фильтров, после, первые top_n (вход ранкера)
CANDGEN_KS = (20, 50, 100, 200)


def evaluate_candidates(pipeline: Pipeline, requests: List[Request], k: int = 20) -> pd.DataFrame:
    """Шаги 1–3 без ранкера: строка = запрос × список (каждый источник и этапы STAGES).

    rank — место цели в списке так, как его видит метрика (без образца и артиста similar_to), 1 = первое;
    NaN — цели нет. ndcg@k — если бы выдачей был этот список как есть. filter — какой фильтр выкинул цель
    (только у строки filtered: banned / exclude_artist / exclude_tag / ... или constraint при fusion.constraints.hard;
    пусто — не выкидывали или её не нашли).
    """
    from recsys.fusion import Filters

    rows: List[Dict[str, Any]] = []
    for req in requests:
        if not req.target_ids:
            continue
        r = pipeline.retrieve(req)
        targets = set(req.target_ids)
        base = {"request_id": req.request_id, "query_type": req.meta.get("query_type"), "is_new": req.meta.get("is_new")}
        lists = {src: [c.track_id for c in cands] for src, cands in r.candidates.items()}
        lists.update(rrf=[f.track_id for f in r.merged], filtered=[f.track_id for f in r.filtered],
                     fused=[f.track_id for f in r.fused])
        reason = None
        if any(t in targets for t in lists["rrf"]) and not any(t in targets for t in lists["filtered"]):
            f = Filters(r.ctx, pipeline.catalog, pipeline.cfg.get("fusion", {}).get("exclude_top_tags", 0))
            reason = next((f.reason(t) or "constraint" for t in req.target_ids if t in lists["rrf"]), None)
        for name, ids in lists.items():
            rec = scored_ids(ids, req, pipeline.catalog)
            rank = next((i for i, t in enumerate(rec, start=1) if t in targets), float("nan"))
            rows.append({**base, "list": name, "is_source": name not in STAGES, "n": len(ids), "rank": rank,
                         f"ndcg@{k}": ndcg_at_k(rec, targets, k),
                         "filter": reason if name == "filtered" else None})
    return pd.DataFrame(rows)


def candidates_summary(per: pd.DataFrame, by: Optional[str] = None, ks: Sequence[int] = CANDGEN_KS,
                       k: int = 20) -> pd.DataFrame:
    """Сводка evaluate_candidates. by=None: строка = список (источник / этап); иначе строка = (by, список).

    recall@N — цель в первых N списка; recall — где угодно в списке; only_this — цель нашёл только этот
    источник; median_rank — место цели, когда нашли; empty — доля запросов, где список пуст.
    """
    df = per.assign(found=per["rank"].notna())
    for n in ks:
        df[f"recall@{n}"] = df["rank"] <= n
    src = df[df["is_source"]]
    n_found = src.groupby("request_id")["found"].transform("sum")
    df["only_this"] = float("nan")
    df.loc[src.index, "only_this"] = (src["found"] & (n_found == 1)).astype(float)
    df["empty"] = df["n"] == 0
    keys = ([by] if by else []) + ["list"]
    agg = df.groupby(keys, sort=False).agg(
        n_requests=("request_id", "size"),
        recall=("found", "mean"),
        **{f"recall@{n}": (f"recall@{n}", "mean") for n in ks},
        **{f"ndcg@{k}": (f"ndcg@{k}", "mean")},
        only_this=("only_this", "mean"),
        median_rank=("rank", "median"),
        mean_candidates=("n", "mean"),
        empty=("empty", "mean"),
    )
    return agg


def filter_losses(per: pd.DataFrame, by: str = "query_type") -> pd.DataFrame:
    """Доля запросов, где цель была среди кандидатов (rrf), но её выкинул фильтр, — по фильтрам и группам."""
    f = per[per["list"] == "filtered"]
    dummies = pd.get_dummies(f["filter"]).astype(float)  # пустые (None) не дают колонки
    return pd.concat([dummies.groupby(f[by]).mean(), dummies.mean().to_frame("ALL").T])


def sources_summary(per_request: pd.DataFrame) -> pd.DataFrame:
    """Строка = источник кандидатов. recall — доля запросов, где он нашёл цель; only_this — где цель нашёл
    только он (уникальный вклад); median_rank / share_top20 — где в его списке цель, когда найдена;
    mean_candidates / share_empty — сколько кандидатов отдаёт и как часто ничего."""
    sources = [c.split("@", 1)[1] for c in per_request.columns if c.startswith("recall@") and c != "recall@fused"]
    found = per_request[[f"recall@{s}" for s in sources]].fillna(0).to_numpy() > 0
    rows = {}
    for i, s in enumerate(sources):
        rank = per_request[f"rank:{s}"]
        n = per_request[f"n:{s}"]
        rows[s] = {
            "recall": found[:, i].mean(),
            "only_this": (found[:, i] & (found.sum(axis=1) == 1)).mean(),
            "median_rank": rank.median(),
            "share_top20": (rank <= 20).sum() / max(int(found[:, i].sum()), 1),
            "mean_candidates": n.mean(),
            "share_empty": (n == 0).mean(),
        }
    out = pd.DataFrame(rows).T.sort_values("recall", ascending=False)
    out.loc["fused (все вместе)", "recall"] = per_request["recall@fused"].mean()
    return out


def sources_by(per_request: pd.DataFrame, by: str = "query_type") -> pd.DataFrame:
    """recall каждого источника по группам: строка = группа (тип запроса), колонка = источник."""
    cols = [c for c in per_request.columns if c.startswith("recall@")]
    out = per_request.groupby(by)[cols].mean()
    out.columns = [c.split("@", 1)[1] for c in cols]
    out.insert(0, "n", per_request.groupby(by).size())
    return out


def metrics_by(per_request: pd.DataFrame, by: str = "query_type") -> pd.DataFrame:
    """Средние метрики по группам (query_type / is_new) + число запросов в группе."""
    cols = [c for c in per_request.columns if "@" in c]
    out = per_request.groupby(by)[cols].mean()
    out.insert(0, "n", per_request.groupby(by).size())
    return out


def compare_configs(base_cfg: Dict[str, Any], variants: Dict[str, Dict[str, Any]], catalog: Catalog,
                    requests: List[Request], k: int = 20, llm: Optional[BaseLLM] = None,
                    by: Optional[str] = None, metric: Optional[str] = None) -> pd.DataFrame:
    """Метрики для нескольких вариантов конфига (BM25-индекс по тегам строится один раз).

    by=None: строка = вариант, колонки = средние метрики.
    by="query_type": строка = группа, колонки = варианты, значение = metric (по умолчанию ndcg@k).
    """
    index = None
    out = {}
    metric = metric or f"ndcg@{k}"
    for name, overrides in variants.items():
        cfg = deep_update(base_cfg, overrides)
        pipe = Pipeline.from_config(cfg, catalog, llm=llm, bm25_index=index)
        index = index or pipe.bm25_index
        per, mean = evaluate(pipe, requests, k=k, verbose=False)
        if by is None:
            out[name] = mean
        else:
            col = metrics_by(per, by)[metric]
            col.loc["ALL"] = mean[metric]
            out[name] = col
    return pd.DataFrame(out).T if by is None else pd.DataFrame(out)
