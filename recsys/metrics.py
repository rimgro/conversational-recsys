"""Метрики по шагам: каждый кандген, их объединение, отбор, ранкер — на одних и тех же запросах.

Метрика датасета — nDCG@20 с одной целью: 1 / log2(место + 1), если цель в первых 20. Для similar_to из выдачи
сначала убираются трек-образец (exclude_m4a_id) и все треки его артиста (exclude_artist) — так считает датасет;
это знание из qrels, в выдаче его нет (там фильтр selection.filters: exclude_artist по разбору запроса).

Списки (колонка list):
  <кандген>   список кандгена как есть
  union       все кандидаты всех кандгенов, по RRF без весов (до фильтров)
  selection   отобранные для ранкера (selection.limit), по скору отбора — это и есть выдача stub-ранкера

query_type здесь — только для разбивки отчёта по типам; модель его не видит.
  ranker      итоговая выдача (ranker.top_k)
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from recsys.data.parts import normalize_query
from recsys.pipeline import Result
from recsys.schemas import Request

STAGES = ["union", "selection", "ranker"]


class TargetView:
    """Как метрика видит списки запроса: позиции цели и исключённые для similar_to."""

    def __init__(self, catalog):
        self.catalog = catalog
        self.artists = np.array([normalize_query(a) for a in catalog.text("artist")], dtype=object)

    def target(self, request: Request) -> Optional[int]:
        return self.catalog.pos(request.target_ids[0]) if request.target_ids else None

    def excluded(self, request: Request, ids: np.ndarray) -> np.ndarray:
        """True — трек не учитывается метрикой (similar_to: образец и треки его артиста)."""
        out = np.zeros(len(ids), dtype=bool)
        ex = self.catalog.positions(request.meta.get("exclude_ids") or [])
        if len(ex):
            out |= np.isin(ids, ex)
        artist = normalize_query(request.meta.get("exclude_artist") or "")
        if artist:
            out |= self.artists[ids] == artist
        return out

    def rank(self, request: Request, ids: np.ndarray) -> float:
        """Место цели в списке (1 — первое) после исключений; NaN — цели нет."""
        t = self.target(request)
        if t is None or not len(ids):
            return float("nan")
        ids = ids[~self.excluded(request, ids)]
        hit = np.flatnonzero(ids == t)
        return float(hit[0] + 1) if len(hit) else float("nan")


def discovery(ids: np.ndarray, history: np.ndarray, k: int = 20) -> float:
    """Доля новых треков в первых k с весом 1 / log2(r + 1): Σ w_r·new_r / Σ w_r (история — всё до начала окна).
    Знаменатель — по позициям, которые есть в списке (короче k — нормируется на его длину)."""
    top = ids[:k]
    if not len(top):
        return float("nan")
    w = 1.0 / np.log2(np.arange(2, len(top) + 2))
    return float((w * ~np.isin(top, history)).sum() / w.sum())


def per_list(results: List[Result], selector, k: int = 20) -> pd.DataFrame:
    """Строка = запрос × список: место цели (rank), длина списка (n), nDCG@k, Discovery@k (новое для пользователя
    в первых k), каким фильтром выкинута цель."""
    view = TargetView(selector.catalog)
    rows = []
    for r in results:
        req = r.ctx.request
        if not req.target_ids:
            continue
        history = selector.catalog.positions(h.track_id for h in req.history)
        base = {"request_id": req.request_id, "query_type": req.meta.get("query_type"),
                "is_new": req.meta.get("is_new")}
        lists = {name: c.ids for name, c in r.lists.items()}
        union = np.unique(np.concatenate([c for c in lists.values() if len(c)] or [np.zeros(0, np.int32)]))
        rrf = selector.scores(r.lists, {name: 1.0 for name in r.lists}, selector.union_k)
        lists["union"] = union[np.lexsort((union, -rrf[union]))]
        lists["selection"] = r.selection.ids[np.lexsort((r.selection.ids, -r.selection.score))]
        if len(r.scores) or not len(r.selection.ids):
            lists["ranker"] = r.top
        target = view.target(req)
        removed_by = next((f for f, ids in r.selection.removed.items() if target in ids), None)
        for name, ids in lists.items():
            rank = view.rank(req, ids)
            rows.append({**base, "list": name, "n": len(ids), "rank": rank,
                         f"ndcg@{k}": 1.0 / np.log2(rank + 1) if rank <= k else 0.0,
                         f"discovery@{k}": discovery(ids, history, k),
                         "filtered_by": removed_by if name == "selection" else None})
    return pd.DataFrame(rows)


def summary(per: pd.DataFrame, recall_at: Sequence[int], k: int = 20, by: Optional[str] = None) -> pd.DataFrame:
    """Строка = список (или (by, список)): recall@N — цель в первых N; recall — где угодно в списке;
    only_this — цель нашёл только этот кандген; nDCG@k, hit@k, mrr@k — если бы выдачей был этот список."""
    df = per.assign(found=per["rank"].notna())
    for n in recall_at:
        df[f"recall@{n}"] = df["rank"] <= n
    df[f"hit@{k}"] = df["rank"] <= k
    df[f"mrr@{k}"] = np.where(df["rank"] <= k, 1.0 / df["rank"], 0.0)
    src = df[~df["list"].isin(STAGES)]
    n_found = src.groupby("request_id")["found"].transform("sum")
    df["only_this"] = np.nan
    df.loc[src.index, "only_this"] = (src["found"] & (n_found == 1)).astype(float)
    df["empty"] = df["n"] == 0
    keys = ([by] if by else []) + ["list"]
    return df.groupby(keys, sort=False).agg(
        queries=("request_id", "size"),
        **{f"ndcg@{k}": (f"ndcg@{k}", "mean"), f"hit@{k}": (f"hit@{k}", "mean"), f"mrr@{k}": (f"mrr@{k}", "mean"),
           f"discovery@{k}": (f"discovery@{k}", "mean")},
        **{f"recall@{n}": (f"recall@{n}", "mean") for n in recall_at},
        recall=("found", "mean"),
        only_this=("only_this", "mean"),
        median_rank=("rank", "median"),
        mean_len=("n", "mean"),
        empty=("empty", "mean"),
    )


def by_type(per: pd.DataFrame, metric: str, k: int = 20, recall_at: Sequence[int] = (20,)) -> pd.DataFrame:
    """Строка = тип запроса (+ ALL), колонка = список, значение = metric."""
    s = summary(per, recall_at, k, by="query_type")[metric].unstack("list")
    order = list(dict.fromkeys(per["list"]))
    s = s.reindex(columns=order)
    s.loc["ALL"] = summary(per, recall_at, k)[metric].reindex(order)
    s.insert(0, "queries", per[per["list"] == order[0]].groupby("query_type").size())
    s.loc["ALL", "queries"] = per["request_id"].nunique()
    return s


def filter_losses(per: pd.DataFrame) -> pd.Series:
    """Доля запросов, где цель была среди кандидатов, но её выкинул фильтр отбора."""
    sel = per[per["list"] == "selection"]
    return sel["filtered_by"].value_counts() / max(len(sel), 1)
