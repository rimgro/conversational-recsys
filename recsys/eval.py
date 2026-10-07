"""Офлайн-оценка: метрики выдачи + recall каждого источника кандидатов.
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


def evaluate(pipeline: Pipeline, requests: List[Request], k: int = 10,
             verbose: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """-> (метрики по запросам, средние). Берутся только запросы с target_ids."""
    rows: List[Dict[str, Any]] = []
    reqs = [r for r in requests if r.target_ids]
    t0 = time.time()
    for i, req in enumerate(reqs):
        resp = pipeline.run(req, debug=True)
        rec = resp.track_ids
        row: Dict[str, Any] = {
            "request_id": req.request_id,
            f"hit@{k}": hit_rate_at_k(rec, req.target_ids, k),
            f"recall@{k}": recall_at_k(rec, req.target_ids, k),
            f"ndcg@{k}": ndcg_at_k(rec, req.target_ids, k),
            f"mrr@{k}": mrr_at_k(rec, req.target_ids, k),
        }
        fused_ids = [f.track_id for f in resp.debug["fused"]]
        row["recall@fused"] = recall_at_k(fused_ids, req.target_ids, len(fused_ids))
        for src, cands in resp.debug["candidates"].items():
            ids = [c.track_id for c in cands]
            row[f"recall@{src}"] = recall_at_k(ids, req.target_ids, len(ids))
        rows.append(row)
        if verbose and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(reqs)}  {time.time() - t0:.1f} c")
    df = pd.DataFrame(rows)
    return df, df.drop(columns=["request_id"]).mean(numeric_only=True) if len(df) else pd.Series(dtype=float)


def compare_configs(base_cfg: Dict[str, Any], variants: Dict[str, Dict[str, Any]], catalog: Catalog,
                    requests: List[Request], k: int = 10, llm: Optional[BaseLLM] = None) -> pd.DataFrame:
    """Таблица метрик для нескольких вариантов конфига (BM25-индекс строится один раз)."""
    index = None
    out = {}
    for name, overrides in variants.items():
        cfg = deep_update(base_cfg, overrides)
        pipe = Pipeline.from_config(cfg, catalog, llm=llm, bm25_index=index)
        index = index or pipe.bm25_index
        _, mean = evaluate(pipe, requests, k=k, verbose=False)
        out[name] = mean
    return pd.DataFrame(out).T
