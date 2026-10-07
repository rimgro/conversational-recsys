"""Линейная комбинация признаков с весами из конфига. Бейзлайн до CatBoost."""
from __future__ import annotations

from typing import Dict

import pandas as pd

from recsys.ranking.base import BaseRanker
from recsys.schemas import Context

DEFAULT_WEIGHTS: Dict[str, float] = {
    "rrf_score": 30.0,          # RRF ~ 0.01–0.05
    "query_tag_frac": 1.0,
    "user_tag_hits": 0.1,
    "profile_tag_score": 0.5,
    "artist_seed": 1.0,
    "artist_listened": 0.2,
    "log_popularity": 0.03,
    "n_sources": 0.05,
}


class HeuristicRanker(BaseRanker):
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    def rank(self, features: pd.DataFrame, ctx: Context) -> pd.DataFrame:
        out = features.copy()
        score = pd.Series(0.0, index=out.index)
        for f, w in self.weights.items():
            if f in out:
                score += w * out[f].astype(float)
        out["rank_score"] = score
        return out.sort_values("rank_score", ascending=False, kind="stable").reset_index(drop=True)
