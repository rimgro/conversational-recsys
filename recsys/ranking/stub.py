"""ЗАГЛУШКА ранкера: оставляет порядок RRF (на схеме: «top-20 первых»)."""
from __future__ import annotations

import pandas as pd

from recsys.ranking.base import BaseRanker
from recsys.schemas import Context


class StubRanker(BaseRanker):
    def rank(self, features: pd.DataFrame, ctx: Context) -> pd.DataFrame:
        out = features.copy()
        out["rank_score"] = out["rrf_score"] if len(out) else []
        return out.sort_values("rank_score", ascending=False, kind="stable").reset_index(drop=True)
