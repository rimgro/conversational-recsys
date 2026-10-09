"""Инференс: загруженная модель + признаки пула -> скоры. Встраивается в пайплайн как шаг ранжирования."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


class LGBMRanker:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.booster = lgb.Booster(model_str=(model_dir / "model.txt").read_text(encoding="utf-8"))
        meta = json.loads((model_dir / "features.json").read_text(encoding="utf-8"))
        self.features = meta["features"]
        self.categorical = meta["categorical"]

    def score(self, feats: pd.DataFrame) -> np.ndarray:
        x = feats[self.features].copy()
        for col, cats in self.categorical.items():
            x[col] = pd.Categorical(x[col], categories=cats)
        return self.booster.predict(x)

    def rank(self, feats: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
        """Признаки одного или нескольких запросов -> top_k по каждому qid, по убыванию rank_score."""
        out = feats.assign(rank_score=self.score(feats))
        out = out.sort_values(["qid", "rank_score"], ascending=[True, False], kind="stable")
        return out.groupby("qid", sort=False).head(top_k).reset_index(drop=True)
