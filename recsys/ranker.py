"""Шаг 4б: ранкер — скор каждого отобранного кандидата по его признакам (features.py).

  catboost  CatBoostRanker (YetiRank), обучается на части ranker_train, early stopping — по valid
            (python -m recsys train); группа = запрос, метка 1 у цели; запросы, где цели нет среди
            кандидатов, в обучение не идут (в метриках они остаются и дают 0)
  stub      без обучения: порядок отбора (скор политики RRF)

Модель — папка ranker.model_dir (по умолчанию runs/<прогон>/ranker): model.cbm, features.json.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from recsys.features import CATEGORICAL


class BaseRanker:
    kind = ""

    def score(self, features: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class StubRanker(BaseRanker):
    """Порядок отбора: скор политики (признак selection_rrf_score)."""
    kind = "stub"

    def score(self, features: pd.DataFrame) -> np.ndarray:
        if "selection_rrf_score" not in features:
            raise ValueError("ranker.type: stub нужна группа признаков selection")
        return features["selection_rrf_score"].to_numpy(dtype=np.float64)


class CatBoostRanker(BaseRanker):
    kind = "catboost"

    def __init__(self, model=None, features: Optional[List[str]] = None, threads: int = 6):
        self.model = model
        self.features = features or []
        self.threads = threads
        self.importance: Optional[pd.DataFrame] = None   # важность признаков (после fit)

    def score(self, features: pd.DataFrame) -> np.ndarray:
        missing = [f for f in self.features if f not in features]
        if missing:
            raise ValueError(f"у модели есть признаки, которых нет сейчас: {missing} "
                             f"(другие кандгены или группы признаков, чем при обучении)")
        return self.model.predict(features[self.features], thread_count=self.threads)

    @classmethod
    def fit(cls, train: Tuple[pd.DataFrame, np.ndarray, np.ndarray], valid: Tuple[pd.DataFrame, np.ndarray, np.ndarray],
            params: Dict[str, Any], threads: int = 6, train_dir: Optional[str] = None,
            verbose: int = 50) -> "CatBoostRanker":
        """train / valid — (признаки, метки, номер группы); группы идут подряд."""
        from catboost import CatBoostRanker as CB, Pool
        params = dict(params)
        border_count = params.pop("border_count", 128)
        early_stopping = params.pop("early_stopping_rounds", 50)
        names = list(train[0].columns)
        cats = [c for c in CATEGORICAL if c in names]
        pools = []
        for x, y, g in (train, valid):
            pools.append(Pool(x[names], label=y, group_id=g, cat_features=cats, feature_names=names))
        pools[0].quantize(border_count=border_count, random_seed=params.get("random_seed", 42))
        model = CB(thread_count=threads, train_dir=train_dir, **params)
        model.fit(pools[0], eval_set=pools[1], early_stopping_rounds=early_stopping, use_best_model=True,
                  verbose=verbose)
        ranker = cls(model, names, threads)
        # для ранжирования важность считается только по данным: здесь, пока обучающий пул в памяти
        ranker.importance = (pd.DataFrame({"feature": names, "importance": model.get_feature_importance(
            pools[0], type="PredictionValuesChange")}).sort_values("importance", ascending=False)
            .reset_index(drop=True))
        return ranker

    def save(self, directory: str, extra: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(directory, exist_ok=True)
        self.model.save_model(os.path.join(directory, "model.cbm"))
        with open(os.path.join(directory, "features.json"), "w", encoding="utf8") as f:
            json.dump({"features": self.features, **(extra or {})}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: str, threads: int = 6) -> "CatBoostRanker":
        from catboost import CatBoostRanker as CB
        path = os.path.join(directory, "model.cbm")
        if not os.path.exists(path):
            raise FileNotFoundError(f"нет модели ранкера {path}: обучить — python -m recsys train "
                                    f"(или ranker.type: stub)")
        model = CB()
        model.load_model(path)
        with open(os.path.join(directory, "features.json"), encoding="utf8") as f:
            meta = json.load(f)
        return cls(model, meta["features"], threads)


def model_dir(cfg: Dict[str, Any]) -> str:
    rcfg, run = cfg["ranker"], cfg.get("run", {})
    return rcfg.get("model_dir") or os.path.join(run.get("dir", "runs"), run.get("name", "baseline"), "ranker")


def load_ranker(cfg: Dict[str, Any]) -> BaseRanker:
    kind = cfg["ranker"].get("type", "catboost")
    if kind == "stub":
        return StubRanker()
    if kind == "catboost":
        return CatBoostRanker.load(model_dir(cfg), threads=cfg["ranker"].get("threads", 6))
    raise ValueError(f"неизвестный ranker.type: {kind} (catboost | stub)")
