"""LightGBM LambdaRank: группа = запрос, label 1 у таргета, nDCG@20.

Гиперпараметры консервативные (как у swyoo): сильная регуляризация против сдвига train -> test.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from ranker.features import CATEGORICAL, FEATURES

PARAMS: Dict = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [20],
    "lambdarank_truncation_level": 20,
    "learning_rate": 0.02,
    "num_leaves": 15,
    "max_depth": 5,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "cat_l2": 10.0,
    "cat_smooth": 10.0,
    "seed": 42,
    "verbosity": -1,
}
NUM_BOOST_ROUND = 5000
EARLY_STOPPING = 200


def drop_groups_without_positive(df: pd.DataFrame, group: str = "qid") -> pd.DataFrame:
    has_pos = df.groupby(group, sort=False)["label"].transform("max") > 0
    return df[has_pos.to_numpy()]


def _dataset(df: pd.DataFrame, features: List[str], categorical: List[str], group: str,
             reference: Optional[lgb.Dataset] = None) -> lgb.Dataset:
    df = df.sort_values(group, kind="stable")
    groups = df.groupby(group, sort=False).size().to_numpy()
    return lgb.Dataset(df[features], label=df["label"].to_numpy(), group=groups,
                       categorical_feature=categorical, reference=reference, free_raw_data=True)


def train(train_df: pd.DataFrame, valid_df: pd.DataFrame, params: Optional[Dict] = None,
          num_boost_round: int = NUM_BOOST_ROUND, early_stopping: int = EARLY_STOPPING,
          log_every: int = 100, features: Optional[List[str]] = None,
          categorical: Optional[Dict[str, List[str]]] = None, group: str = "qid") -> Tuple[lgb.Booster, Dict]:
    """train_df / valid_df: строки (запрос, кандидат) с label и признаками; группы без позитива выкидываются здесь.
    categorical: {колонка: список категорий} — колонки должны быть pandas.Categorical с этими категориями."""
    features = features or FEATURES
    categorical = CATEGORICAL if categorical is None else categorical
    train_df = drop_groups_without_positive(train_df, group)
    valid_df = drop_groups_without_positive(valid_df, group)
    if train_df.empty or valid_df.empty:
        raise ValueError(f"нет групп с таргетом в пуле: train={train_df[group].nunique()}, "
                         f"valid={valid_df[group].nunique()} (мало данных или пул не находит таргеты)")
    dtrain = _dataset(train_df, features, list(categorical), group)
    dvalid = _dataset(valid_df, features, list(categorical), group, reference=dtrain)
    booster = lgb.train({**PARAMS, **(params or {})}, dtrain, num_boost_round=num_boost_round,
                        valid_sets=[dvalid], valid_names=["valid"],
                        callbacks=[lgb.early_stopping(early_stopping, verbose=False),
                                   lgb.log_evaluation(log_every)])
    info = {"best_iteration": booster.best_iteration,
            "valid_ndcg@20": float(booster.best_score["valid"]["ndcg@20"]),
            "train_groups": int(train_df[group].nunique()), "valid_groups": int(valid_df[group].nunique())}
    return booster, info


def feature_importance(booster: lgb.Booster) -> pd.DataFrame:
    return pd.DataFrame({"feature": booster.feature_name(),
                         "gain": booster.feature_importance("gain"),
                         "split": booster.feature_importance("split")}).sort_values("gain", ascending=False)


def save(booster: lgb.Booster, path: str | Path, info: Optional[Dict] = None,
         features: Optional[List[str]] = None, categorical: Optional[Dict[str, List[str]]] = None) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    # через строку: LightGBM на Windows не открывает пути с кириллицей
    (path / "model.txt").write_text(booster.model_to_string(num_iteration=booster.best_iteration or None),
                                    encoding="utf-8")
    meta = {"features": features or FEATURES, "categorical": CATEGORICAL if categorical is None else categorical,
            "params": PARAMS, **(info or {})}
    (path / "features.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    feature_importance(booster).to_csv(path / "importance.csv", index=False)
