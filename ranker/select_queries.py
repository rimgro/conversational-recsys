"""Запросы train для обучения ранкера: детерминированный сэмпл, доли query_type как в train,
валидация — отдельные пользователи (по hash(user_id)).

    python -m ranker.select_queries --data-dir ../new_ds_v1/music4all_crs --n 100000 --out ranker_train_queries.csv
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def unit_hash(values, salt: str) -> np.ndarray:
    return np.array([int(hashlib.md5(f"{salt}|{v}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF for v in values])


def select(queries: pd.DataFrame, n: int, valid_frac: float = 0.1) -> pd.DataFrame:
    """Внутри каждого query_type берём одну и ту же долю n/len по hash(query_id): доли типов сохраняются,
    результат не зависит от порядка строк и при большем n расширяется (старые id остаются)."""
    frac = min(1.0, n / len(queries))
    picked = queries[unit_hash(queries["query_id"], "ranker-train") < frac].copy()
    picked["part"] = np.where(unit_hash(picked["user_id"], "ranker-valid") < valid_frac, "valid", "train")
    return picked.sort_values(["part", "user_id", "query_id"]).reset_index(drop=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--valid-frac", type=float, default=0.1)
    ap.add_argument("--out", default="ranker_train_queries.csv")
    args = ap.parse_args(argv)

    q = pd.read_parquet(Path(args.data_dir) / "train_queries.parquet",
                        columns=["query_id", "user_id", "query_type", "source"])
    picked = select(q, args.n, args.valid_frac)
    picked.to_csv(args.out, index=False)

    share = pd.DataFrame({"all": q["query_type"].value_counts(normalize=True),
                          "picked": picked["query_type"].value_counts(normalize=True)}).round(4)
    print(f"запросов {len(picked)} из {len(q)}; пользователей {picked['user_id'].nunique()}; "
          f"valid: {(picked['part'] == 'valid').sum()} запросов")
    print(share.to_string())


if __name__ == "__main__":
    main()
