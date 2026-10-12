"""Freeze held-out query/profile inputs without re-encoding track features."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from recsys.two_tower_qwen.config import TrainConfig
from recsys.two_tower_qwen.data import balanced_validation
from recsys.two_tower_qwen.features import padded_histories, recent_history
from recsys.two_tower_qwen.metrics import CatalogScorer
from recsys.track_tower import text_value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--previous-validation", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(**json.loads((a.bundle / "config.json").read_text())["training"])
    if cfg.max_users is not None or cfg.max_queries_per_user is not None:
        raise RuntimeError("This full-data evaluation requires the unbounded training split")
    catalog = json.loads((a.bundle / "catalog.json").read_text())
    pairs = pq.read_table(a.data / "train_queries.parquet").to_pandas().merge(
        pq.read_table(a.data / "train_qrels.parquet").to_pandas(), on="query_id", validate="one_to_one")
    pairs["query"] = pairs["query"].map(text_value)
    users = np.unique(pairs.user_id)
    heldout = set(np.random.default_rng(cfg.seed).choice(users,
        max(1, int(len(users) * cfg.validation_user_fraction)), replace=False))
    all_rows = np.flatnonzero(pairs.user_id.isin(heldout).to_numpy())
    selected = balanced_validation(pairs, all_rows, cfg.validation_per_type, cfg.seed)
    selected_ids = set(pairs.iloc[selected].query_id)
    previous_ids = set(pd.read_parquet(a.previous_validation, columns=["query_id"]).query_id)
    if selected_ids != previous_ids:
        raise RuntimeError("Reconstructed checkpoint-selection split differs from the saved validation")
    scorer = CatalogScorer(catalog, set(pairs.query_type))
    rows, listened_flat, listened_offsets = [], [], [0]
    for batch in pq.ParquetFile(a.data / "train.parquet").iter_batches(batch_size=32,
            columns=["user_id", "history", "history_ts", "history_len"]):
        for row in batch.to_pylist():
            if row["user_id"] not in heldout:
                continue
            history, timestamps = row["history"], row["history_ts"]
            if len(history) != len(timestamps) or len(history) != row["history_len"]:
                raise ValueError("Misaligned user history")
            if timestamps and (max(timestamps) >= 1579564800 or any(x > y for x, y in zip(timestamps, timestamps[1:]))):
                raise ValueError("History leaks future events")
            groups = sorted({int(scorer.group_ids[scorer.lookup[str(t)]])
                             for t in history if str(t) in scorer.lookup})
            rows.append((row["user_id"], recent_history(history, scorer.lookup, cfg.history_events)))
            listened_flat.extend(groups)
            listened_offsets.append(len(listened_flat))
    if len(rows) != len(heldout) or len({u for u, _ in rows}) != len(heldout):
        raise RuntimeError("Missing or duplicate held-out user profiles")
    histories, masks = padded_histories([h for _, h in rows])
    user_lookup = {uid: i for i, (uid, _) in enumerate(rows)}
    evaluation = pairs.iloc[all_rows].copy()
    evaluation["profile_index"] = evaluation.user_id.map(user_lookup).astype(np.int64)
    evaluation["selection_validation"] = evaluation.query_id.isin(selected_ids)
    # Run the previously scored, balanced subset first for a smoke estimate.
    evaluation = evaluation.sort_values("selection_validation", ascending=False, kind="stable").reset_index(drop=True)
    evaluation.to_parquet(a.output / "queries.parquet", index=False)
    np.savez_compressed(a.output / "profiles.npz", histories=histories, masks=masks,
        listened_groups=np.asarray(listened_flat, dtype=np.int64),
        listened_offsets=np.asarray(listened_offsets, dtype=np.int64))
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in [a.output / "queries.parquet", a.output / "profiles.npz"]}
    manifest = {"queries": len(evaluation), "selection_validation_queries": len(selected),
        "additional_holdout_queries": len(evaluation) - len(selected), "heldout_users": len(heldout),
        "query_types": sorted(set(evaluation.query_type)), "catalog_tracks": len(catalog),
        "sha256": files, "split_matches_training_validation": True,
        "heldout_is_internal_validation_not_an_independent_official_test": True}
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
