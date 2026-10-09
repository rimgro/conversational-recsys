"""Загрузка Music4All-CRS для ранкера: каталог, пользователи, история, запросы.

Сплит читается из data_dir: либо один файл `{split}.parquet`, либо шарды `{split}-*-of-*.parquet`
(в любой подпапке). Всё переводится в плоские таблицы с целочисленными индексами треков и
пользователей, чтобы признаки на десятках миллионов строк считались через numpy, а не merge.

  tracks   track_idx, track_id, artist, album, release_year, lang, popularity, tags (list)
  users    user_id, age, age_bucket, gender
  history  user_id, track_idx, count, last_ts       (агрегат прослушиваний до окна)
  queries  qid, user_id, target_idx, query, query_family, is_new
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# «сейчас» для recency и возраста = начало окна сплита
WINDOW_START = {
    "train": int(datetime(2020, 1, 21, tzinfo=timezone.utc).timestamp()),
    "test_public": int(datetime(2020, 2, 20, tzinfo=timezone.utc).timestamp()),
    "test_private": int(datetime(2020, 2, 20, tzinfo=timezone.utc).timestamp()),
}
WINDOW_YEAR = 2020

QUERY_FAMILIES = ["exact", "lyrics_recall", "lyrics_theme", "genre", "mood", "situation", "era_region",
                  "audio_attributes", "complex", "negative_constraint", "vague_recall", "unknown"]
AGE_BUCKETS = ["10s", "20s", "30s", "40s", "50+", "unknown"]
GENDERS = ["m", "f", "unknown"]

_TRACK_COLUMNS = ["m4a_id", "m4a_artist", "m4a_album", "release_year", "lang", "spotify_popularity",
                  "m4a_genres_full", "m4a_tags_full", "lastfm_tag_weights"]


def find_files(data_dir: str | Path, name: str) -> List[Path]:
    root = Path(data_dir)
    shards = sorted(root.rglob(f"{name}-*-of-*.parquet"))
    if shards:
        return shards
    single = root / f"{name}.parquet"
    if single.exists():
        return [single]
    raise FileNotFoundError(f"нет {name}.parquet и шардов {name}-*-of-*.parquet в {root}")


# ---------------------------------------------------------------- каталог

def _csv(raw) -> List[str]:
    return [s.strip().lower() for s in raw.split(",") if s.strip()] if isinstance(raw, str) else []


def _tag_keys(raw) -> List[str]:
    if not isinstance(raw, str) or not raw:
        return []
    try:
        return [str(t).strip().lower() for t in json.loads(raw)]
    except ValueError:
        return []


def load_tracks(data_dir: str | Path) -> pd.DataFrame:
    parts = []
    for f in find_files(data_dir, "tracks_meta"):
        present = set(pq.read_schema(f).names)
        parts.append(pq.read_table(f, columns=[c for c in _TRACK_COLUMNS if c in present]).to_pandas())
    meta = pd.concat(parts, ignore_index=True).drop_duplicates("m4a_id").reset_index(drop=True)

    def col(name, default=None):
        return meta[name] if name in meta else pd.Series([default] * len(meta))

    year = pd.to_numeric(col("release_year"), errors="coerce")
    artist = col("m4a_artist", "").fillna("").astype(str)
    album = col("m4a_album", "").fillna("").astype(str)
    tags = [list(dict.fromkeys(_csv(g) + _csv(t) + _tag_keys(w)))
            for g, t, w in zip(col("m4a_genres_full"), col("m4a_tags_full"), col("lastfm_tag_weights"))]
    return pd.DataFrame({
        "track_idx": np.arange(len(meta), dtype=np.int32),
        "track_id": meta["m4a_id"].astype(str).to_numpy(),
        "artist": artist.to_numpy(),
        # альбом с артистом: «Greatest Hits» у разных артистов — разные альбомы
        "album": np.where(album != "", artist + "|" + album, "").astype(object),
        "release_year": year.where(year > 1900).astype(float).to_numpy(),
        "lang": col("lang", "").fillna("").astype(str).str.lower().to_numpy(),
        "popularity": pd.to_numeric(col("spotify_popularity"), errors="coerce").astype(float).to_numpy(),
        "tags": tags,
    })


# ---------------------------------------------------------------- пользователи

def age_bucket(age) -> str:
    if age is None or pd.isna(age) or age < 10 or age > 100:   # 0 и 100+ в LFM-2b — мусор
        return "unknown"
    return "50+" if age >= 50 else f"{int(age) // 10 * 10}s"


def _demographics(col: pa.ChunkedArray, n: int):
    """struct{age, gender, country} или JSON-строка (как в docs/dataset.md) -> (age, gender)."""
    arr = col.combine_chunks()
    if pa.types.is_struct(arr.type):
        ages = arr.field("age").to_pylist() if "age" in [f.name for f in arr.type] else [None] * n
        genders = arr.field("gender").to_pylist() if "gender" in [f.name for f in arr.type] else [None] * n
        valid = arr.is_valid().to_pylist()
        ages = [a if v else None for a, v in zip(ages, valid)]
        genders = [g if v else None for g, v in zip(genders, valid)]
    else:
        parsed = [json.loads(x) if isinstance(x, str) and x else {} for x in arr.to_pylist()]
        ages = [d.get("age") for d in parsed]
        genders = [d.get("gender") for d in parsed]
    ages = [float(a) if a is not None and float(a) > 0 else np.nan for a in ages]
    genders = [g if g in ("m", "f") else "unknown" for g in genders]
    return ages, genders


# ---------------------------------------------------------------- сплиты

class Split:
    def __init__(self, name: str, users: pd.DataFrame, history: pd.DataFrame, queries: pd.DataFrame):
        self.name, self.users, self.history, self.queries = name, users, history, queries

    @property
    def window_start(self) -> int:
        return WINDOW_START[self.name]


def _iter_tables(files: Sequence[Path]) -> Iterator[pa.Table]:
    cols = ["user_id", "history", "history_ts", "positives", "user_demographics"]
    for f in files:
        present = set(pq.read_schema(f).names)
        yield pq.read_table(f, columns=[c for c in cols if c in present])


def load_split(data_dir: str | Path, split: str, tracks: pd.DataFrame,
               user_ids: Optional[set] = None) -> Split:
    """Читает сплит пошардово; история сразу агрегируется до (user, track) -> count, last_ts."""
    idx_of = pd.Series(tracks["track_idx"].to_numpy(), index=tracks["track_id"].to_numpy())
    users, hist, queries = [], [], []
    for t in _iter_tables(find_files(data_dir, split)):
        uid = t.column("user_id").to_numpy().astype(np.int64)
        if user_ids is not None:
            keep = np.isin(uid, np.fromiter(user_ids, dtype=np.int64))
            t, uid = t.filter(pa.array(keep)), uid[keep]
        if t.num_rows == 0:
            continue
        if "user_demographics" in t.column_names:
            ages, genders = _demographics(t.column("user_demographics"), t.num_rows)
        else:
            ages, genders = [np.nan] * t.num_rows, ["unknown"] * t.num_rows
        users.append(pd.DataFrame({"user_id": uid, "age": ages, "gender": genders}))

        h = t.column("history").combine_chunks()
        ts = pc.list_flatten(t.column("history_ts").combine_chunks()).to_numpy()
        listens = pd.DataFrame({
            "user_id": uid[pc.list_parent_indices(h).to_numpy()],
            "track_idx": idx_of.reindex(pc.list_flatten(h).to_numpy(zero_copy_only=False)).to_numpy(),
            "ts": ts,
        }).dropna(subset=["track_idx"])
        hist.append(listens.groupby(["user_id", "track_idx"], sort=False)
                    .agg(count=("ts", "size"), last_ts=("ts", "max")).reset_index())

        p = t.column("positives").combine_chunks()
        flat = pc.list_flatten(p)
        target = np.asarray(flat.field("m4a_id").to_pylist(), dtype=object)
        p_uid = uid[pc.list_parent_indices(p).to_numpy()]
        fam = [f if f in QUERY_FAMILIES else "unknown" for f in flat.field("query_family").to_pylist()]
        queries.append(pd.DataFrame({
            "qid": [f"{split}|{u}|{m}" for u, m in zip(p_uid, target)],
            "user_id": p_uid,
            "target_idx": idx_of.reindex(target).to_numpy(),
            "query": flat.field("query").to_pylist(),
            "query_family": fam,
            "is_new": flat.field("is_new").to_pylist(),
        }))

    users_df = pd.concat(users, ignore_index=True).drop_duplicates("user_id")
    users_df["age_bucket"] = [age_bucket(a) for a in users_df["age"]]
    history_df = pd.concat(hist, ignore_index=True)
    history_df["track_idx"] = history_df["track_idx"].astype(np.int32)
    queries_df = pd.concat(queries, ignore_index=True)
    # таргеты вне каталога оценить нельзя — выбрасываем (в датасете их быть не должно)
    queries_df = queries_df.dropna(subset=["target_idx"]).reset_index(drop=True)
    queries_df["target_idx"] = queries_df["target_idx"].astype(np.int32)
    return Split(split, users_df.reset_index(drop=True), history_df, queries_df)


def subset(split: Split, user_ids) -> Split:
    ids = set(user_ids)
    return Split(split.name,
                 split.users[split.users["user_id"].isin(ids)].reset_index(drop=True),
                 split.history[split.history["user_id"].isin(ids)].reset_index(drop=True),
                 split.queries[split.queries["user_id"].isin(ids)].reset_index(drop=True))


def hash_split(user_ids, frac: float, salt: str) -> np.ndarray:
    """Детерминированное деление по user_id: True для доли frac (не зависит от порядка и seed numpy)."""
    def u(x):
        return int(hashlib.md5(f"{salt}|{x}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return np.array([u(x) < frac for x in user_ids], dtype=bool)
