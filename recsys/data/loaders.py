"""Чтение файлов Music4All-Onion (Zenodo 6609677) и метаданных треков.

Форматы проверены по первым строкам файлов:
  id_tags_dict.tsv.bz2       id \\t {'pop': 100, 'rock': 55, ...}
  id_genres_tf-idf.tsv.bz2   id \\t <genre_1> \\t <genre_2> ...   (широкая tf-idf матрица)
  userid_trackid_count.tsv.bz2       user_id \\t track_id \\t count
  userid_trackid_timestamp.tsv.bz2   user_id \\t track_id \\t timestamp
"""
from __future__ import annotations

import ast
import os
from typing import Optional

import numpy as np
import pandas as pd

from recsys.data.catalog import normalize_tag_list

ONION_TAGS = "id_tags_dict.tsv.bz2"
ONION_GENRES = "id_genres_tf-idf.tsv.bz2"
ONION_COUNTS = "userid_trackid_count.tsv.bz2"
ONION_TIMESTAMPS = "userid_trackid_timestamp.tsv.bz2"


def _find(directory: str, name: str) -> Optional[str]:
    """Ищет файл как есть или без .bz2 (если уже распаковали)."""
    for cand in (name, name.replace(".bz2", "")):
        path = os.path.join(directory, cand)
        if os.path.exists(path):
            return path
    return None


def read_tags_dict(path: str, max_tags: int = 20) -> pd.DataFrame:
    """-> [track_id, tags, tag_weights], теги нормализованы и отсортированы по весу."""
    raw = pd.read_csv(path, sep="\t", dtype=str)
    id_col, dict_col = raw.columns[0], raw.columns[1]
    tags, weights = [], []
    for s in raw[dict_col].fillna("{}"):
        try:
            d = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            d = {}
        t, w = normalize_tag_list(list(d.keys()), [float(v) for v in d.values()])
        tags.append(t[:max_tags])
        weights.append(w[:max_tags])
    return pd.DataFrame({"track_id": raw[id_col].astype(str), "tags": tags, "tag_weights": weights})


def read_genres_tfidf(path: str, max_genres: int = 5, chunksize: int = 5000) -> pd.DataFrame:
    """Широкая tf-idf матрица -> [track_id, genres] (top-max_genres ненулевых по весу)."""
    ids, genres = [], []
    for chunk in pd.read_csv(path, sep="\t", chunksize=chunksize):
        names = np.array(chunk.columns[1:])
        vals = chunk.iloc[:, 1:].to_numpy(dtype=np.float32)
        top = np.argsort(-vals, axis=1)[:, :max_genres]
        for row_vals, row_top in zip(vals, top):
            genres.append([str(names[j]) for j in row_top if row_vals[j] > 0])
        ids.extend(chunk.iloc[:, 0].astype(str).tolist())
    return pd.DataFrame({"track_id": ids, "genres": genres})


def read_interactions(
    path: str,
    user_sample_mod: Optional[int] = None,
    max_rows: Optional[int] = None,
    chunksize: int = 2_000_000,
) -> pd.DataFrame:
    """-> [user_id, track_id, count] (+ timestamp, если это файл с timestamp).

    user_sample_mod=N оставляет пользователей с user_id % N == 0 (детерминированный сэмпл ~1/N).
    max_rows ограничивает число прочитанных строк (для быстрых проверок).
    """
    parts, read = [], 0
    for chunk in pd.read_csv(path, sep="\t", chunksize=chunksize, nrows=max_rows):
        read += len(chunk)
        if user_sample_mod and user_sample_mod > 1:
            uid = pd.to_numeric(chunk["user_id"], errors="coerce")
            chunk = chunk[(uid % user_sample_mod) == 0]
        parts.append(chunk)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["user_id", "track_id", "count"])
    df["user_id"] = df["user_id"].astype(str)
    df["track_id"] = df["track_id"].astype(str)
    if "count" not in df:  # файл с timestamp: одна строка = одно прослушивание
        df["count"] = 1.0
    df["count"] = df["count"].astype(float)
    return df


def popularity_from_interactions(interactions: pd.DataFrame) -> pd.Series:
    """Популярность трека = число уникальных слушателей."""
    return interactions.groupby("track_id")["user_id"].nunique().astype(float)


def read_table(path: str) -> pd.DataFrame:
    """csv / tsv / parquet / json / jsonl по расширению (bz2/gz тоже)."""
    p = path.lower().replace(".bz2", "").replace(".gz", "")
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if p.endswith(".json"):
        return pd.read_json(path)
    if p.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def build_onion_catalog_frame(
    onion_dir: str,
    popularity: Optional[pd.Series] = None,
    use_genres: bool = True,
    max_tags: int = 20,
    max_genres: int = 5,
) -> pd.DataFrame:
    """Склеивает теги, жанры и популярность Onion в одну таблицу (без названий)."""
    tags_path = _find(onion_dir, ONION_TAGS)
    if tags_path is None:
        raise FileNotFoundError(f"{ONION_TAGS} не найден в {onion_dir}")
    df = read_tags_dict(tags_path, max_tags=max_tags)
    if use_genres:
        genres_path = _find(onion_dir, ONION_GENRES)
        if genres_path:
            df = df.merge(read_genres_tfidf(genres_path, max_genres=max_genres), on="track_id", how="outer")
    df["popularity"] = df["track_id"].map(popularity).fillna(0.0) if popularity is not None else 0.0
    return df


def merge_metadata(catalog_df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Добавляет title/artist (и tags, если их нет в Onion) из метаданных нашего датасета."""
    meta = meta.copy()
    meta["track_id"] = meta["track_id"].astype(str)
    keep = [c for c in ["track_id", "title", "artist", "tags", "genres", "popularity"] if c in meta]
    meta = meta[keep].drop_duplicates("track_id")
    out = catalog_df.merge(meta, on="track_id", how="outer", suffixes=("", "_meta"))
    for col in ["title", "artist"]:
        if f"{col}_meta" in out:
            out[col] = out[f"{col}_meta"].where(out[f"{col}_meta"].notna(), out.get(col))
            out = out.drop(columns=[f"{col}_meta"])
    for col in ["tags", "genres", "popularity"]:
        meta_col = f"{col}_meta"
        if meta_col in out:
            if col == "popularity":
                base = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
                out[col] = base.where(base > 0, pd.to_numeric(out[meta_col], errors="coerce").fillna(0.0))
            else:
                out[col] = [b if isinstance(b, list) and len(b) > 0 else m
                            for b, m in zip(out[col], out[meta_col])]
            out = out.drop(columns=[meta_col])
    return out
