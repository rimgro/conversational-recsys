"""Music4All-CRS (описание: docs/dataset.md).

tracks_meta.parquet            каталог: 64k треков, теги, жанры, аудио-атрибуты, тексты, эмбеддинги MuQ
train / test_public.parquet    строка = пользователь: история (все прослушивания до окна) + позитивы окна,
                               у каждого позитива свой запрос на русском и тип запроса (query_family)
Каждый файл может лежать частями: train-00000-of-00078.parquet, ... (так выложен полный датасет).

Один Request = один позитив: запрос -> ровно одна цель (m4a_id). 93% целей уже есть в истории
(is_new=False), поэтому прослушанное нельзя отфильтровывать.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from recsys.data.catalog import Catalog
from recsys.schemas import HistoryItem, Message, Request

# колонки tracks_meta, которые попадают в каталог как есть (если есть в файле)
EXTRA_COLUMNS = [
    "release_year", "lang", "is_instrumental", "artist_country", "pseudo_caption",
    "spotify_popularity", "energy", "valence", "tempo", "danceability",
]
_RAW_COLUMNS = ["m4a_id", "m4a_artist", "m4a_song", "m4a_album", "artist", "title",
                "m4a_genres_full", "lastfm_tag_weights", "onion_listens", "muq_embedding"]


# ---------------------------------------------------------------- каталог

def parquet_files(directory: str, name: str) -> List[str]:
    """<name>.parquet или его части <name>-00000-of-000NN.parquet."""
    single = os.path.join(directory, f"{name}.parquet")
    if os.path.exists(single):
        return [single]
    parts = sorted(glob.glob(os.path.join(directory, f"{name}-*.parquet")))
    if not parts:
        raise FileNotFoundError(f"нет {single} и частей {name}-*.parquet в {directory}")
    return parts


def _read_table(path: Union[str, List[str]], columns: Optional[List[str]] = None) -> pa.Table:
    files = [path] if isinstance(path, str) else list(path)
    return pa.concat_tables([pq.read_table(f, columns=columns) for f in files])


def read_tracks_meta(path: Union[str, List[str]], with_lyrics: bool = False,
                     with_embeddings: bool = True) -> pd.DataFrame:
    """Читает только нужные колонки (в файле их ~100, включая длинные тексты). path — файл или список частей."""
    wanted = _RAW_COLUMNS + EXTRA_COLUMNS + (["lyrics"] if with_lyrics else [])
    if not with_embeddings:
        wanted.remove("muq_embedding")
    first = path if isinstance(path, str) else path[0]
    present = set(pq.read_schema(first).names)
    return _read_table(path, columns=[c for c in wanted if c in present]).to_pandas()


def _parse_tag_weights(raw: Any, max_tags: int):
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return [], []
    d = json.loads(raw) if isinstance(raw, str) else dict(raw)
    items = sorted(d.items(), key=lambda kv: -float(kv[1]))[:max_tags]
    return [str(t) for t, _ in items], [float(w) for _, w in items]


def _split_csv(raw: Any) -> List[str]:
    if not isinstance(raw, str):
        return []
    return list(dict.fromkeys(s.strip() for s in raw.split(",") if s.strip()))


def catalog_from_meta(meta: pd.DataFrame, max_tags: int = 20) -> Catalog:
    def pick(primary: str, fallback: str) -> pd.Series:
        a = meta[primary] if primary in meta else pd.Series([None] * len(meta))
        b = meta[fallback] if fallback in meta else pd.Series([None] * len(meta))
        return a.where(a.notna() & (a.astype(str) != ""), b).fillna("").astype(str)

    parsed = [_parse_tag_weights(x, max_tags) for x in meta.get("lastfm_tag_weights", [None] * len(meta))]
    df = pd.DataFrame({
        "track_id": meta["m4a_id"].astype(str),
        "title": pick("m4a_song", "title").to_numpy(),
        "artist": pick("m4a_artist", "artist").to_numpy(),
        "album": meta["m4a_album"].fillna("").astype(str).to_numpy() if "m4a_album" in meta else "",
        "tags": [p[0] for p in parsed],
        "tag_weights": [p[1] for p in parsed],
        "genres": [_split_csv(x) for x in meta.get("m4a_genres_full", [None] * len(meta))],
        "popularity": pd.to_numeric(meta.get("onion_listens", 0), errors="coerce"),
    })
    for col in EXTRA_COLUMNS + ["lyrics"]:
        if col in meta:
            df[col] = meta[col].to_numpy()
    emb = None
    if "muq_embedding" in meta and meta["muq_embedding"].notna().all():
        emb = np.stack([np.asarray(e, dtype=np.float32) for e in meta["muq_embedding"]])
    return Catalog(df, embeddings=emb)


# ---------------------------------------------------------------- сплиты

def read_split(path: Union[str, List[str]], n_users: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    """Сэмпл пользователей до перевода в pandas: полная история в pandas занимает гигабайты.
    path — файл или список частей; части читаются по одной, из каждой берутся только выбранные строки."""
    files = [path] if isinstance(path, str) else list(path)
    sizes = [pq.read_metadata(f).num_rows for f in files]
    total = sum(sizes)
    idx = np.arange(total)
    if n_users is not None and n_users < total:
        idx = np.sort(np.random.default_rng(seed).choice(total, size=n_users, replace=False))
    tables, offset = [], 0
    for f, n in zip(files, sizes):
        local = idx[(idx >= offset) & (idx < offset + n)] - offset
        if len(local):
            t = pq.read_table(f)
            tables.append(t if len(local) == n else t.take(local))
        offset += n
    return pa.concat_tables(tables).to_pandas()


def aggregate_history(history: Sequence[str], history_ts: Optional[Sequence[int]] = None) -> List[HistoryItem]:
    """Прослушивания (с повторами) -> уникальные треки: count и время последнего, свежие первыми."""
    counts: Dict[str, int] = {}
    last: Dict[str, int] = {}
    ts_seq = history_ts if history_ts is not None else range(len(history))
    for tid, ts in zip(history, ts_seq):
        counts[tid] = counts.get(tid, 0) + 1
        last[tid] = int(ts)
    items = [HistoryItem(track_id=t, count=float(c), timestamp=last[t]) for t, c in counts.items()]
    items.sort(key=lambda h: -h.timestamp)
    return items


def parse_demographics(raw: Any) -> Dict[str, Any]:
    """{'age', 'gender', 'country'}; age 0 и country 'AQ' в выгрузке выглядят как пропуски."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return {}
    d = json.loads(raw) if isinstance(raw, str) else dict(raw)
    out: Dict[str, Any] = {}
    age = d.get("age")
    if age is not None and not (isinstance(age, float) and np.isnan(age)) and float(age) > 0:
        out["age"] = int(age)
    if d.get("gender") in ("m", "f"):
        out["gender"] = d["gender"]
    if d.get("country") and d["country"] != "AQ":
        out["country"] = d["country"]
    return out


def requests_from_split(df: pd.DataFrame, max_positives_per_user: Optional[int] = None,
                        seed: int = 42) -> List[Request]:
    """Строка пользователя -> по Request на позитив. История одного пользователя — общий список."""
    rng = np.random.default_rng(seed)
    out: List[Request] = []
    for row in df.itertuples(index=False):
        positives = list(row.positives)
        if max_positives_per_user is not None and len(positives) > max_positives_per_user:
            keep = np.sort(rng.choice(len(positives), size=max_positives_per_user, replace=False))
            positives = [positives[i] for i in keep]
        history = aggregate_history(row.history, getattr(row, "history_ts", None))
        attrs = parse_demographics(getattr(row, "user_demographics", None))
        user_id = str(row.user_id)
        split = str(getattr(row, "split", ""))
        for p in positives:
            out.append(Request(
                dialog=[Message("user", str(p["query"]))],
                history=history,
                user_info=str(getattr(row, "user_profile", "") or ""),
                user_attrs=attrs,
                user_id=user_id,
                request_id=f"{split}|{user_id}|{p['m4a_id']}",
                target_ids=[str(p["m4a_id"])],
                meta={
                    "split": split,
                    "query_family": p.get("query_family"),
                    "is_new": bool(p.get("is_new")),
                    "n_listens": int(p.get("n_listens") or 0),
                    "ts": int(p.get("ts") or 0),
                    "artist": p.get("artist"),
                    "title": p.get("title"),
                },
            ))
    return out
