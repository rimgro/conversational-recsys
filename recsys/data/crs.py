"""Music4All-CRS (описание: docs/dataset.md), файлы в папке data.crs.dir.

tracks_meta.parquet               каталог: 64k треков, теги, жанры, аудио-атрибуты, тексты, эмбеддинги MuQ
train / test_public.parquet       строка = пользователь: история (все прослушивания до окна) и профиль
{split}_queries.parquet           запросы на английском: query_id, user_id, query, source, query_type
{split}_qrels.parquet             ответы: target_m4a_id; для similar_to — exclude_m4a_id / exclude_artist

Один Request = один query_id: запрос -> ровно одна цель. 13 типов: 11 к позитивам (большинство целей уже есть
в истории, поэтому прослушанное по умолчанию не отфильтровывается) + novelty и similar_to (цель — новый трек).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from recsys.data.catalog import Catalog
from recsys.schemas import HistoryItem, Message, Request

# колонки tracks_meta, которые попадают в каталог как есть (если есть в файле)
EXTRA_COLUMNS = [
    "release_year", "lang", "is_instrumental", "artist_country", "pseudo_caption",
    "spotify_popularity", "energy", "valence", "tempo", "danceability",
    "mode", "artist_gender",   # тональность (1 — мажор) и пол артиста: ограничения запроса (fusion.constraints)
]
_RAW_COLUMNS = ["m4a_id", "m4a_artist", "m4a_song", "m4a_album", "artist", "title",
                "m4a_genres_full", "lastfm_tag_weights", "onion_listens", "muq_embedding"]


# ---------------------------------------------------------------- каталог

def split_path(directory: str, name: str) -> str:
    path = os.path.join(directory, f"{name}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"нет {path}: положите файлы Music4All-CRS в {directory} (docs/dataset.md)")
    return path


def read_tracks_meta(path: str, with_lyrics: bool = False) -> pd.DataFrame:
    """Читает только нужные колонки (в файле их ~75, включая длинные тексты)."""
    wanted = _RAW_COLUMNS + EXTRA_COLUMNS + (["lyrics"] if with_lyrics else [])
    present = set(pq.read_schema(path).names)
    return pq.read_table(path, columns=[c for c in wanted if c in present]).to_pandas()


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

_USER_COLUMNS = ["user_id", "split", "history", "history_ts", "user_profile", "user_demographics"]


def read_split(directory: str, split: str, n_users: Optional[int] = None,
               seed: int = 42) -> Dict[str, pd.DataFrame]:
    """{users, queries, qrels} сплита. Пользователи сэмплируются до перевода в pandas (полная история — гигабайты),
    запросы и ответы читаются только для выбранных пользователей."""
    users = pq.read_table(split_path(directory, split), columns=_USER_COLUMNS)
    if n_users is not None and n_users < users.num_rows:
        users = users.take(np.sort(np.random.default_rng(seed).choice(users.num_rows, size=n_users, replace=False)))
    queries = pq.read_table(split_path(directory, f"{split}_queries"))
    queries = queries.filter(pc.is_in(queries["user_id"], value_set=users["user_id"]))
    qrels = pq.read_table(split_path(directory, f"{split}_qrels"))
    qrels = qrels.filter(pc.is_in(qrels["query_id"], value_set=queries["query_id"]))
    return {"users": users.to_pandas(), "queries": queries.to_pandas(), "qrels": qrels.to_pandas()}


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


def _missing(x: Any) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def requests_from_tables(tables: Dict[str, pd.DataFrame], max_queries_per_user: Optional[int] = None,
                         seed: int = 42, catalog: Optional[Catalog] = None) -> List[Request]:
    """users + queries + qrels -> по Request на query_id. История одного пользователя — общий список.
    catalog — чтобы подписать цель (артист, название) в meta."""
    rng = np.random.default_rng(seed)
    qrels = tables["qrels"].set_index("query_id").to_dict("index")
    by_user = {u: g for u, g in tables["queries"].groupby("user_id", sort=False)}
    out: List[Request] = []
    for row in tables["users"].itertuples(index=False):
        queries = by_user.get(row.user_id)
        if queries is None:
            continue
        if max_queries_per_user is not None and len(queries) > max_queries_per_user:
            queries = queries.iloc[np.sort(rng.choice(len(queries), size=max_queries_per_user, replace=False))]
        history = aggregate_history(row.history, row.history_ts)
        attrs = parse_demographics(row.user_demographics)
        for q in queries.itertuples(index=False):
            ans = qrels[q.query_id]
            target = str(ans["target_m4a_id"])
            info = (catalog.row(target) if catalog is not None else None) or {}
            out.append(Request(
                dialog=[Message("user", str(q.query))],
                history=history,
                user_info=str(row.user_profile or ""),
                user_attrs=attrs,
                user_id=str(row.user_id),
                request_id=str(q.query_id),
                target_ids=[target],
                meta={
                    "split": str(q.split),
                    "query_type": str(q.query_type),
                    "source": str(q.source),
                    "is_new": bool(ans["is_new"]),
                    "ts": int(ans["ts"]),
                    "exclude_ids": [] if _missing(ans["exclude_m4a_id"]) else [str(ans["exclude_m4a_id"])],
                    "exclude_artist": None if _missing(ans["exclude_artist"]) else str(ans["exclude_artist"]),
                    "artist": info.get("artist"),
                    "title": info.get("title"),
                },
            ))
    return out
