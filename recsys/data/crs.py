"""Music4All-CRS (описание: docs/dataset.md), файлы в папке data.dir.

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
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
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


def read_tracks_meta(path: str) -> pd.DataFrame:
    """Читает только нужные колонки (в файле их ~75, включая длинные тексты)."""
    wanted = _RAW_COLUMNS + EXTRA_COLUMNS
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
    """Строки — по возрастанию m4a_id: номер трека один и тот же в каталоге, индексах кандгенов, semantic ID
    и у бейзлайна team1 (tracks_meta уже так отсортирован, синтетика — нет)."""
    meta = meta.sort_values("m4a_id", kind="stable").reset_index(drop=True)

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
    for col in EXTRA_COLUMNS:
        if col in meta:
            df[col] = meta[col].to_numpy()
    emb = None
    if "muq_embedding" in meta and meta["muq_embedding"].notna().all():
        emb = np.stack([np.asarray(e, dtype=np.float32) for e in meta["muq_embedding"]])
        emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)   # косинус = скалярное произведение
    return Catalog(df, embeddings=emb)


# ---------------------------------------------------------------- сплиты

_USER_COLUMNS = ["user_id", "split", "history", "history_ts", "user_profile", "user_demographics"]


def read_queries(directory: str, split: str) -> pd.DataFrame:
    return pq.read_table(split_path(directory, f"{split}_queries")).to_pandas()


def read_qrels(directory: str, split: str) -> pd.DataFrame:
    return pq.read_table(split_path(directory, f"{split}_qrels")).to_pandas()


def read_users(directory: str, split: str, user_ids: Optional[Iterable[Any]] = None) -> pd.DataFrame:
    """Строки пользователей сплита; user_ids — только эти (фильтр до перевода в pandas: история — гигабайты)."""
    users = pq.read_table(split_path(directory, split), columns=_USER_COLUMNS)
    if user_ids is not None:
        users = users.filter(pc.is_in(users["user_id"], value_set=pa.array(sorted(set(user_ids)),
                                                                          type=users.schema.field("user_id").type)))
    return users.to_pandas()


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


def build_requests(queries: pd.DataFrame, users: pd.DataFrame, qrels: Optional[pd.DataFrame] = None,
                   catalog: Optional[Catalog] = None, cutoff: Optional[int] = None) -> List[Request]:
    """По Request на строку queries, в их порядке. История одного пользователя — общий список (не копируется).

    users — строки сплита (user_id, history, history_ts, user_profile, user_demographics) для всех user_id из queries;
    qrels — ответы (без них target_ids пустые: сабмит); cutoff — начало окна сплита (unix): история строго до него;
    catalog — подписать цель (артист, название) в meta.
    """
    by_user = {row.user_id: row for row in users.itertuples(index=False)}
    answers = qrels.set_index("query_id").to_dict("index") if qrels is not None and len(qrels) else {}
    histories: Dict[Any, List[HistoryItem]] = {}
    out: List[Request] = []
    for q in queries.itertuples(index=False):
        row = by_user.get(q.user_id)
        if row is None:
            raise KeyError(f"нет пользователя {q.user_id} для запроса {q.query_id}")
        if q.user_id not in histories:
            histories[q.user_id] = aggregate_history(row.history, row.history_ts)
        meta: Dict[str, Any] = {"split": str(q.split), "query_type": str(q.query_type),
                                "source": str(getattr(q, "source", "")), "cutoff": cutoff}
        targets: List[str] = []
        ans = answers.get(q.query_id)
        if ans is not None:
            target = str(ans["target_m4a_id"])
            targets = [target]
            info = (catalog.row(target) if catalog is not None else None) or {}
            meta.update(
                is_new=bool(ans["is_new"]), ts=int(ans["ts"]),
                exclude_ids=[] if _missing(ans["exclude_m4a_id"]) else [str(ans["exclude_m4a_id"])],
                exclude_artist=None if _missing(ans["exclude_artist"]) else str(ans["exclude_artist"]),
                artist=info.get("artist"), title=info.get("title"))
        out.append(Request(
            dialog=[Message("user", str(q.query))],
            history=histories[q.user_id],
            user_info=str(row.user_profile or ""),
            user_attrs=parse_demographics(row.user_demographics),
            user_id=str(q.user_id),
            request_id=str(q.query_id),
            target_ids=targets,
            meta=meta,
        ))
    return out
