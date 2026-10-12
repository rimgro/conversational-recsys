"""Кэш выдачи кандгенов: data.cache_dir/candidates/<часть>/<кандген>_<ключ>.parquet (query_id, ids, scores).

Ключ — всё, от чего зависит выдача: Candgen.cache_key() (параметры, индекс, файлы векторов), версия данных,
список query_id части и CACHE_VERSION. Поменяли параметр кандгена или выключили другой — пересчитается только он.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from recsys.candgen.base import Candgen, Cands

CACHE_VERSION = "1"   # поднять, если поменялся код кандгенов, а параметры — нет


def cache_path(cache_dir: Optional[str], part: str, candgen: Candgen, query_ids: List[str],
               data_key: str) -> Optional[str]:
    if not cache_dir or not part:
        return None
    deps: Dict[str, Any] = {"candgen": candgen.cache_key(), "data": data_key, "v": CACHE_VERSION,
                            "queries": hashlib.md5("\n".join(query_ids).encode()).hexdigest()}
    key = hashlib.md5(json.dumps(deps, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return os.path.join(cache_dir, "candidates", part, f"{candgen.name}_{key}.parquet")


def save(path: str, query_ids: List[str], cands: List[Cands]) -> None:
    table = pa.table({
        "query_id": pa.array(query_ids, pa.string()),
        "ids": pa.array([c.ids for c in cands], pa.list_(pa.int32())),
        "scores": pa.array([c.scores for c in cands], pa.list_(pa.float32())),
    })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)   # прерванная запись не оставит битый файл


def load(path: str, query_ids: List[str]) -> Optional[List[Cands]]:
    """Выдача в порядке query_ids; None — файла нет или он не про эти запросы."""
    if not os.path.exists(path):
        return None
    t = pq.read_table(path)
    if t.column("query_id").to_pylist() != list(query_ids):
        return None
    ids, scores = t.column("ids").combine_chunks(), t.column("scores").combine_chunks()
    flat_ids = ids.flatten().to_numpy(zero_copy_only=False).astype(np.int32)
    flat_scores = scores.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
    off = ids.offsets.to_numpy() - ids.offsets[0].as_py()
    return [Cands(flat_ids[a:b], flat_scores[a:b]) for a, b in zip(off[:-1], off[1:])]


def to_frame(query_ids: List[str], cands: List[Cands]) -> pd.DataFrame:
    return pd.DataFrame({"query_id": query_ids, "ids": [c.ids for c in cands], "scores": [c.scores for c in cands]})
