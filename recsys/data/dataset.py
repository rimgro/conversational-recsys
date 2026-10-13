"""Данные по конфигу: каталог, сырые поля треков, части запросов (parts.py) и Request на запрос.

data.source: crs — файлы Music4All-CRS в data.dir (docs/dataset.md) | synthetic — та же структура без файлов (тесты).
Всё читается лениво и один раз: Dataset(cfg).requests("valid") грузит только пользователей этой части.
"""
from __future__ import annotations

import hashlib
import os
import time
from functools import cached_property
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import pyarrow.parquet as pq

from recsys.data.catalog import Catalog
from recsys.data.crs import (EXTRA_COLUMNS, build_requests, catalog_from_meta, read_qrels, read_queries,
                             read_tracks_meta, read_users, split_path)
from recsys.data.parts import PARTS, check_disjoint, index_excludes, make_part, part_split
from recsys.data.synthetic import make_synthetic
from recsys.schemas import Request

CATALOG_VERSION = "2"   # меняется вместе с catalog_from_meta: старый кэш каталога не подхватится


class Dataset:
    def __init__(self, cfg: Dict[str, Any], verbose: bool = True):
        self.cfg = cfg
        self.dcfg = cfg["data"]
        self.source = self.dcfg.get("source", "crs")
        if self.source not in ("crs", "synthetic"):
            raise ValueError(f"неизвестный data.source: {self.source} (crs | synthetic)")
        self.verbose = verbose
        self._parts: Dict[str, pd.DataFrame] = {}
        self._tables: Dict[str, pd.DataFrame] = {}

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[data] {msg}", flush=True)

    # ------------------------------------------------------------ треки

    @cached_property
    def _synthetic(self):
        s = self.dcfg.get("synthetic", {})
        return make_synthetic(n_tracks=s.get("n_tracks", 5000), n_users=s.get("n_users", 300), seed=s.get("seed", 42))

    @cached_property
    def catalog(self) -> Catalog:
        """Каталог (recsys/data/catalog.py), строки по возрастанию m4a_id; кэш в data.cache_dir."""
        max_tags = self.dcfg.get("max_tags", 20)
        if self.source == "synthetic":
            return catalog_from_meta(self._synthetic[0], max_tags=max_tags)
        src = split_path(self.dcfg["dir"], "tracks_meta")
        cache = self._cache_path(f"catalog_{self.source_key}_{max_tags}.parquet")
        if cache and os.path.exists(cache):
            return Catalog.load(cache)
        t0 = time.time()
        catalog = catalog_from_meta(read_tracks_meta(src), max_tags=max_tags)
        if cache:
            catalog.save(cache)
        self.log(f"каталог: {len(catalog)} треков за {time.time() - t0:.0f} c")
        return catalog

    @cached_property
    def source_key(self) -> str:
        """Ключ версии данных для кэшей: содержимое tracks_meta (md5, не время файла — кэш переносится между
        машинами, например на Google Drive) и набор колонок каталога."""
        if self.source == "synthetic":
            s = self.dcfg.get("synthetic", {})
            return f"syn{s.get('n_tracks')}_{s.get('n_users')}_{s.get('seed')}"
        md5 = hashlib.md5()
        with open(split_path(self.dcfg["dir"], "tracks_meta"), "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                md5.update(block)
        columns = hashlib.md5((",".join(EXTRA_COLUMNS) + CATALOG_VERSION).encode()).hexdigest()[:6]
        return f"{md5.hexdigest()[:12]}_{columns}"

    def tracks_meta(self, columns: Sequence[str]) -> pd.DataFrame:
        """Колонки tracks_meta в порядке каталога; нет колонки в файле — пустые строки (синтетика беднее)."""
        if self.source == "synthetic":
            meta = self._synthetic[0]
            present = [c for c in columns if c in meta]
            frame = meta[["m4a_id"] + [c for c in present if c != "m4a_id"]]
        else:
            path = split_path(self.dcfg["dir"], "tracks_meta")
            present = [c for c in columns if c in set(pq.read_schema(path).names)]
            frame = pq.read_table(path, columns=list(dict.fromkeys(["m4a_id"] + present))).to_pandas()
        frame = frame.set_index(frame["m4a_id"].astype(str)).reindex(self.catalog.track_ids)
        assert frame["m4a_id"].notna().all(), "tracks_meta и каталог разошлись"
        for c in columns:
            if c not in frame:
                frame[c] = ""
        return frame[list(columns)].reset_index(drop=True)

    # ------------------------------------------------------------ запросы

    def table(self, split: str, kind: str) -> pd.DataFrame:
        """queries | qrels сплита целиком (небольшие)."""
        key = f"{split}/{kind}"
        if key not in self._tables:
            if self.source == "synthetic":
                self._tables[key] = self._synthetic[1][split][kind]
            else:
                read = read_queries if kind == "queries" else read_qrels
                self._tables[key] = read(self.dcfg["dir"], split)
        return self._tables[key]

    def part(self, name: str) -> pd.DataFrame:
        """Запросы части (index | ranker_train | valid | test | submit), см. recsys/data/parts.py."""
        if name not in self._parts:
            pcfg = self.dcfg["parts"]
            exclude = self.table(index_excludes(pcfg), "queries") if name == "index" else None
            self._parts[name] = make_part(name, self.table(pcfg["train_split"], "queries"),
                                          self.table(pcfg["test_split"], "queries"), pcfg, exclude)
            check_disjoint(self._parts)
        return self._parts[name]

    def index_targets(self) -> pd.DataFrame:
        """Запросы части index с целями: из них обучаемые кандгены берут «тексты запросов к треку»."""
        q = self.part("index")
        qrels = self.table(self.dcfg["parts"]["train_split"], "qrels")
        return q.merge(qrels[["query_id", "target_m4a_id"]], on="query_id", validate="one_to_one")

    def users(self, split: str, user_ids) -> pd.DataFrame:
        """Строки пользователей сплита (история, профиль) только для user_ids."""
        if self.source == "synthetic":
            users = self._synthetic[1][split]["users"]
            return users[users["user_id"].isin(set(user_ids))]
        return read_users(self.dcfg["dir"], split, user_ids)

    def cutoff(self, split: str) -> Optional[int]:
        """Начало окна сплита (unix): история пользователей строго раньше (data.windows)."""
        day = self.dcfg.get("windows", {}).get(split)
        return int(pd.Timestamp(day, tz="UTC").timestamp()) if day else None

    def requests(self, part: str, labels: bool = True) -> List[Request]:
        """Request на каждый запрос части, в её порядке. labels=False — без ответов (как при сабмите)."""
        t0 = time.time()
        split = part_split(part, self.dcfg["parts"])
        queries = self.part(part)
        self.log(f"{part}: читаю историю {queries['user_id'].nunique()} пользователей ({split}), ~1–2 мин...")
        users = self.users(split, queries["user_id"].unique())
        qrels = self.table(split, "qrels") if labels else None
        if qrels is not None:
            qrels = qrels[qrels["query_id"].isin(queries["query_id"])]
        out = build_requests(queries, users, qrels, self.catalog, cutoff=self.cutoff(split))
        self.log(f"{part}: {len(out)} запросов, {queries['user_id'].nunique()} пользователей "
                 f"({split}) за {time.time() - t0:.0f} c")
        return out

    # ------------------------------------------------------------ кэш

    def _cache_path(self, name: str) -> Optional[str]:
        d = self.dcfg.get("cache_dir")
        if not d:
            return None
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, name)


def history_frame(request: Request) -> pd.DataFrame:
    """История запроса таблицей (для просмотра в ноутбуке)."""
    return pd.DataFrame([h.__dict__ for h in request.history])


__all__ = ["Dataset", "PARTS", "history_frame"]
