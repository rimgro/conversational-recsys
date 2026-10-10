"""Единая точка загрузки по конфигу: data.source = synthetic | crs."""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from recsys.data.catalog import Catalog
from recsys.data.crs import EXTRA_COLUMNS, catalog_from_meta, read_split, read_tracks_meta, requests_from_tables, split_path
from recsys.data.synthetic import make_synthetic
from recsys.schemas import Request


@dataclass
class DataBundle:
    catalog: Catalog
    splits: Dict[str, List[Request]] = field(default_factory=dict)  # сплит -> запросы (по одному на query_id)
    eval_split: str = "test_public"

    @property
    def requests(self) -> List[Request]:
        """Запросы сплита для оценки (data.eval_split)."""
        return self.splits.get(self.eval_split, [])

    def __repr__(self) -> str:
        sizes = ", ".join(f"{k}={len(v)}" for k, v in self.splits.items())
        return f"DataBundle(tracks={len(self.catalog)}, requests: {sizes})"


def load_data(cfg: Dict[str, Any], verbose: bool = True) -> DataBundle:
    dcfg = cfg["data"]
    t0 = time.time()
    if dcfg["source"] == "synthetic":
        s = dcfg.get("synthetic", {})
        meta, tables = make_synthetic(n_tracks=s.get("n_tracks", 5000), n_users=s.get("n_users", 300),
                                      seed=s.get("seed", 42))
        catalog = catalog_from_meta(meta, max_tags=dcfg.get("max_tags", 20))
    elif dcfg["source"] == "crs":
        c = dcfg["crs"]
        catalog = _crs_catalog(c, dcfg)
        tables = {name: read_split(c["dir"], name, n_users=c.get("n_users"), seed=c.get("seed", 42))
                  for name in c.get("splits", ["test_public"])}
    else:
        raise ValueError(f"Неизвестный data.source: {dcfg['source']}")

    splits = {name: requests_from_tables(t, max_queries_per_user=dcfg.get("max_queries_per_user"),
                                         seed=dcfg.get("seed", 42), catalog=catalog)
              for name, t in tables.items()}
    bundle = DataBundle(catalog, splits, eval_split=dcfg.get("eval_split", "test_public"))
    if verbose:
        print(f"[data] {dcfg['source']}: {bundle} за {time.time() - t0:.1f} c")
    return bundle


def load_catalog(cfg: Dict[str, Any]) -> Catalog:
    """Только каталог треков (без пользователей и запросов): для scripts/make_index.py и проверок."""
    dcfg = cfg["data"]
    if dcfg["source"] == "crs":
        return _crs_catalog(dcfg["crs"], dcfg)
    return load_data(cfg, verbose=False).catalog


def _crs_catalog(c: Dict[str, Any], dcfg: Dict[str, Any]) -> Catalog:
    """tracks_meta -> Catalog, с кэшем (parquet + эмбеддинги .npy) в data.cache_dir."""
    src = split_path(c["dir"], "tracks_meta")
    st = os.stat(src)
    # ключ кэша зависит от исходного файла: новый tracks_meta не подхватит старый кэш
    # и от набора колонок: новая колонка в EXTRA_COLUMNS не должна молча взять кэш без неё
    columns = hashlib.md5(",".join(EXTRA_COLUMNS).encode()).hexdigest()[:6]
    key = f"{st.st_size}_{int(st.st_mtime)}_{dcfg.get('max_tags', 20)}_{columns}"
    cache_dir = dcfg.get("cache_dir")
    cache = os.path.join(cache_dir, f"catalog_crs_{key}.parquet") if cache_dir else None
    if cache and dcfg.get("use_cache", True) and os.path.exists(cache):
        return Catalog.load(cache)
    meta = read_tracks_meta(src)
    catalog = catalog_from_meta(meta, max_tags=dcfg.get("max_tags", 20))
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        catalog.save(cache)
    return catalog


def history_frame(request: Request) -> pd.DataFrame:
    """История запроса таблицей (для просмотра в ноутбуке)."""
    return pd.DataFrame([h.__dict__ for h in request.history])
