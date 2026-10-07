"""Единая точка загрузки по конфигу: data.source = synthetic | crs."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from recsys.data.catalog import Catalog
from recsys.data.crs import catalog_from_meta, read_split, read_tracks_meta, requests_from_split
from recsys.data.synthetic import make_synthetic
from recsys.schemas import Request


@dataclass
class DataBundle:
    catalog: Catalog
    splits: Dict[str, List[Request]] = field(default_factory=dict)  # сплит -> запросы (по одному на позитив)
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
        meta, split_dfs = make_synthetic(n_tracks=s.get("n_tracks", 5000), n_users=s.get("n_users", 300),
                                         seed=s.get("seed", 42))
        catalog = catalog_from_meta(meta, max_tags=dcfg.get("max_tags", 20))
    elif dcfg["source"] == "crs":
        c = dcfg["crs"]
        catalog = _crs_catalog(c, dcfg)
        split_dfs = {name: read_split(os.path.join(c["dir"], f"{name}.parquet"), n_users=c.get("n_users"),
                                      seed=c.get("seed", 42))
                     for name in c.get("splits", ["train", "test_public"])}
    else:
        raise ValueError(f"Неизвестный data.source: {dcfg['source']}")

    max_pos = dcfg.get("max_positives_per_user")
    splits = {name: requests_from_split(df, max_positives_per_user=max_pos, seed=dcfg.get("seed", 42))
              for name, df in split_dfs.items()}
    bundle = DataBundle(catalog, splits, eval_split=dcfg.get("eval_split", "test_public"))
    if verbose:
        print(f"[data] {dcfg['source']}: {bundle} за {time.time() - t0:.1f} c")
    return bundle


def _crs_catalog(c: Dict[str, Any], dcfg: Dict[str, Any]) -> Catalog:
    """tracks_meta -> Catalog, с кэшем (parquet + эмбеддинги .npy) в data.cache_dir."""
    with_lyrics = c.get("with_lyrics", False)
    cache_dir = dcfg.get("cache_dir")
    cache = os.path.join(cache_dir, f"catalog_crs{'_lyrics' if with_lyrics else ''}.parquet") if cache_dir else None
    if cache and dcfg.get("use_cache", True) and os.path.exists(cache):
        return Catalog.load(cache)
    meta = read_tracks_meta(os.path.join(c["dir"], "tracks_meta.parquet"), with_lyrics=with_lyrics)
    catalog = catalog_from_meta(meta, max_tags=dcfg.get("max_tags", 20))
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        catalog.save(cache)
    return catalog


def history_frame(request: Request) -> pd.DataFrame:
    """История запроса таблицей (для просмотра в ноутбуке)."""
    return pd.DataFrame([h.__dict__ for h in request.history])
