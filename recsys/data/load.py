"""Единая точка загрузки данных по конфигу: synthetic | onion | dataset."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from recsys.data.catalog import Catalog
from recsys.data.dataset import load_dialogs, record_to_request
from recsys.data.history import InteractionStore
from recsys.data.loaders import (ONION_COUNTS, find_file, build_onion_catalog_frame, merge_metadata,
                                 popularity_from_interactions, read_interactions, read_table)
from recsys.data.synthetic import make_requests_from_interactions, make_synthetic
from recsys.schemas import Request


@dataclass
class DataBundle:
    catalog: Catalog
    interactions: Optional[InteractionStore] = None
    requests: List[Request] = field(default_factory=list)  # запросы с target_ids для оценки

    def __repr__(self) -> str:
        n_int = 0 if self.interactions is None else len(self.interactions)
        return f"DataBundle(tracks={len(self.catalog)}, interactions={n_int}, requests={len(self.requests)})"


def load_data(cfg: Dict[str, Any], verbose: bool = True) -> DataBundle:
    dcfg = cfg["data"]
    source = dcfg["source"]
    t0 = time.time()
    if source == "synthetic":
        bundle = _load_synthetic(dcfg)
    elif source == "onion":
        bundle = _load_onion(dcfg)
    elif source == "dataset":
        bundle = _load_dataset(dcfg)
    else:
        raise ValueError(f"Неизвестный data.source: {source}")
    if verbose:
        print(f"[data] {source}: {bundle} за {time.time() - t0:.1f} c")
    return bundle


# ---------------------------------------------------------------- synthetic

def _load_synthetic(dcfg: Dict[str, Any]) -> DataBundle:
    s = dcfg.get("synthetic", {})
    tracks, inter, dialogs = make_synthetic(
        n_tracks=s.get("n_tracks", 5000), n_users=s.get("n_users", 500),
        n_dialogs=s.get("n_dialogs", 300), seed=s.get("seed", 42),
    )
    out_dir = s.get("save_dir")
    if out_dir:  # сохранить в формате нашего датасета, чтобы проверить режим dataset
        os.makedirs(out_dir, exist_ok=True)
        tracks.to_parquet(os.path.join(out_dir, "tracks.parquet"), index=False)
        inter.to_parquet(os.path.join(out_dir, "interactions.parquet"), index=False)
        with open(os.path.join(out_dir, "dialogs.jsonl"), "w", encoding="utf8") as f:
            for d in dialogs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    store = InteractionStore(inter)
    requests = [record_to_request(d, interactions=store) for d in dialogs]
    return DataBundle(Catalog(tracks), store, [r for r in requests if r.target_ids])


# ---------------------------------------------------------------- onion

def _onion_interactions(ocfg: Dict[str, Any]) -> Optional[InteractionStore]:
    path = find_file(ocfg["dir"], ocfg.get("interactions_file", ONION_COUNTS))
    if path is None:
        print(f"[data] файл взаимодействий не найден в {ocfg['dir']}, история только из запросов")
        return None
    df = read_interactions(path, user_sample_mod=ocfg.get("user_sample_mod"), max_rows=ocfg.get("max_rows"))
    return InteractionStore(df)


def _onion_catalog(ocfg: Dict[str, Any], interactions: Optional[InteractionStore], cache_dir: Optional[str],
                   use_cache: bool, metadata: Optional[pd.DataFrame] = None, cache_name: str = "onion") -> Catalog:
    cache = os.path.join(cache_dir, f"catalog_{cache_name}.parquet") if cache_dir else None
    if cache and use_cache and os.path.exists(cache):
        print(f"[data] каталог из кэша {cache}")
        return Catalog.load(cache)
    pop = popularity_from_interactions(interactions.df) if interactions is not None else None
    df = build_onion_catalog_frame(ocfg["dir"], popularity=pop, use_genres=ocfg.get("use_genres", True),
                                   max_tags=ocfg.get("max_tags", 20))
    if metadata is not None:
        df = merge_metadata(df, metadata)
    catalog = Catalog(df)
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        catalog.save(cache)
    return catalog


def _load_onion(dcfg: Dict[str, Any]) -> DataBundle:
    ocfg = dcfg["onion"]
    store = _onion_interactions(ocfg)
    meta_path = ocfg.get("metadata_file")
    meta = read_table(meta_path) if meta_path and os.path.exists(meta_path) else None
    catalog = _onion_catalog(ocfg, store, dcfg.get("cache_dir"), dcfg.get("use_cache", True), meta)
    requests = []
    if store is not None:
        requests = make_requests_from_interactions(catalog, store, n_requests=ocfg.get("n_eval_requests", 200))
    return DataBundle(catalog, store, requests)


# ---------------------------------------------------------------- наш датасет

def _load_dataset(dcfg: Dict[str, Any]) -> DataBundle:
    ds = dcfg["dataset"]
    root = ds["dir"]
    tracks = read_table(os.path.join(root, ds["tracks_file"]))

    store = None
    inter_file = ds.get("interactions_file")
    if inter_file and os.path.exists(os.path.join(root, inter_file)):
        inter = read_table(os.path.join(root, inter_file))
        inter["user_id"] = inter["user_id"].astype(str)
        inter["track_id"] = inter["track_id"].astype(str)
        if "count" not in inter:
            inter["count"] = 1.0
        store = InteractionStore(inter)
    elif ds.get("use_onion_interactions") and os.path.isdir(dcfg["onion"]["dir"]):
        store = _onion_interactions(dcfg["onion"])

    # теги и жанры из Onion, названия из нашего датасета
    if ds.get("use_onion_tags") and os.path.isdir(dcfg["onion"]["dir"]):
        catalog = _onion_catalog(dcfg["onion"], store, dcfg.get("cache_dir"), dcfg.get("use_cache", True),
                                 metadata=tracks, cache_name="dataset")
    else:
        if store is not None and "popularity" not in tracks:
            tracks["popularity"] = tracks["track_id"].astype(str).map(popularity_from_interactions(store.df))
        catalog = Catalog(tracks)

    dialogs = load_dialogs(os.path.join(root, ds["dialogs_file"]))
    requests = [record_to_request(d, fields=ds.get("fields"), interactions=store,
                                  max_history=ds.get("max_history", 30)) for d in dialogs]
    return DataBundle(catalog, store, requests)
