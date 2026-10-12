"""Источник twotower: двухбашенный кандген по готовым векторам (scripts/encode_twotower.py), без GPU и torch.

  item_vectors.npy + item_ids.npy     векторы треков (n, dim) и их m4a_id
  query_vectors_<сплит>.npz           query_id -> вектор запроса (Request.request_id = query_id датасета)

Скор трека — u · v (оба нормированы), top_k по каталогу без ctx.banned_ids. Запроса нет в файлах (новый запрос
в демо) — encoder.encode(request), если передан OnlineEncoder (recsys/twotower/build.py, нужен GPU), иначе пусто.
"""
from __future__ import annotations

import glob
import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from recsys.data.catalog import Catalog
from recsys.retrieval.sources import BaseRetriever
from recsys.schemas import Candidate, Context


class TwoTowerRetriever(BaseRetriever):
    name = "twotower"

    def __init__(self, catalog: Catalog, directory: str, top_k: int = 200, encoder: Any = None,
                 query_files: str = "query_vectors_*.npz"):
        super().__init__(catalog, top_k)
        if not os.path.exists(os.path.join(directory, "item_vectors.npy")):
            raise FileNotFoundError(f"retrieval.twotower: нет {directory}/item_vectors.npy "
                                    f"(python scripts/encode_twotower.py --run-dir {directory})")
        v = np.load(os.path.join(directory, "item_vectors.npy")).astype(np.float32)
        ids = np.load(os.path.join(directory, "item_ids.npy")).astype(str)
        self.vectors = np.zeros((len(catalog), v.shape[1]), dtype=np.float32)   # в порядке каталога
        self.known = np.zeros(len(catalog), dtype=bool)
        pos = np.array([-1 if p is None else p for p in map(catalog.pos, ids)])
        ok = pos >= 0
        self.vectors[pos[ok]] = v[ok]
        self.known[pos[ok]] = True
        self.queries: Dict[str, np.ndarray] = {}
        for path in sorted(glob.glob(os.path.join(directory, query_files))):
            with np.load(path, allow_pickle=False) as z:
                self.queries.update(zip(z["query_id"].astype(str), z["vectors"].astype(np.float32)))
        self.encoder = encoder
        self.n_missing = 0   # запросы без вектора (нет в файлах и нет encoder)

    def query_vector(self, ctx: Context) -> Optional[np.ndarray]:
        u = self.queries.get(str(ctx.request.request_id)) if ctx.request.request_id else None
        if u is None and self.encoder is not None:
            u = self.encoder.encode(ctx.request)
        return u

    def search(self, ctx: Context, top_k: Optional[int] = None) -> List[Candidate]:
        u = self.query_vector(ctx)
        if u is None:
            if self.n_missing == 0:
                warnings.warn("twotower: нет вектора запроса (запрос не из encode_twotower.py и без encoder), "
                              "источник пропущен")
            self.n_missing += 1
            return []
        with np.errstate(all="ignore"):  # numpy 2.0 + Accelerate (macOS) даёт ложные warnings в matmul
            scores = self.vectors @ u
        scores[~self.known] = -np.inf
        scores[self.catalog.positions(ctx.banned_ids)] = -np.inf
        k = min(top_k or self.top_k, int(np.isfinite(scores).sum()))
        top = np.argpartition(-scores, k - 1)[:k] if k else np.array([], dtype=np.int64)
        top = top[np.argsort(-scores[top], kind="stable")]
        return self.to_candidates(top, scores[top])


def online_encoder(cfg: Dict[str, Any], directory: str, checkpoint: str = "model.pt"):
    """OnlineEncoder для запросов, которых нет в файлах (демо): грузит Qwen и обученные веса — нужны torch,
    transformers, peft; на GPU быстро, на CPU — секунды на запрос."""
    from recsys.twotower.build import OnlineEncoder
    return OnlineEncoder.load(cfg, os.path.join(directory, checkpoint))
