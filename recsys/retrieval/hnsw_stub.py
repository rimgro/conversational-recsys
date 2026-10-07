"""ЗАГЛУШКА вместо HNSW (text / collab / audio).

Возвращает детерминированную псевдослучайную выборку треков, смещённую к популярным.
Seed берётся из запроса и user_id: одинаковый запрос даёт одинаковый ответ.

Замена: hnsw.py с hnswlib по эмбеддингам (e5 по тексту трека, ALS, ivec256) с тем же
интерфейсом search(ctx) -> [Candidate], либо HTTP-клиент к Retrieval API на ВМ.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from recsys.data.catalog import Catalog
from recsys.retrieval.base import BaseRetriever
from recsys.schemas import Candidate, Context
from recsys.text import stable_hash


class HNSWStubRetriever(BaseRetriever):
    name = "hnsw"

    def __init__(self, catalog: Catalog, top_k: int = 200, exclude_listened: bool = True,
                 popularity_power: float = 0.5):
        super().__init__(catalog, top_k, exclude_listened)
        pop = catalog.df["popularity"].to_numpy(dtype=np.float64)
        p = np.power(np.maximum(pop, 0.0) + 1.0, popularity_power)
        self._p = p / p.sum()

    def search(self, ctx: Context, top_k: Optional[int] = None) -> List[Candidate]:
        k = min(top_k or self.top_k, len(self.catalog))
        seed = stable_hash(f"{ctx.summary.query}|{ctx.request.user_id}")
        rng = np.random.default_rng(seed)
        p = self._p.copy()
        excl = self.excluded_positions(ctx)
        p[excl] = 0.0
        if p.sum() <= 0 or k == 0:
            return []
        p /= p.sum()
        k = min(k, int((p > 0).sum()))
        pos = rng.choice(len(p), size=k, replace=False, p=p)
        scores = 1.0 / np.arange(1, k + 1)  # «косинус» убывает с рангом
        return self.to_candidates(pos, scores)
