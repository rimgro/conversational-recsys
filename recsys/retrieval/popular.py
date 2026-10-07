"""Популярное: в жанрах/тегах пользователя (история + user_info), иначе по всему каталогу.

Нужен для холодного старта и как «страховка», когда остальные источники пусты.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from recsys.data.catalog import Catalog
from recsys.retrieval.base import BaseRetriever
from recsys.schemas import Candidate, Context


class PopularRetriever(BaseRetriever):
    name = "popular"

    def __init__(self, catalog: Catalog, top_k: int = 100, exclude_listened: bool = True,
                 n_profile_tags: int = 5):
        super().__init__(catalog, top_k, exclude_listened)
        self.n_profile_tags = n_profile_tags
        self._pop = catalog.df["popularity"].to_numpy(dtype=np.float64)

    def search(self, ctx: Context, top_k: Optional[int] = None) -> List[Candidate]:
        k = top_k or self.top_k
        tags = (ctx.profile.top_genres(self.n_profile_tags) + ctx.profile.top_tags(self.n_profile_tags)
                + ctx.summary.user_tags)
        pool = self.catalog.positions_with_any_tag(dict.fromkeys(tags))
        if len(pool) < k:
            pool = np.arange(len(self.catalog))
        excl = self.excluded_positions(ctx)
        if len(excl):
            pool = np.setdiff1d(pool, excl, assume_unique=False)
        if len(pool) == 0:
            return []
        order = pool[np.argsort(-self._pop[pool], kind="stable")[:k]]
        return self.to_candidates(order, self._pop[order])
