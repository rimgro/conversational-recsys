"""Кандидаты по истории пользователя (персонализация).

Пока content-based: профиль тегов и артистов из истории -> взвешенный запрос в тот же BM25.
Позже заменить на collab-HNSW (ALS / item2vec): вектор истории -> ближайшие треки.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Optional

from recsys.data.catalog import Catalog
from recsys.retrieval.base import BaseRetriever
from recsys.retrieval.bm25 import BM25Index, artist_token, tag_token
from recsys.schemas import Candidate, Context


class HistoryRetriever(BaseRetriever):
    name = "history"

    def __init__(self, catalog: Catalog, index: BM25Index, top_k: int = 200, exclude_listened: bool = True,
                 n_tags: int = 15, n_artists: int = 5, use_liked: bool = True):
        super().__init__(catalog, top_k, exclude_listened)
        self.index = index
        self.n_tags = n_tags
        self.n_artists = n_artists
        self.use_liked = use_liked

    def build_query(self, ctx: Context) -> dict:
        prof = ctx.profile
        q: Counter = Counter()
        top_tags = sorted(prof.tag_weights.items(), key=lambda kv: -kv[1])[:self.n_tags]
        if top_tags:
            mx = top_tags[0][1]
            for tag, w in top_tags:
                q[tag_token(tag)] += w / mx
        top_art = sorted(prof.artist_weights.items(), key=lambda kv: -kv[1])[:self.n_artists]
        if top_art:
            mx = top_art[0][1]
            for a, w in top_art:
                q[artist_token(a)] += 1.5 * w / mx
        if self.use_liked:  # лайки в текущем диалоге
            for tid in ctx.request.liked_ids:
                for tag in self.catalog.tags(tid)[:5]:
                    q[tag_token(tag)] += 0.5
        return dict(q)

    def search(self, ctx: Context, top_k: Optional[int] = None) -> List[Candidate]:
        q = self.build_query(ctx)
        if not q:
            return []
        pos, scores = self.index.search(q, top_k or self.top_k, exclude=self.excluded_positions(ctx))
        return self.to_candidates(pos, scores)
