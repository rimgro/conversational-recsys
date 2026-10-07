"""Шаг 2: источники кандидатов. Каждый: search(ctx) -> [Candidate], по убыванию score.

bm25     запрос из саммари диалога -> BM25
history  профиль тегов и артистов из истории -> тот же BM25 (позже: collab-HNSW по ALS / item2vec)
popular  популярное в жанрах пользователя; холодный старт
hnsw     ЗАГЛУШКА: детерминированная выборка с перекосом в популярное
         (позже: hnswlib по эмбеддингам или HTTP к Retrieval API, с тем же интерфейсом)

Треки из ctx.banned_ids (прослушанное, показанное, скипнутое) источники не возвращают,
чтобы не тратить на них top_k.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from recsys.data.catalog import Catalog
from recsys.retrieval.bm25 import BM25Index, artist_token, build_bm25_index, index_words, tag_token
from recsys.schemas import Candidate, Context
from recsys.text import stable_hash


class BaseRetriever(ABC):
    name: str = "base"

    def __init__(self, catalog: Catalog, top_k: int = 200):
        self.catalog = catalog
        self.top_k = top_k

    @abstractmethod
    def search(self, ctx: Context, top_k: Optional[int] = None) -> List[Candidate]:
        ...

    def to_candidates(self, positions: np.ndarray, scores: np.ndarray) -> List[Candidate]:
        ids = self.catalog.track_ids
        return [Candidate(track_id=str(ids[p]), source=self.name, score=float(s), rank=r)
                for r, (p, s) in enumerate(zip(positions, scores), start=1)]


class _BM25Source(BaseRetriever):
    """Общее для источников поверх BM25: строим взвешенный запрос и ищем."""

    def __init__(self, catalog: Catalog, index: BM25Index, top_k: int = 200):
        super().__init__(catalog, top_k)
        self.index = index

    def build_query(self, ctx: Context) -> Dict[str, float]:
        raise NotImplementedError

    def search(self, ctx: Context, top_k: Optional[int] = None) -> List[Candidate]:
        q = self.build_query(ctx)
        if not q:
            return []
        exclude = self.catalog.positions(ctx.banned_ids)
        pos, scores = self.index.search(q, top_k or self.top_k, exclude=exclude)
        return self.to_candidates(pos, scores)


class BM25Retriever(_BM25Source):
    name = "bm25"

    def build_query(self, ctx: Context) -> Dict[str, float]:
        """Теги (фраза ×2 + слова), артисты-сиды (×3), слова query; слова исключений убираем."""
        s = ctx.summary
        q: Counter = Counter()
        for tag in s.include_tags:
            q[tag_token(tag)] += 2.0
            for w in index_words(tag):
                q[w] += 1.0
        for a in s.seed_artists:
            q[artist_token(a)] += 3.0
        for w in index_words(s.query):
            q[w] += 1.0
        for w in {w for t in s.exclude_tags for w in index_words(t)}:
            q.pop(w, None)
        return dict(q)


class HistoryRetriever(_BM25Source):
    name = "history"

    def __init__(self, catalog: Catalog, index: BM25Index, top_k: int = 200,
                 n_tags: int = 15, n_artists: int = 5):
        super().__init__(catalog, index, top_k)
        self.n_tags = n_tags
        self.n_artists = n_artists

    def build_query(self, ctx: Context) -> Dict[str, float]:
        prof = ctx.profile
        q: Counter = Counter()
        top_tags = sorted(prof.tag_weights.items(), key=lambda kv: -kv[1])[:self.n_tags]
        for tag, w in top_tags:
            q[tag_token(tag)] += w / top_tags[0][1]
        top_art = sorted(prof.artist_weights.items(), key=lambda kv: -kv[1])[:self.n_artists]
        for a, w in top_art:
            q[artist_token(a)] += 1.5 * w / top_art[0][1]
        for tid in ctx.request.liked_ids:  # лайки в текущем диалоге
            for tag in self.catalog.tags(tid)[:5]:
                q[tag_token(tag)] += 0.5
        return dict(q)


class PopularRetriever(BaseRetriever):
    name = "popular"

    def __init__(self, catalog: Catalog, top_k: int = 100, n_profile_tags: int = 5):
        super().__init__(catalog, top_k)
        self.n_profile_tags = n_profile_tags
        self._pop = catalog.df["popularity"].to_numpy(dtype=np.float64)

    def search(self, ctx: Context, top_k: Optional[int] = None) -> List[Candidate]:
        k = top_k or self.top_k
        tags = (ctx.profile.top_genres(self.n_profile_tags) + ctx.profile.top_tags(self.n_profile_tags)
                + ctx.summary.user_tags)
        pool = self.catalog.positions_with_any_tag(dict.fromkeys(tags))
        if len(pool) < k:
            pool = np.arange(len(self.catalog))
        pool = np.setdiff1d(pool, self.catalog.positions(ctx.banned_ids))
        if len(pool) == 0:
            return []
        order = pool[np.argsort(-self._pop[pool], kind="stable")[:k]]
        return self.to_candidates(order, self._pop[order])


class HNSWStubRetriever(BaseRetriever):
    """ЗАГЛУШКА. Seed из запроса и user_id: одинаковый запрос -> одинаковый ответ."""
    name = "hnsw"

    def __init__(self, catalog: Catalog, top_k: int = 200, popularity_power: float = 0.5):
        super().__init__(catalog, top_k)
        pop = catalog.df["popularity"].to_numpy(dtype=np.float64)
        p = np.power(np.maximum(pop, 0.0) + 1.0, popularity_power)
        self._p = p / p.sum()

    def search(self, ctx: Context, top_k: Optional[int] = None) -> List[Candidate]:
        p = self._p.copy()
        p[self.catalog.positions(ctx.banned_ids)] = 0.0
        k = min(top_k or self.top_k, int((p > 0).sum()))
        if k == 0:
            return []
        rng = np.random.default_rng(stable_hash(f"{ctx.summary.query}|{ctx.request.user_id}"))
        pos = rng.choice(len(p), size=k, replace=False, p=p / p.sum())
        return self.to_candidates(pos, 1.0 / np.arange(1, k + 1))  # «косинус» убывает с рангом


def build_retrievers(cfg: Dict[str, Any], catalog: Catalog,
                     bm25_index: Optional[BM25Index] = None) -> List[BaseRetriever]:
    """Источники из cfg['retrieval'] с enabled: true, в порядке конфига."""
    rcfg = cfg.get("retrieval", {})
    enabled = {name: c for name, c in rcfg.items() if c.get("enabled", False)}
    if bm25_index is None and ({"bm25", "history"} & set(enabled)):
        b = rcfg.get("bm25", {})
        bm25_index = build_bm25_index(catalog, k1=b.get("k1", 1.2), b=b.get("b", 0.75))
    out: List[BaseRetriever] = []
    for name, c in enabled.items():
        k = c.get("top_k", 200)
        if name == "bm25":
            out.append(BM25Retriever(catalog, bm25_index, top_k=k))
        elif name == "history":
            out.append(HistoryRetriever(catalog, bm25_index, top_k=k,
                                        n_tags=c.get("n_tags", 15), n_artists=c.get("n_artists", 5)))
        elif name == "popular":
            out.append(PopularRetriever(catalog, top_k=k))
        elif name == "hnsw":
            out.append(HNSWStubRetriever(catalog, top_k=k))
        else:
            raise ValueError(f"Неизвестный источник кандидатов: {name}")
    return out
