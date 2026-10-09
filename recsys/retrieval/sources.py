"""Шаг 2: источники кандидатов. Каждый: search(ctx) -> [Candidate], по убыванию score.

Четыре источника, у каждого своя роль:
  relisten  свои треки пользователя: совпадение с запросом + вес в истории (93% целей датасета — повторные
            прослушивания, их ищет только он)
  audio     эмбеддинги MuQ: похожие по звуку на трек-образец («like X by Y», similar_to) или на центр вкуса
  bm25      сервис BM25 по карточке трека: слова запроса (remote.py)
  hnsw      сервис HNSW: семантический поиск по вектору реплики (remote.py)

Треки из ctx.banned_ids источники не возвращают, чтобы не тратить на них top_k.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from recsys.data.catalog import Catalog
from recsys.retrieval.local_index import (BM25Index, artist_token, build_bm25_index, country_token, index_words,
                                          lang_token, tag_token, year_token)
from recsys.schemas import Candidate, Context, DialogSummary


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


def query_terms(s: DialogSummary) -> Dict[str, float]:
    """Саммари -> взвешенные токены локального индекса (local_index.py): тег целиком ×2 и его слова, артисты ×3,
    слова query, страна / язык / год; слова исключённых тегов убираем, если они не входят в желаемые
    ('рок, но не хард-рок')."""
    q: Counter = Counter()
    for tag in s.include_tags:
        q[tag_token(tag)] += 2.0
        for w in index_words(tag):
            q[w] += 1.0
    for a in s.seed_artists:
        q[artist_token(a)] += 3.0
    for w in index_words(s.query):
        q[w] += 1.0
    for c in s.countries:
        q[country_token(c)] += 2.0
    for lang in s.languages:
        q[lang_token(lang)] += 2.0
    for y in s.years:
        q[year_token(y)] += 2.0
    wanted = {w for t in s.include_tags for w in index_words(t)}
    for w in {w for t in s.exclude_tags for w in index_words(t)} - wanted:
        q.pop(w, None)
    return dict(q)


class RelistenRetriever(BaseRetriever):
    """Свои треки пользователя: score = совпадение с запросом (0..1) + history_weight * вес в истории (0..1).
    Совпадение считает локальный BM25-индекс по тегам, жанрам, артисту, году, стране и языку трека."""
    name = "relisten"

    def __init__(self, catalog: Catalog, index: BM25Index, top_k: int = 100, history_weight: float = 0.3,
                 recency_decay: float = 0.995):
        super().__init__(catalog, top_k)
        self.index = index
        self.history_weight = history_weight
        self.recency_decay = recency_decay

    def search(self, ctx: Context, top_k: Optional[int] = None) -> List[Candidate]:
        items = [(self.catalog.pos(h.track_id), h) for h in ctx.request.history]  # свежие первыми
        items = [(p, h) for p, h in items if p is not None and h.track_id not in ctx.banned_ids]
        if not items:
            return []
        pos = np.array([p for p, _ in items])
        hist = np.array([np.log1p(h.count) * self.recency_decay ** i for i, (_, h) in enumerate(items)])
        score = self.history_weight * hist / hist.max()
        q = query_terms(ctx.summary)
        if q:
            match = self.index.scores(q)[pos]
            if match.max() > 0:
                score = score + match / match.max()
        k = min(top_k or self.top_k, len(pos))
        order = np.argsort(-score, kind="stable")[:k]
        return self.to_candidates(pos[order], score[order])


def taste_track_weights(ctx: Context, catalog: Catalog) -> Dict[str, float]:
    """Треки «центра вкуса» с весами: история (вес из профиля) + лайки и треки артистов-сидов
    (с весом самого тяжёлого трека истории): запрос для audio. Если в запросе есть образец
    («like X by Y») — только он: ищем похожее на X, а не на всю историю."""
    seed_tracks = [t for t in ctx.summary.seed_track_ids if t in catalog]
    if seed_tracks:
        return {t: 1.0 for t in seed_tracks}
    weights = {t: w for t, w in ctx.profile.track_weights.items() if t in catalog}
    top = max(weights.values(), default=1.0)
    extra = [t for t in ctx.request.liked_ids if t in catalog]
    seeds = {a.lower() for a in ctx.summary.seed_artists}
    if seeds:
        mask = catalog.df["artist"].str.lower().isin(seeds).to_numpy()
        extra += [str(t) for t in catalog.track_ids[mask]]
    for t in extra:
        weights[t] = weights.get(t, 0.0) + top
    return weights


class AudioRetriever(BaseRetriever):
    """Косинус MuQ-эмбеддингов к взвешенному центру истории (+ лайки и треки артистов-сидов) или к образцу."""
    name = "audio"

    def __init__(self, catalog: Catalog, top_k: int = 200):
        super().__init__(catalog, top_k)
        if catalog.embeddings is None:
            raise ValueError("audio: в каталоге нет эмбеддингов")

    def query_vector(self, ctx: Context) -> Optional[np.ndarray]:
        weights = {self.catalog.pos(t): w for t, w in taste_track_weights(ctx, self.catalog).items()}
        if not weights:
            return None
        pos = np.fromiter(weights.keys(), dtype=np.int64)
        v = (self.catalog.embeddings[pos] * np.fromiter(weights.values(), dtype=np.float32)[:, None]).sum(0)
        n = np.linalg.norm(v)
        return v / n if n > 0 else None

    def search(self, ctx: Context, top_k: Optional[int] = None) -> List[Candidate]:
        v = self.query_vector(ctx)
        if v is None:
            return []
        with np.errstate(all="ignore"):  # numpy 2.0 + Accelerate (macOS) даёт ложные warnings в matmul
            scores = self.catalog.embeddings @ v
        scores[self.catalog.positions(ctx.banned_ids)] = -np.inf
        k = min(top_k or self.top_k, int(np.isfinite(scores).sum()))
        top = np.argpartition(-scores, k - 1)[:k] if k else np.array([], dtype=np.int64)
        top = top[np.argsort(-scores[top], kind="stable")]
        return self.to_candidates(top, scores[top])


def build_retrievers(cfg: Dict[str, Any], catalog: Catalog,
                     bm25_index: Optional[BM25Index] = None) -> List[BaseRetriever]:
    """Источники из cfg['retrieval'] с enabled: true, в порядке конфига. bm25_index — готовый индекс для relisten,
    чтобы не строить его заново (строится за секунды, но в ноутбуке пайплайн пересобирают часто)."""
    from recsys.retrieval.remote import BM25APIRetriever, CandgenClient, HNSWAPIRetriever

    out: List[BaseRetriever] = []
    for name, c in cfg.get("retrieval", {}).items():
        if not c.get("enabled", False):
            continue
        k = c.get("top_k", 200)
        if name == "relisten":
            index = bm25_index or build_bm25_index(catalog, k1=c.get("k1", 1.2), b=c.get("b", 0.75))
            out.append(RelistenRetriever(catalog, index, top_k=k, history_weight=c.get("history_weight", 0.3)))
        elif name == "audio":
            if catalog.embeddings is None:
                raise ValueError("retrieval.audio: в каталоге нет эмбеддингов MuQ")
            out.append(AudioRetriever(catalog, top_k=k))
        elif name == "bm25":
            out.append(BM25APIRetriever(name, catalog, CandgenClient.from_config(cfg, "bm25", c),
                                        index=c.get("index", "cards"), top_k=k))
        elif name == "hnsw":
            from recsys.retrieval.query_embedder import GemmaQueryEmbedder
            embedder = GemmaQueryEmbedder(**cfg.get("candgen", {}).get("hnsw", {}).get("embedder", {}))
            out.append(HNSWAPIRetriever(name, catalog, CandgenClient.from_config(cfg, "hnsw", c), top_k=k,
                                        embedder=embedder))
        else:
            raise ValueError(f"Неизвестный источник кандидатов: {name} (есть relisten, audio, bm25, hnsw)")
    return out
