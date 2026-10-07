"""Шаг 2: источники кандидатов. Каждый: search(ctx) -> [Candidate], по убыванию score.

bm25      запрос из саммари -> BM25 по тегам/жанрам/артисту/году/стране/языку
relisten  треки из истории самого пользователя, по совпадению с запросом и весу в истории
          (в датасете 93% целей — повторные прослушивания)
title     артист + название по символьным триграммам, с транслитом кириллицы (запросы exact)
lyrics    строчка текста песни (нужен data.crs.with_lyrics: true)
history   профиль тегов и артистов из истории -> BM25 (новые треки во вкусе пользователя)
audio     эмбеддинги MuQ: ближайшие к «центру вкуса» истории
popular   популярное в жанрах пользователя; холодный старт
hnsw      ЗАГЛУШКА текстового семантического поиска: выборка с перекосом в популярное

Треки из ctx.banned_ids источники не возвращают, чтобы не тратить на них top_k.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from recsys.data.catalog import Catalog
from recsys.retrieval.bm25 import (BM25Index, artist_token, build_bm25_index, build_lyrics_index,
                                   build_title_index, char_trigrams, country_token, index_words, lang_token,
                                   tag_token, year_token)
from recsys.ru import STOPWORDS_RU, translit
from recsys.schemas import Candidate, Context
from recsys.text import is_cyrillic, stable_hash, tokenize


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
        for c in s.countries:
            q[country_token(c)] += 2.0
        for lang in s.languages:
            q[lang_token(lang)] += 2.0
        for y in s.years:
            q[year_token(y)] += 2.0
        # слова исключённых тегов убираем, если они не входят в желаемые ('рок, но не хард-рок')
        wanted = {w for t in s.include_tags for w in index_words(t)}
        for w in {w for t in s.exclude_tags for w in index_words(t)} - wanted:
            q.pop(w, None)
        return dict(q)


class RelistenRetriever(BaseRetriever):
    """Свои треки пользователя: score = совпадение с запросом (0..1) + history_weight * вес в истории (0..1)."""
    name = "relisten"

    def __init__(self, catalog: Catalog, index: BM25Index, top_k: int = 100, history_weight: float = 0.3,
                 recency_decay: float = 0.995):
        super().__init__(catalog, top_k)
        self.query_source = BM25Retriever(catalog, index)
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
        q = self.query_source.build_query(ctx)
        if q:
            match = self.index.scores(q)[pos]
            if match.max() > 0:
                score = score + match / match.max()
        k = min(top_k or self.top_k, len(pos))
        order = np.argsort(-score, kind="stable")[:k]
        return self.to_candidates(pos[order], score[order])


class TitleRetriever(BaseRetriever):
    """Артист + название по триграммам. Отдаёт только треки, покрывающие >= min_coverage триграмм запроса."""
    name = "title"

    def __init__(self, catalog: Catalog, top_k: int = 50, min_coverage: float = 0.4):
        super().__init__(catalog, top_k)
        self.index = build_title_index(catalog)
        self.min_coverage = min_coverage

    @staticmethod
    def query_text(ctx: Context) -> str:
        """Латиница как есть, кириллица транслитом; служебные русские слова выкидываем."""
        last = ctx.request.user_messages[-1] if ctx.request.user_messages else ""
        words = [translit(w) if is_cyrillic(w) else w for w in tokenize(last) if w not in STOPWORDS_RU]
        return " ".join(words + ctx.summary.seed_artists)

    def search(self, ctx: Context, top_k: Optional[int] = None) -> List[Candidate]:
        grams = char_trigrams(self.query_text(ctx))
        if len(grams) < 3:
            return []
        q = dict(Counter(grams))
        pos, scores = self.index.search(q, top_k or self.top_k, exclude=self.catalog.positions(ctx.banned_ids))
        qset = set(grams)
        keep = [i for i, p in enumerate(pos)
                if len(qset & set(char_trigrams(f"{self.catalog.df.at[p, 'artist']} {self.catalog.df.at[p, 'title']}")))
                >= self.min_coverage * len(qset)]
        return self.to_candidates(pos[keep], scores[keep])


class LyricsRetriever(_BM25Source):
    """Строчка текста: латинские слова последней реплики -> BM25 по текстам. Нужно >= 2 слов."""
    name = "lyrics"

    def build_query(self, ctx: Context) -> Dict[str, float]:
        last = ctx.request.user_messages[-1] if ctx.request.user_messages else ""
        words = [w for w in index_words(last) if not is_cyrillic(w)]
        return dict(Counter(words)) if len(words) >= 2 else {}


def taste_track_weights(ctx: Context, catalog: Catalog) -> Dict[str, float]:
    """Треки «центра вкуса» с весами: история (вес из профиля) + лайки и треки артистов-сидов
    (с весом самого тяжёлого трека истории). Общий запрос для audio и hnsw_api."""
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
    """Косинус MuQ-эмбеддингов к взвешенному центру истории (+ лайки и треки артистов-сидов)."""
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
    """Источники из cfg['retrieval'] с enabled: true, в порядке конфига.

    Тип источника — поле type (по умолчанию = имя), так можно завести несколько источников
    одного типа: bm25_genres, bm25_tags (type: bm25_api) и т.д. Удалённые типы: recsys/retrieval/remote.py.
    """
    from recsys.retrieval.remote import BM25APIRetriever, CandgenClient, HNSWAPIRetriever

    rcfg = cfg.get("retrieval", {})
    enabled = {name: c for name, c in rcfg.items() if c.get("enabled", False)}
    kinds = {name: c.get("type", name) for name, c in enabled.items()}
    if bm25_index is None and ({"bm25", "history", "relisten"} & set(kinds.values())):
        b = rcfg.get("bm25", {})
        bm25_index = build_bm25_index(catalog, k1=b.get("k1", 1.2), b=b.get("b", 0.75))
    out: List[BaseRetriever] = []
    for name, c in enabled.items():
        k = c.get("top_k", 200)
        kind = kinds[name]
        if kind == "bm25_api":
            out.append(BM25APIRetriever(name, catalog, CandgenClient.from_config(cfg, c), index=c.get("index", "genres"),
                                        top_k=k, query=c.get("query", "tags")))
        elif kind == "hnsw_api":
            out.append(HNSWAPIRetriever(name, catalog, CandgenClient.from_config(cfg, c), index=c.get("index", "audio"),
                                        top_k=k))
        elif kind == "bm25":
            out.append(BM25Retriever(catalog, bm25_index, top_k=k))
        elif kind == "history":
            out.append(HistoryRetriever(catalog, bm25_index, top_k=k,
                                        n_tags=c.get("n_tags", 15), n_artists=c.get("n_artists", 5)))
        elif kind == "popular":
            out.append(PopularRetriever(catalog, top_k=k))
        elif kind == "hnsw":
            out.append(HNSWStubRetriever(catalog, top_k=k))
        elif kind == "relisten":
            out.append(RelistenRetriever(catalog, bm25_index, top_k=k,
                                         history_weight=c.get("history_weight", 0.3)))
        elif kind == "title":
            out.append(TitleRetriever(catalog, top_k=k, min_coverage=c.get("min_coverage", 0.4)))
        elif kind == "lyrics":
            lyrics_index = build_lyrics_index(catalog)
            if lyrics_index is None:
                warnings.warn("retrieval.lyrics: в каталоге нет текстов (data.crs.with_lyrics: false), источник выключен")
                continue
            out.append(LyricsRetriever(catalog, lyrics_index, top_k=k))
        elif kind == "audio":
            if catalog.embeddings is None:
                warnings.warn("retrieval.audio: в каталоге нет эмбеддингов, источник выключен")
                continue
            out.append(AudioRetriever(catalog, top_k=k))
        else:
            raise ValueError(f"Неизвестный тип источника кандидатов: {name} (type={kind})")
    return out
