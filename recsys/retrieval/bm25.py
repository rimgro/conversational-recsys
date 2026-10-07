"""BM25 по тексту трека: теги (повтор по весу) + жанры + артист + название.

Свой индекс на scipy.sparse: на 109k треков строится за секунды, без внешних зависимостей.
В документе и запросе есть два вида токенов:
  слова       'indie', 'rock'       — мягкое совпадение
  фразы       't:indie_rock'        — точное совпадение тега
              'a:the_velvet_owls'   — точное совпадение артиста
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from recsys.data.catalog import Catalog
from recsys.retrieval.base import BaseRetriever
from recsys.schemas import Candidate, Context
from recsys.text import STOPWORDS, tokenize


def tag_token(tag: str) -> str:
    return "t:" + tag.replace(" ", "_")


def artist_token(name: str) -> str:
    return "a:" + "_".join(tokenize(name))


def _words(text: str) -> List[str]:
    return [w for w in tokenize(text) if w not in STOPWORDS]


def track_document(row: dict, max_tags: int = 20) -> Counter:
    doc: Counter = Counter()
    for tag, w in list(zip(row["tags"], row["tag_weights"]))[:max_tags]:
        rep = 1 + int(round(3 * min(float(w), 100.0) / 100.0))
        doc[tag_token(tag)] += rep
        for word in _words(tag):
            doc[word] += rep
    for g in row["genres"]:
        doc[tag_token(g)] += 2
        for word in _words(g):
            doc[word] += 2
    if row["artist"]:
        doc[artist_token(row["artist"])] += 3
        for word in _words(row["artist"]):
            doc[word] += 2
    for word in _words(row["title"]):
        doc[word] += 1
    return doc


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vocab: Dict[str, int] = {}
        self.W: Optional[sp.csc_matrix] = None  # docs x terms, уже взвешенные BM25
        self.n_docs = 0

    def fit(self, docs: Iterable[Counter]) -> "BM25Index":
        rows, cols, vals = [], [], []
        for i, doc in enumerate(docs):
            for term, tf in doc.items():
                j = self.vocab.setdefault(term, len(self.vocab))
                rows.append(i)
                cols.append(j)
                vals.append(tf)
            self.n_docs = i + 1
        tf = sp.csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)),
                           shape=(self.n_docs, len(self.vocab)))
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = float(dl.mean()) if self.n_docs else 1.0
        df = np.bincount(tf.indices, minlength=len(self.vocab))
        idf = np.log1p((self.n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        coo = tf.tocoo()
        norm = self.k1 * (1 - self.b + self.b * dl[coo.row] / avgdl)
        w = coo.data * (self.k1 + 1) / (coo.data + norm) * idf[coo.col]
        self.W = sp.csc_matrix((w.astype(np.float32), (coo.row, coo.col)), shape=tf.shape)
        return self

    def scores(self, query: Dict[str, float]) -> np.ndarray:
        cols = [(self.vocab[t], w) for t, w in query.items() if t in self.vocab and w > 0]
        if not cols:
            return np.zeros(self.n_docs, dtype=np.float32)
        idx = np.array([c for c, _ in cols])
        weights = np.array([w for _, w in cols], dtype=np.float32)
        return np.asarray(self.W[:, idx] @ weights).ravel()

    def search(self, query: Dict[str, float], top_k: int,
               exclude: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        s = self.scores(query)
        if exclude is not None and len(exclude):
            s[exclude] = 0.0
        nz = np.flatnonzero(s > 0)
        if len(nz) == 0:
            return nz, s[nz]
        k = min(top_k, len(nz))
        top = nz[np.argpartition(-s[nz], k - 1)[:k]]
        top = top[np.argsort(-s[top], kind="stable")]
        return top, s[top]


def build_bm25_index(catalog: Catalog, k1: float = 1.2, b: float = 0.75) -> BM25Index:
    cols = ["tags", "tag_weights", "genres", "artist", "title"]
    docs = (track_document(dict(zip(cols, vals))) for vals in zip(*(catalog.df[c] for c in cols)))
    return BM25Index(k1, b).fit(docs)


def summary_query(ctx: Context) -> Dict[str, float]:
    """Запрос BM25 из саммари: теги (фраза ×2 + слова), артисты-сиды (×3), слова query."""
    s = ctx.summary
    q: Counter = Counter()
    for tag in s.include_tags:
        q[tag_token(tag)] += 2.0
        for w in _words(tag):
            q[w] += 1.0
    for a in s.seed_artists:
        q[artist_token(a)] += 3.0
    for w in _words(s.query):
        q[w] += 1.0
    for w in {w for t in s.exclude_tags for w in _words(t)}:
        q.pop(w, None)
    return dict(q)


class BM25Retriever(BaseRetriever):
    """Шаг 2, источник «запрос»: BM25 по саммари диалога."""
    name = "bm25"

    def __init__(self, catalog: Catalog, index: BM25Index, top_k: int = 200, exclude_listened: bool = True):
        super().__init__(catalog, top_k, exclude_listened)
        self.index = index

    def search(self, ctx: Context, top_k: Optional[int] = None) -> List[Candidate]:
        q = summary_query(ctx)
        if not q:
            return []
        pos, scores = self.index.search(q, top_k or self.top_k, exclude=self.excluded_positions(ctx))
        return self.to_candidates(pos, scores)
