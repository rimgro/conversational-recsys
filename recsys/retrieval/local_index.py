"""Локальные индексы нашей части (не путать с сервисом BM25 из ветки dev/bm25): BM25 на scipy.sparse,
64k треков строятся в памяти за секунды, без внешних зависимостей.

build_bm25_index    документ трека: теги (повтор по весу), жанры, артист, название, альбом,
                    десятилетие, страна, язык, инструментал. Виды токенов:
                      слова           'indie', 'rock'        мягкое совпадение
                      't:indie_rock'  тег целиком            'a:the_velvet_owls'  артист целиком
                      'c:us' страна артиста, 'l:en' язык текста, 'y:1983' год релиза
build_title_index   символьные триграммы «артист + название» (опечатки, транслит)
build_lyrics_index  слова текста песни (поиск по строчке)
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.data.catalog import Catalog
from recsys.text import STOPWORDS, decade_tag, tokenize


def tag_token(tag: str) -> str:
    return "t:" + tag.replace(" ", "_")


def artist_token(name: str) -> str:
    return "a:" + "_".join(tokenize(name))


def country_token(code: str) -> str:
    return "c:" + str(code).lower()


def lang_token(code: str) -> str:
    return "l:" + str(code).lower()


def year_token(year: int) -> str:
    return f"y:{int(year)}"


def index_words(text: str) -> List[str]:
    return [w for w in tokenize(text) if w not in STOPWORDS]


def char_trigrams(text: str) -> List[str]:
    """'kate bush' -> ['_ka', 'kat', 'ate', 'te_', '_bu', ...]: устойчиво к опечаткам и транслиту."""
    out: List[str] = []
    for w in tokenize(text):
        w = f"_{w}_"
        out.extend(w[i:i + 3] for i in range(len(w) - 2))
    return out


def _present(x) -> bool:
    return x is not None and not (np.isscalar(x) and pd.isna(x)) and str(x) != ""


def track_document(row: dict, max_tags: int = 20) -> Counter:
    doc: Counter = Counter()
    for tag, w in list(zip(row["tags"], row["tag_weights"]))[:max_tags]:
        rep = 1 + int(round(3 * min(float(w), 100.0) / 100.0))
        doc[tag_token(tag)] += rep
        for word in index_words(tag):
            doc[word] += rep
    for g in row["genres"]:
        doc[tag_token(g)] += 2
        for word in index_words(g):
            doc[word] += 2
    if row["artist"]:
        doc[artist_token(row["artist"])] += 3
        for word in index_words(row["artist"]):
            doc[word] += 2
    for word in index_words(row["title"]) + index_words(row.get("album") or ""):
        doc[word] += 1
    year = row.get("release_year")
    if _present(year) and int(year) > 1900:
        doc[tag_token(decade_tag(int(year)))] += 2
        doc[year_token(int(year))] += 2
    if _present(row.get("artist_country")):
        doc[country_token(row["artist_country"])] += 2
    if _present(row.get("lang")):
        doc[lang_token(row["lang"])] += 2
    if _present(row.get("is_instrumental")) and bool(row["is_instrumental"]):
        doc[tag_token("instrumental")] += 3
        doc["instrumental"] += 3
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
    optional = ["album", "release_year", "artist_country", "lang", "is_instrumental"]
    cols = ["tags", "tag_weights", "genres", "artist", "title"] + [c for c in optional if c in catalog.df]
    docs = (track_document(dict(zip(cols, vals))) for vals in zip(*(catalog.df[c] for c in cols)))
    return BM25Index(k1, b).fit(docs)


def build_title_index(catalog: Catalog) -> BM25Index:
    docs = (Counter(char_trigrams(f"{a} {t}")) for a, t in zip(catalog.df["artist"], catalog.df["title"]))
    return BM25Index(k1=1.2, b=0.3).fit(docs)


def build_lyrics_index(catalog: Catalog) -> Optional[BM25Index]:
    """None, если в каталоге нет текстов (data.crs.with_lyrics: false)."""
    if "lyrics" not in catalog.df:
        return None
    docs = (Counter(index_words(x)) if isinstance(x, str) else Counter() for x in catalog.df["lyrics"])
    return BM25Index().fit(docs)

