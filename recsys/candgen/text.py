"""Текстовые кандгены по документам треков: BM25 (bm25s) и TF-IDF (sklearn). Как у бейзлайна team1.

Документы трека (TrackDocs):
  full      артист, название, альбом, жанры, теги, pseudo_caption, текст песни, страна, язык, год + query_document
  metadata  то же без текста песни и query_document
  queries   только query_document

query_document — тексты запросов части index (data/parts.py), у которых трек был целью. Это «память» о том,
как люди просят этот трек; строится только по части index, поэтому на ranker_train / valid / test она
не знает их собственных запросов.

Индексы (TEXT_INDEXES, параметры можно поменять в candgen.sources.<имя>.index):
  bm25           BM25 по full (k1 1.5, b 0.75)
  metadata_bm25  BM25 по metadata (b 0.3: короткие документы)
  tfidf          TF-IDF слов и биграмм по full
  query_tfidf    TF-IDF по queries
  metadata_char  TF-IDF символьных 3–5-грамм по metadata: опечатки и точные названия (exact)
Строятся один раз и кэшируются в data.cache_dir/indexes/ (ключ — параметры и версия данных).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.candgen.base import Candgen, Cands, register, top_positions
from recsys.schemas import Context

TEXT_FIELDS = ["m4a_artist", "m4a_song", "m4a_album", "m4a_genres_full", "m4a_tags_full", "pseudo_caption",
               "lyrics", "artist_country", "lang", "release_year"]
DOCS = {
    "full": TEXT_FIELDS + ["query_document"],
    "metadata": [f for f in TEXT_FIELDS if f != "lyrics"],
    "queries": ["query_document"],
}
TEXT_INDEXES: Dict[str, Dict[str, Any]] = {
    "bm25": {"type": "bm25", "docs": "full", "k1": 1.5, "b": 0.75},
    "metadata_bm25": {"type": "bm25", "docs": "metadata", "k1": 1.5, "b": 0.3},
    "tfidf": {"type": "tfidf", "docs": "full", "stop_words": "english", "ngram_range": [1, 2],
              "sublinear_tf": True, "max_features": 150000, "norm": "l2"},
    "query_tfidf": {"type": "tfidf", "docs": "queries", "stop_words": "english", "ngram_range": [1, 2],
                    "sublinear_tf": True, "max_features": 150000},
    "metadata_char": {"type": "tfidf", "docs": "metadata", "analyzer": "char_wb", "ngram_range": [3, 5],
                      "min_df": 3, "max_features": 120000, "sublinear_tf": True},
}
INDEX_VERSION = "1"


class TrackDocs:
    """Строки-документы треков в порядке каталога (сырые поля tracks_meta, как у team1)."""

    def __init__(self, meta: pd.DataFrame, query_document: pd.Series):
        self.meta = meta.copy()
        self.meta["query_document"] = query_document.to_numpy()

    @classmethod
    def from_dataset(cls, dataset) -> "TrackDocs":
        meta = dataset.tracks_meta(["m4a_id"] + TEXT_FIELDS)
        return cls(meta, query_documents(dataset, meta["m4a_id"]))

    def get(self, kind: str) -> List[str]:
        return self.meta[DOCS[kind]].fillna("").astype(str).agg(" ".join, axis=1).tolist()


def query_documents(dataset, track_ids: pd.Series) -> pd.Series:
    """Трек -> тексты запросов части index, где он цель (без повторов текста), через пробел; '' — нет таких."""
    rows = dataset.index_targets()
    docs = (rows.drop_duplicates(["target_m4a_id", "normalized"])
            .groupby("target_m4a_id")["query"].agg(" ".join))
    return track_ids.astype(str).map(docs).fillna("")


def uses_index_part(spec: Dict[str, Any]) -> bool:
    return "query_document" in DOCS[spec["docs"]]


# ---------------------------------------------------------------- индексы

class BM25sIndex:
    def __init__(self, model, n_threads: int = 0):
        self.model = model
        self.n_threads = n_threads

    @classmethod
    def build(cls, docs: List[str], k1: float, b: float) -> "BM25sIndex":
        import bm25s
        model = bm25s.BM25(k1=k1, b=b, method="lucene")
        model.index(bm25s.tokenize(docs, stopwords="en", show_progress=False), show_progress=False)
        return cls(model)

    def save(self, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "BM25sIndex":
        import bm25s
        return cls(bm25s.BM25.load(path, mmap=True))

    def search(self, texts: List[str], k: int) -> List[Cands]:
        import bm25s
        tokens = bm25s.tokenize(texts, stopwords="en", show_progress=False)
        k = min(k, self.model.scores["num_docs"])
        ids, values = self.model.retrieve(tokens, k=k, sorted=True, n_threads=self.n_threads, show_progress=False)
        return [Cands(i[v > 0].astype(np.int32), v[v > 0].astype(np.float32)) for i, v in zip(ids, values)]

    def scores(self, text: str) -> np.ndarray:
        """BM25 запроса по всем трекам."""
        import bm25s
        tokens = bm25s.tokenize([text], stopwords="en", return_ids=False, show_progress=False)[0]
        if not tokens:
            return np.zeros(self.model.scores["num_docs"], dtype=np.float32)
        return np.asarray(self.model.get_scores(tokens), dtype=np.float32)


class TfidfIndex:
    def __init__(self, vectorizer, matrix: sp.csr_matrix):
        self.vectorizer = vectorizer
        self.matrix = matrix                  # (треки, признаки), строки L2-нормированы
        self.transposed = matrix.T.tocsr()    # (признаки, треки): запросы @ transposed — скоры по каталогу

    @classmethod
    def build(cls, docs: List[str], **params) -> "TfidfIndex":
        from sklearn.feature_extraction.text import TfidfVectorizer
        params = {k: tuple(v) if k == "ngram_range" else v for k, v in params.items()}
        vectorizer = TfidfVectorizer(dtype=np.float32, **params)
        return cls(vectorizer, vectorizer.fit_transform(docs).tocsr())

    def save(self, path: str) -> None:
        import joblib
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(path, "vectorizer.joblib"))
        sp.save_npz(os.path.join(path, "tracks.npz"), self.matrix)

    @classmethod
    def load(cls, path: str) -> "TfidfIndex":
        import joblib
        return cls(joblib.load(os.path.join(path, "vectorizer.joblib")),
                   sp.load_npz(os.path.join(path, "tracks.npz")).tocsr())

    def scores_batch(self, texts: List[str]) -> np.ndarray:
        """(len(texts), треки): косинус TF-IDF запроса и трека."""
        return (self.vectorizer.transform(texts) @ self.transposed).toarray()

    def search(self, texts: List[str], k: int) -> List[Cands]:
        return [top_positions(row, k) for row in self.scores_batch(texts)]


def index_key(name: str, spec: Dict[str, Any], data_key: str, parts_cfg: Dict[str, Any]) -> str:
    deps: Dict[str, Any] = {"spec": spec, "data": data_key, "v": INDEX_VERSION}
    if uses_index_part(spec):   # query_document зависит от части index
        deps["parts"] = {k: parts_cfg.get(k) for k in ("train_split", "test_split", "buckets", "exclude_test_texts")}
    return hashlib.md5(json.dumps(deps, sort_keys=True, default=str).encode()).hexdigest()[:10]


def load_text_index(name: str, spec: Dict[str, Any], res, log=print):
    """Индекс из кэша data.cache_dir/indexes/<имя>_<ключ>/ или собрать и сохранить."""
    spec = {**TEXT_INDEXES[name], **(spec or {})}
    key = index_key(name, spec, res.dataset.source_key, res.cfg["data"]["parts"])
    cls = BM25sIndex if spec["type"] == "bm25" else TfidfIndex
    cache_dir = res.cfg["data"].get("cache_dir")
    path = os.path.join(cache_dir, "indexes", f"{name}_{key}") if cache_dir else None
    if path and os.path.exists(os.path.join(path, "done")):
        return cls.load(path)
    t0 = time.time()
    docs = res.track_docs.get(spec["docs"])
    params = {k: v for k, v in spec.items() if k not in ("type", "docs")}
    index = cls.build(docs, **params)
    if path:
        os.makedirs(path, exist_ok=True)
        index.save(path)
        with open(os.path.join(path, "spec.json"), "w") as f:
            json.dump(spec, f, indent=2)
        open(os.path.join(path, "done"), "w").close()
    log(f"индекс {name}: собран за {time.time() - t0:.0f} c")
    return cls.load(path) if path else index


# ---------------------------------------------------------------- кандгены

class _TextCandgen(Candgen):
    """Поиск по последней реплике в индексе с тем же именем, что у кандгена."""

    @property
    def index(self):
        return self.res.text_index(self.name, self.cfg.get("index"))

    def retrieve(self, batch: List[Context]) -> List[Cands]:
        return self.index.search([query_text(ctx) for ctx in batch], self.top_k)

    def cache_key(self) -> Dict[str, Any]:
        key = super().cache_key()
        spec = {**TEXT_INDEXES[self.name], **(self.cfg.get("index") or {})}
        key["index"] = index_key(self.name, spec, self.res.dataset.source_key, self.res.cfg["data"]["parts"])
        return key


def query_text(ctx: Context) -> str:
    """Текст для текстового поиска: последняя реплика пользователя."""
    messages = ctx.request.user_messages
    return messages[-1] if messages else ""


@register("bm25", "text")
class BM25(_TextCandgen):
    """BM25 по full: основной текстовый кандген."""


@register("metadata_bm25", "text")
class MetadataBM25(_TextCandgen):
    """BM25 по metadata."""


@register("tfidf", "text")
class Tfidf(_TextCandgen):
    """TF-IDF слов и биграмм по full."""


@register("query_tfidf", "text")
class QueryTfidf(_TextCandgen):
    """TF-IDF по запросам части index к треку (query_document)."""


@register("metadata_char", "text")
class MetadataChar(_TextCandgen):
    """TF-IDF символьных n-грамм по metadata: точные названия и опечатки."""
