"""Локальный BM25 по карточке трека: источник bm25 (топ по каталогу) и совпадение с запросом в relisten
(те же скоры, но только по трекам истории). BM25 на scipy.sparse, без сервисов.

Карточка — несколько полей, у каждого свой BM25-индекс; скор трека = сумма скоров полей с весами
(bm25_index.weights, задаются при запросе — индекс пересобирать не нужно). Поля отдельно, а не одним текстом:
иначе длинные тексты песен и описания из-за нормировки на длину «размывают» совпадения по тегам.

  meta     теги Last.fm (повтор по весу), жанры, артист, название, альбом, год / десятилетие, страна, язык, инструментал;
           токены: слова 'indie', 'rock'; 't:indie_rock' тег целиком; 'a:the_velvet_owls' артист целиком;
           'c:us' страна артиста, 'l:en' язык текста, 'y:1983' год релиза
  caption  pseudo_caption — автоописание по тегам
  lyrics   текст песни
  album    название, теги и жанры альбома
  artist   жанры артиста
  about    описание альбома и статья Википедии об артисте

Сборка на полном каталоге — около минуты, поэтому индекс кэшируется в data.cache_dir (python scripts/make_index.py).
"""
from __future__ import annotations

import hashlib
import os
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

    def arrays(self) -> Dict[str, np.ndarray]:
        """Для сохранения: матрица, словарь (термины по номеру столбца), параметры."""
        terms = np.empty(len(self.vocab), dtype=object)
        for t, j in self.vocab.items():
            terms[j] = t
        return {"data": self.W.data, "indices": self.W.indices, "indptr": self.W.indptr,
                "shape": np.array(self.W.shape), "terms": terms.astype(str), "params": np.array([self.k1, self.b])}

    @classmethod
    def from_arrays(cls, a: Dict[str, np.ndarray]) -> "BM25Index":
        idx = cls(float(a["params"][0]), float(a["params"][1]))
        idx.W = sp.csc_matrix((a["data"], a["indices"], a["indptr"]), shape=tuple(a["shape"]))
        idx.vocab = {str(t): j for j, t in enumerate(a["terms"])}
        idx.n_docs = int(a["shape"][0])
        return idx


def top_scores(s: np.ndarray, top_k: int, exclude: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Позиции и скоры top_k треков с положительным скором, по убыванию; exclude — позиции, которые не берём."""
    if exclude is not None and len(exclude):
        s = s.copy()
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



# ---------------------------------------------------------------- карточка трека: несколько полей

CARD_FIELDS: Dict[str, List[str]] = {   # поле карточки -> колонки tracks_meta (meta строится из каталога)
    "caption": ["pseudo_caption"],
    "lyrics": ["lyrics"],
    "album": ["album_name", "album_tags", "album_genres"],
    "artist": ["artist_genres"],
    "about": ["album_description", "artist_wiki_en"],
}
CARD_VERSION = 1   # поднять при изменении рецепта карточки: старый кэш не подхватится


def _text(x: Any) -> str:
    """Ячейка tracks_meta -> текст: строка как есть, список / массив тегов через запятую, пропуск -> ''."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if isinstance(x, str):
        return x
    if hasattr(x, "__len__"):
        return ", ".join(str(v) for v in x)
    return str(x)


class CardIndex:
    """Поля карточки, у каждого свой BM25Index; scores = сумма скоров полей с весами."""

    def __init__(self, fields: Dict[str, BM25Index]):
        self.fields = fields
        self.n_docs = next(iter(fields.values())).n_docs

    def scores(self, query: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """weights: поле -> вес (поля без веса не участвуют); None — все поля с весом 1."""
        s = np.zeros(self.n_docs, dtype=np.float32)
        for name, idx in self.fields.items():
            w = 1.0 if weights is None else float(weights.get(name, 0.0))
            if w:
                s += w * idx.scores(query)
        return s

    def search(self, query: Dict[str, float], top_k: int, exclude: Optional[np.ndarray] = None,
               weights: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        return top_scores(self.scores(query, weights), top_k, exclude)

    def save(self, path: str) -> None:
        arrays = {f"{name}/{k}": v for name, idx in self.fields.items() for k, v in idx.arrays().items()}
        tmp = path + ".tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, path)  # не оставить половину файла, если сборку прервали

    @classmethod
    def load(cls, path: str) -> "CardIndex":
        with np.load(path) as z:
            names = list(dict.fromkeys(k.split("/")[0] for k in z.files))
            return cls({n: BM25Index.from_arrays({k.split("/")[1]: z[k] for k in z.files if k.startswith(n + "/")})
                        for n in names})


def build_card_index(catalog: Catalog, meta: Optional[pd.DataFrame] = None, k1: float = 1.2,
                     b: float = 0.75) -> CardIndex:
    """meta — tracks_meta с колонками CARD_FIELDS и m4a_id (строки в любом порядке); без неё текстовые поля
    берутся из колонок каталога, если они там есть (синтетика), иначе поля нет."""
    fields = {"meta": build_bm25_index(catalog, k1, b)}
    src = catalog.df
    if meta is not None:
        src = meta.drop_duplicates("m4a_id").set_index("m4a_id").reindex(catalog.track_ids)
    for name, cols in CARD_FIELDS.items():
        cols = [c for c in cols if c in src]
        if not cols:
            continue
        texts = (" ".join(_text(v) for v in row) for row in zip(*(src[c].to_numpy() for c in cols)))
        fields[name] = BM25Index(k1, b).fit(Counter(index_words(t)) for t in texts)
    return CardIndex(fields)


def card_index_path(cfg: Dict[str, Any], catalog: Catalog) -> Optional[str]:
    """Файл кэша: зависит от tracks_meta (размер, время), каталога, рецепта и k1 / b; None — без кэша."""
    dcfg, icfg = cfg["data"], cfg.get("bm25_index", {})
    if dcfg.get("source") != "crs" or not dcfg.get("cache_dir") or not dcfg.get("use_cache", True):
        return None
    from recsys.data.crs import split_path
    st = os.stat(split_path(dcfg["crs"]["dir"], "tracks_meta"))
    key = (f"{CARD_VERSION}_{st.st_size}_{int(st.st_mtime)}_{len(catalog)}_{dcfg.get('max_tags', 20)}_"
           f"{icfg.get('k1', 1.2)}_{icfg.get('b', 0.75)}")
    return os.path.join(dcfg["cache_dir"], f"bm25_cards_{hashlib.md5(key.encode()).hexdigest()[:10]}.npz")


def load_card_index(cfg: Dict[str, Any], catalog: Catalog, verbose: bool = True) -> CardIndex:
    """Из кэша, если есть; иначе собирает (на Music4All-CRS — с текстовыми полями tracks_meta) и кладёт в кэш."""
    path = card_index_path(cfg, catalog)
    if path and os.path.exists(path):
        return CardIndex.load(path)
    icfg = cfg.get("bm25_index", {})
    t0 = time.time()
    meta = None
    if cfg["data"].get("source") == "crs":
        import pyarrow.parquet as pq
        from recsys.data.crs import split_path
        src = split_path(cfg["data"]["crs"]["dir"], "tracks_meta")
        present = set(pq.read_schema(src).names)
        cols = ["m4a_id"] + [c for cs in CARD_FIELDS.values() for c in cs if c in present]
        meta = pq.read_table(src, columns=cols).to_pandas()
    index = build_card_index(catalog, meta, k1=icfg.get("k1", 1.2), b=icfg.get("b", 0.75))
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        index.save(path)
    if verbose:
        sizes = ", ".join(f"{n}: {len(i.vocab)}" for n, i in index.fields.items())
        print(f"[bm25] индекс карточек собран за {time.time() - t0:.0f} c (словарь — {sizes})"
              + (f" -> {path}" if path else ""))
    return index
