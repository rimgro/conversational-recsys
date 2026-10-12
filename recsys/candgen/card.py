"""Карточка трека: свой BM25 на scipy.sparse по полям с весами. Кандгены card_bm25 (топ по каталогу) и relisten
(те же скоры, но только по трекам истории). Ищут по разбору запроса (DialogSummary), а не по сырому тексту.

Карточка — несколько полей, у каждого свой BM25-индекс; скор трека = сумма скоров полей с весами
(candgen.card_index.weights, задаются при запросе — индекс пересобирать не нужно). Поля отдельно, а не одним текстом:
иначе длинные тексты песен и описания из-за нормировки на длину «размывают» совпадения по тегам.

  meta     теги Last.fm (повтор по весу), жанры, артист, название, альбом, год / десятилетие, страна, язык, инструментал;
           токены: слова 'indie', 'rock'; 't:indie_rock' тег целиком; 'a:the_velvet_owls' артист целиком;
           'c:us' страна артиста, 'l:en' язык текста, 'y:1983' год релиза
  caption  pseudo_caption — автоописание по тегам
  lyrics   текст песни
  album    название, теги и жанры альбома
  artist   жанры артиста
  about    описание альбома и статья Википедии об артисте

Сборка на полном каталоге — около минуты; индекс кэшируется в data.cache_dir/indexes/ (python -m recsys build-index).
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

from recsys.candgen.base import Candgen, Cands, ranked, register, top_positions
from recsys.data.catalog import Catalog
from recsys.schemas import Context, DialogSummary
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


def load_card_index(res, k1: float = 1.2, b: float = 0.75, log=print) -> CardIndex:
    """Из кэша data.cache_dir/indexes/ или собрать (текстовые поля — из tracks_meta) и сохранить.
    Ключ — версия данных, рецепт карточки (CARD_VERSION), k1 и b; веса полей задаются при запросе."""
    key = hashlib.md5(f"{CARD_VERSION}_{res.dataset.source_key}_{k1}_{b}".encode()).hexdigest()[:10]
    cache_dir = res.cfg["data"].get("cache_dir")
    path = os.path.join(cache_dir, "indexes", f"card_{key}.npz") if cache_dir else None
    if path and os.path.exists(path):
        return CardIndex.load(path)
    t0 = time.time()
    meta = res.dataset.tracks_meta(["m4a_id"] + [c for cs in CARD_FIELDS.values() for c in cs])
    index = build_card_index(res.catalog, meta, k1=k1, b=b)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        index.save(path)
    log(f"индекс карточек: собран за {time.time() - t0:.0f} c")
    return index


# ---------------------------------------------------------------- кандгены

def query_terms(s: DialogSummary) -> Dict[str, float]:
    """Разбор запроса -> взвешенные токены карточки: тег целиком ×2 и его слова, артисты ×3, слова query,
    страна / язык / год; слова исключённых тегов убираем, если они не входят в желаемые ('рок, но не хард-рок')."""
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


class _CardCandgen(Candgen):
    """Общий индекс карточек (candgen.card_index: k1, b, weights) для card_bm25 и relisten."""

    @property
    def index(self) -> CardIndex:
        return self.res.card_index

    @property
    def weights(self) -> Optional[Dict[str, float]]:
        return self.res.cfg["candgen"].get("card_index", {}).get("weights")

    def cache_key(self) -> Dict[str, Any]:
        return {**super().cache_key(), "card_index": self.res.cfg["candgen"].get("card_index", {}),
                "card_version": CARD_VERSION, "summarizer": self.res.cfg.get("summarizer", {}).get("type")}


@register("card_bm25", "text")
class CardBM25(_CardCandgen):
    """BM25 по карточке трека (поля с весами) по разбору запроса: теги, артисты, слова, страна, язык, год."""

    def retrieve(self, batch: List[Context]) -> List[Cands]:
        out = []
        for ctx in batch:
            q = query_terms(ctx.summary)
            out.append(top_positions(self.index.scores(q, self.weights), self.top_k) if q else Cands.empty())
        return out


@register("relisten", "history")
class Relisten(_CardCandgen):
    """Свои треки пользователя: совпадение карточки с запросом (0..1, к лучшему треку истории)
    + history_weight * вес в истории (log1p(count) * recency_decay^(место по свежести), 0..1)."""

    def retrieve(self, batch: List[Context]) -> List[Cands]:
        hw = float(self.cfg.get("history_weight", 0.3))
        decay = float(self.cfg.get("recency_decay", 0.995))
        out = []
        for ctx in batch:
            items = [(self.res.catalog.pos(h.track_id), h) for h in ctx.request.history]   # свежие первыми
            items = [(p, h) for p, h in items if p is not None]
            if not items:
                out.append(Cands.empty())
                continue
            pos = np.array([p for p, _ in items], dtype=np.int32)
            hist = np.array([np.log1p(h.count) * decay ** i for i, (_, h) in enumerate(items)])
            score = hw * hist / hist.max()
            q = query_terms(ctx.summary)
            if q:
                match = self.index.scores(q, self.weights)[pos]
                if match.max() > 0:
                    score = score + match / match.max()
            out.append(ranked(pos, score.astype(np.float32), self.top_k))
        return out
