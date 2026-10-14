"""Шаг 4а: признаки отобранных кандидатов (строка = запрос × трек) для ранкера. Группы — features.groups:

  text       совпадение запроса с треком: BM25 и TF-IDF (скор, место, доля), покрытие слов запроса полями
             (документ, название, артист, теги, текст), фразы название / артист в запросе, биграммы текста,
             3-граммы названия и артиста
  history    свой ли трек: вес в истории, место в history, сколько раз и как давно слушал, доля артиста,
             косинус MuQ к центру истории
  item       spotify_popularity
  rrf        RRF списков bm25 + tfidf + history
  sources    место и попадание в список каждого включённого кандгена, сколько кандгенов нашли, all_rrf
  selection  скор и место в отборе (взвешенный RRF)

Кандген с candgen.sources.<имя>.features: false (twotower: учится на всём train, на ranker_train / valid он
«помнит» ответы) в признаки не попадает ни напрямую, ни через сводные (all_source_count, all_rrf_score,
selection_*): в отборе участвует, а ранкер о нём не знает.
  summary    из разбора запроса (DialogSummary): теги запроса и профиля у трека, артист-образец, нарушения
             ограничений (год, bpm, ...), страна / десятилетие — признаки старого пайплайна; по умолчанию выключена

Группы text … selection — признаки бейзлайна team1 (те же имена и формулы) без query_type: тип запроса
модели не известен (в датасете он только для метрик).
"""
from __future__ import annotations

import math
import re
from functools import cached_property
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.candgen.base import Cands
from recsys.data.parts import normalize_query
from recsys.selection import Selection, selection_rank
from recsys.schemas import Context
from recsys.text import decade_tag, tag_matches

GROUPS = ("text", "history", "item", "rrf", "sources", "selection", "summary")
CATEGORICAL: List[str] = []   # категориальные признаки (для CatBoost); query_type модели не известен
HASH_FEATURES = 2 ** 22


# ---------------------------------------------------------------- тексты треков

def words(text: str) -> List[str]:
    """Уникальные слова нормализованного текста (порядок не важен)."""
    return list(set(normalize_query(text).split()))


def bigrams(text: str) -> List[str]:
    w = normalize_query(text).split()
    return list({f"{a} {b}" for a, b in zip(w, w[1:])})


def trigrams(text: str) -> List[str]:
    t = normalize_query(text)
    return list({t[i:i + 3] for i in range(max(len(t) - 2, 0))})


class TrackTexts:
    """Поля треков для текстовых признаков: бинарные матрицы слов полей, хешированные биграммы текста песни
    и 3-граммы названия / артиста. Хранятся по столбцам (CSC): пересечение с запросом — сумма нескольких столбцов
    (слов запроса) по всем трекам, без разворота вектора запроса размером со словарь."""

    FIELDS = {"document_coverage": None, "title_coverage": ["m4a_song"], "artist_coverage": ["m4a_artist"],
              "tag_coverage": ["m4a_genres_full", "m4a_tags_full"], "lyrics_coverage": ["lyrics"]}

    def __init__(self, res):
        from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
        docs = res.track_docs
        meta = docs.meta
        self.word_vectorizer = CountVectorizer(analyzer=words, binary=True, dtype=np.float32)
        self.word_vectorizer.fit(docs.get("full"))
        self.fields: Dict[str, sp.csr_matrix] = {}
        for name, cols in self.FIELDS.items():
            texts = docs.get("full") if cols is None else meta[cols].fillna("").astype(str).agg(" ".join, axis=1)
            self.fields[name] = self.word_vectorizer.transform(texts).tocsc()
        self.lengths = {name: np.asarray(m.sum(axis=1)).ravel() for name, m in self.fields.items()}
        self.titles = np.array([normalize_query(t) for t in meta["m4a_song"].fillna("")], dtype=object)
        self.artists = np.array([normalize_query(t) for t in meta["m4a_artist"].fillna("")], dtype=object)
        self.bigram_hash = HashingVectorizer(analyzer=bigrams, n_features=HASH_FEATURES, alternate_sign=False,
                                             binary=True, norm=None, dtype=np.float32)
        self.trigram_hash = HashingVectorizer(analyzer=trigrams, n_features=HASH_FEATURES, alternate_sign=False,
                                              binary=True, norm=None, dtype=np.float32)
        self.lyrics_bigrams = self.bigram_hash.transform(meta["lyrics"].fillna("").astype(str)).tocsc()
        self.title_trigrams = self.trigram_hash.transform(self.titles).tocsc()
        self.artist_trigrams = self.trigram_hash.transform(self.artists).tocsc()
        self.title_trigram_len = np.array([len(trigrams(t)) for t in self.titles], dtype=np.float32)
        self.artist_trigram_len = np.array([len(trigrams(t)) for t in self.artists], dtype=np.float32)


def hits(matrix: sp.csc_matrix, columns: np.ndarray) -> np.ndarray:
    """Сколько из columns (слов / n-грамм запроса) есть у каждого трека: сумма этих столбцов бинарной матрицы."""
    if not len(columns):
        return np.zeros(matrix.shape[0], dtype=np.float32)
    return np.asarray(matrix[:, columns].sum(axis=1), dtype=np.float32).ravel()


# ---------------------------------------------------------------- признаки

class FeatureBuilder:
    def __init__(self, cfg: Dict[str, Any], res, sources: Sequence[str]):
        self.cfg = cfg
        self.res = res
        self.sources = list(sources)
        ccfg = cfg["candgen"]
        self.hidden = {s for s in self.sources if not ccfg["sources"].get(s, {}).get("features", True)}
        self.visible = [s for s in self.sources if s not in self.hidden]
        self.groups = list(cfg["features"]["groups"])
        unknown = set(self.groups) - set(GROUPS)
        if unknown:
            raise ValueError(f"features.groups: неизвестные {sorted(unknown)}, есть: {', '.join(GROUPS)}")
        self.missing_rank = {s: int(ccfg["sources"][s].get("top_k", ccfg.get("top_k", 1000))) + 1
                             for s in ccfg["sources"]}
        catalog = res.catalog
        self.popularity = np.nan_to_num(catalog.numeric("spotify_popularity"), nan=-1.0).astype(np.float32)

    @cached_property
    def texts(self) -> TrackTexts:
        return TrackTexts(self.res)

    def warm(self) -> None:
        """Загрузить заранее всё, что признаки берут лениво (тексты треков, индексы bm25 и tfidf): перед fork,
        чтобы процессы-копии не строили их каждый заново."""
        if "text" in self.groups:
            _ = self.texts
            for name in ("bm25", "tfidf"):
                self.res.text_index(name, self.cfg["candgen"]["sources"].get(name, {}).get("index"))

    def build(self, ctx: Context, lists: Dict[str, Cands], sel: Selection) -> pd.DataFrame:
        """Строка на каждый трек sel.ids (в их порядке)."""
        ids = sel.ids
        ranks = {name: _rank_lookup(c, ids, self.missing_rank.get(name, 1001)) for name, c in lists.items()}
        values: Dict[str, Any] = {}
        for group in self.groups:
            getattr(self, f"_{group}")(values, ctx, lists, sel, ids, ranks)
        frame = pd.DataFrame({k: v for k, v in values.items() if k not in CATEGORICAL}, index=range(len(ids)),
                             dtype=np.float32)
        for c in CATEGORICAL:
            if c in values:
                frame[c] = values[c]
        return frame

    # ------------------------------------------------------------ группы

    def _text(self, v, ctx, lists, sel, ids, ranks):
        query = _query(ctx)
        bm25_index = self.res.text_index("bm25", self.cfg["candgen"]["sources"].get("bm25", {}).get("index"))
        tfidf_index = self.res.text_index("tfidf", self.cfg["candgen"]["sources"].get("tfidf", {}).get("index"))
        bm25 = bm25_index.scores(query)
        bm25_list = lists.get("bm25", Cands.empty()).ids
        tfidf = tfidf_index.scores_batch([query])[0].astype(np.float32)
        b, t = bm25[ids], tfidf[ids]
        union = sel.union
        v["bm25_score"] = b
        v["bm25_rank"] = ranks.get("bm25", self._missing(ids, "bm25"))
        v["bm25_relative_score"] = b / max(float(bm25[bm25_list].max()) if len(bm25_list) else 0.0, 1e-8)
        v["tfidf_score"] = t
        v["tfidf_rank"] = ranks.get("tfidf", self._missing(ids, "tfidf"))
        v["tfidf_relative_score"] = t / max(float(tfidf[union].max()) if len(union) else 0.0, 1e-8)
        v["bm25_score_share"] = b / max(float(bm25[union].sum()), 1e-8)
        v["tfidf_score_share"] = t / max(float(tfidf[union].sum()), 1e-8)

        tx = self.texts
        query_words = set(words(query))
        n_words = max(len(query_words), 1)
        q_words = tx.word_vectorizer.transform([query]).indices      # слова запроса, которые есть в словаре
        for name, matrix in tx.fields.items():
            overlap = hits(matrix, q_words)[ids]
            v[name] = overlap / n_words
            length = tx.lengths[name][ids]
            if name == "title_coverage":
                v["title_token_recall"] = overlap / np.maximum(length, 1)
            elif name == "artist_coverage":
                v["artist_token_recall"] = overlap / np.maximum(length, 1)
            elif name == "tag_coverage":
                v["tag_jaccard"] = overlap / np.maximum(len(query_words) + length - overlap, 1)
        v["query_token_count"] = np.full(len(ids), len(query_words), dtype=np.float32)
        text = f" {normalize_query(query)} "
        v["title_phrase_match"] = [float(bool(x) and f" {x} " in text) for x in tx.titles[ids]]
        v["artist_phrase_match"] = [float(bool(x) and f" {x} " in text) for x in tx.artists[ids]]
        q_bigrams = tx.bigram_hash.transform([query]).indices
        v["lyrics_bigram_coverage"] = hits(tx.lyrics_bigrams, q_bigrams)[ids] / max(len(bigrams(query)), 1)
        q_tri = tx.trigram_hash.transform([query]).indices
        n_tri = len(trigrams(query))
        for name, matrix, lengths in (("title_trigram_dice", tx.title_trigrams, tx.title_trigram_len),
                                      ("artist_trigram_dice", tx.artist_trigrams, tx.artist_trigram_len)):
            common = hits(matrix, q_tri)[ids]
            v[name] = 2 * common / np.maximum(n_tri + lengths[ids], 1)

    def _history(self, v, ctx, lists, sel, ids, ranks):
        user = ctx.user
        retrieval = user.retrieval_scores
        pos_of = dict(zip(user.positions.tolist(), range(len(user.positions))))
        idx = np.array([pos_of.get(int(i), -1) for i in ids])
        mine = idx >= 0
        h = np.array([retrieval.get(int(i), 0.0) for i in ids], dtype=np.float32)
        v["history_retrieval_score"] = h
        v["history_rank"] = ranks.get("history", self._missing(ids, "history"))
        v["history_score_normalized"] = h / max(max(retrieval.values(), default=0.0), 1e-8)
        log_count, days = np.zeros(len(ids)), np.full(len(ids), -1.0)   # не свой трек: 0 прослушиваний, -1 день
        log_count[mine] = np.log1p(user.counts[idx[mine]])
        days[mine] = user.days[idx[mine]]
        v["history_track_log_count"] = log_count
        v["history_track_recency_days"] = days
        total = max(user.total, 1.0)
        artists = self.res.users.artists
        v["history_artist_share"] = [user.artist_counts.get(artists[i], 0) / total for i in ids]
        emb = self.res.catalog.embeddings
        if emb is not None and user.centroid is not None:
            with np.errstate(all="ignore"):
                v["history_audio_cosine"] = emb[ids] @ user.centroid
        else:
            v["history_audio_cosine"] = np.zeros(len(ids), dtype=np.float32)

    def _item(self, v, ctx, lists, sel, ids, ranks):
        v["spotify_popularity"] = self.popularity[ids]

    def _rrf(self, v, ctx, lists, sel, ids, ranks):
        k = float(self.cfg["selection"].get("union_rrf_k", 60))
        s = np.zeros(len(ids), dtype=np.float32)
        for name in ("bm25", "tfidf", "history"):
            if name in ranks:
                r = ranks[name]
                s += np.where(r < self.missing_rank[name], 1.0 / (k + r), 0.0).astype(np.float32)
        v["rrf_score"] = s

    def _sources(self, v, ctx, lists, sel, ids, ranks):
        count = np.zeros(len(ids), dtype=np.float32)
        for name in self.visible:
            r = ranks.get(name, self._missing(ids, name))
            hit = (r < self.missing_rank[name]).astype(np.float32)
            v[f"{name}_source_rank"] = r
            v[f"{name}_source_hit"] = hit
            count += hit
        v["all_source_count"] = count
        k = float(self.cfg["selection"].get("union_rrf_k", 60))
        v["all_rrf_score"] = sel.all_rrf if not self.hidden else self._rrf_of(ranks, {s: 1.0 for s in self.visible}, k)

    def _selection(self, v, ctx, lists, sel, ids, ranks):
        if not self.hidden:
            v["selection_rrf_score"] = sel.score
            v["selection_rank"] = selection_rank(sel)
            return
        scfg = self.cfg["selection"]   # скор отбора без скрытых кандгенов
        weights = {s: float(w) for s, w in (scfg.get("weights") or {}).items() if w and s in self.visible}
        score = self._rrf_of(ranks, weights, float(scfg.get("rrf_k", 60)))
        v["selection_rrf_score"] = score
        v["selection_rank"] = selection_rank(Selection(ids, score, score, sel.union, sel.n_union, {}))

    def _rrf_of(self, ranks, weights, k: float) -> np.ndarray:
        """Сумма weight / (k + место) по спискам weights (место — из ranks; нет в списке — 0)."""
        out = np.zeros(len(next(iter(ranks.values()))) if ranks else 0, dtype=np.float32)
        for name, w in weights.items():
            if name in ranks:
                r = ranks[name]
                out += np.where(r < self.missing_rank[name], w / (k + r), 0.0).astype(np.float32)
        return out

    def _summary(self, v, ctx, lists, sel, ids, ranks):
        s, prof, catalog = ctx.summary, ctx.profile, self.res.catalog
        df = catalog.df
        tags = [list(df.at[i, "tags"]) + list(df.at[i, "genres"]) for i in ids]
        prof_total = sum(prof.tag_weights.values()) or 1.0
        seeds = {a.lower() for a in s.seed_artists}
        artists = catalog.text("artist")[ids]
        years = catalog.numeric("release_year")[ids]
        countries = catalog.text("artist_country")[ids]
        decades = {t for t in s.include_tags if re.fullmatch(r"\d0s", t)}
        q_hits = np.array([sum(tag_matches(t, tt) for t in s.include_tags) for tt in tags], dtype=np.float32)
        v["query_tag_hits"] = q_hits
        v["query_tag_frac"] = q_hits / len(s.include_tags) if s.include_tags else np.zeros(len(ids), np.float32)
        v["user_tag_hits"] = [sum(tag_matches(t, tt) for t in s.user_tags) for tt in tags]
        v["profile_tag_score"] = [sum(prof.tag_weights.get(t, 0.0) for t in tt) / prof_total for tt in tags]
        v["artist_seed"] = [float(bool(a) and a.lower() in seeds) for a in artists]
        # страна — нет: в запросах это чаще стиль, а не факт (цель ей удовлетворяет в ~80%)
        v["n_violations"] = constraint_violations(ids, ctx, catalog, [c for c in CONSTRAINTS if c != "country"])
        v["country_match"] = [float(bool(s.countries) and c in s.countries) for c in countries]
        v["decade_match"] = [float(bool(decades) and y > 1900 and decade_tag(int(y)) in decades)
                             if not math.isnan(y) else 0.0 for y in years]

    def _missing(self, ids: np.ndarray, name: str) -> np.ndarray:
        return np.full(len(ids), self.missing_rank.get(name, 1001), dtype=np.float32)


CONSTRAINTS = ("year", "mode", "bpm", "energy", "instrumental", "vocals", "lang", "country")
BPM_TOLERANCE = 10.0


def constraint_violations(positions: np.ndarray, ctx: Context, catalog,
                          use: Sequence[str] = CONSTRAINTS) -> np.ndarray:
    """Сколько ограничений запроса нарушает каждый трек (позиции каталога). Нет значения в каталоге — не нарушение.

    year / mode / bpm / energy / instrumental / vocals — из summary.constraints (text.parse_constraints);
    lang — summary.languages (язык текста назван явно); country — summary.countries (страна артиста: в запросах
    это часто стиль, а не факт, цель ей удовлетворяет в ~80% — поэтому по умолчанию не используется).
    """
    s, c = ctx.summary, ctx.summary.constraints
    out = np.zeros(len(positions), dtype=np.int64)
    if not len(positions):
        return out
    if "year" in use and ("year_min" in c or "year_max" in c):
        y = catalog.numeric("release_year")[positions]
        known = y > 1900
        out += known & ((y < c.get("year_min", -np.inf)) | (y > c.get("year_max", np.inf)))
    if "mode" in use and "mode" in c:
        m = catalog.numeric("mode")[positions]
        out += ~np.isnan(m) & (m != c["mode"])
    if "bpm" in use and "bpm" in c:
        t, b = catalog.numeric("tempo")[positions], c["bpm"]
        dist = np.minimum.reduce([np.abs(t - b), np.abs(2 * t - b), np.abs(t / 2 - b)])  # темп вдвое — тот же бит
        out += (t > 0) & (dist > BPM_TOLERANCE)
    if "energy" in use and ("energy_min" in c or "energy_max" in c):
        e = catalog.numeric("energy")[positions]
        out += ~np.isnan(e) & ((e < c.get("energy_min", -np.inf)) | (e > c.get("energy_max", np.inf)))
    if "instrumental" in use and c.get("instrumental"):
        inst = catalog.text("is_instrumental")[positions]
        lang = catalog.text("lang")[positions]
        out += ~((inst == "True") | np.char.startswith(lang.astype(str), "INTRU"))
    if "vocals" in use and c.get("vocals"):
        other = "Male" if c["vocals"] == "female" else "Female"
        out += catalog.text("artist_gender")[positions] == other
    if "lang" in use and s.languages:
        lang = catalog.text("lang")[positions]
        out += (lang != "") & ~np.isin(lang, list(s.languages))
    if "country" in use and s.countries:
        country = catalog.text("artist_country")[positions]
        out += (country != "") & ~np.isin(country, list(s.countries))
    return out


def _rank_lookup(c: Cands, ids: np.ndarray, missing: int) -> np.ndarray:
    """Место каждого из ids в списке c (1 — первое); нет в списке — missing."""
    out = np.full(len(ids), missing, dtype=np.float32)
    if len(c) and len(ids):
        order = np.argsort(c.ids, kind="stable")
        sorted_ids = c.ids[order]
        at = np.searchsorted(sorted_ids, ids)
        at = np.minimum(at, len(sorted_ids) - 1)
        found = sorted_ids[at] == ids
        out[found] = order[at[found]] + 1
    return out


def _query(ctx: Context) -> str:
    messages = ctx.request.user_messages
    return messages[-1] if messages else ""
