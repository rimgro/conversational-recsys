"""Признаки (запрос, кандидат). Одна функция для обучения и инференса.

Пары (user, track) / (user, artist) / ... ищутся по int64-ключам через searchsorted,
пересечения тегов — построчным произведением разреженных матриц: без merge на 30M+ строк.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ranker.data import AGE_BUCKETS, GENDERS, WINDOW_YEAR, Split
from ranker.priors import Priors, lookup

CATEGORICAL: Dict[str, List[str]] = {"age_bucket": AGE_BUCKETS, "gender": GENDERS}
FEATURES = [
    # источники
    "rrf_score", "n_sources",
    # пользователь, трек
    "age_bucket", "gender", "popularity",
    # CTR артиста
    "ctr_artist",
    # возраст x год релиза
    "age_at_release", "year_diff_history",
    # пользователь x трек / артист / альбом
    "in_history", "log_history_count", "artist_share", "log_album_listens",
    # пользователь x язык
    "lang_share",
    # теги запроса и профиля
    "query_tag_frac", "profile_tag_score",
]


class Codes:
    """Целочисленные коды артиста/альбома/языка и матрица тегов треков (бинарная)."""

    def __init__(self, tracks: pd.DataFrame):
        self.n_tracks = len(tracks)
        self.artist, self.artists = pd.factorize(tracks["artist"])
        album = tracks["album"].replace("", np.nan)
        self.album, _ = pd.factorize(album)                      # -1 = нет альбома
        self.lang, _ = pd.factorize(tracks["lang"].replace("", np.nan))
        self.n_artist, self.n_album, self.n_lang = len(self.artists), self.album.max() + 2, self.lang.max() + 2
        self.vocab: Dict[str, int] = {}
        rows, cols = [], []
        for i, tags in enumerate(tracks["tags"]):
            for t in tags:
                rows.append(i), cols.append(self.vocab.setdefault(t, len(self.vocab)))
        self.tags = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                               shape=(self.n_tracks, max(len(self.vocab), 1)))
        self.tags.data[:] = 1.0
        self.year = tracks["release_year"].to_numpy(float)
        self.popularity = tracks["popularity"].to_numpy(float)
        self.artist_name = tracks["artist"].to_numpy()

    def words_matrix(self, words_lists) -> csr_matrix:
        rows, cols = [], []
        for i, words in enumerate(words_lists):
            for w in dict.fromkeys(words):
                j = self.vocab.get(w)
                if j is not None:
                    rows.append(i), cols.append(j)
        return csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(words_lists), self.tags.shape[1]))


def _keyed(keys: np.ndarray, *values: np.ndarray):
    order = np.argsort(keys, kind="stable")
    return (keys[order],) + tuple(v[order] for v in values)


def _lookup(sorted_keys: np.ndarray, values: np.ndarray, q: np.ndarray, default=np.nan) -> np.ndarray:
    pos = np.searchsorted(sorted_keys, q)
    pos_c = np.minimum(pos, len(sorted_keys) - 1) if len(sorted_keys) else pos
    ok = (pos < len(sorted_keys)) & (sorted_keys[pos_c] == q) if len(sorted_keys) else np.zeros(len(q), bool)
    out = np.full(len(q), default, dtype=float)
    out[ok] = values[pos_c[ok]]
    return out


def _rowdot(a: csr_matrix, ai: np.ndarray, b: csr_matrix, bi: np.ndarray, chunk: int = 1_000_000) -> np.ndarray:
    """out[r] = <a[ai[r]], b[bi[r]]>. Строки b (теги трека) короткие: разворачиваем их ненулевые
    элементы и ищем пары (строка a, колонка) по отсортированным ключам a — без построчных matmul."""
    a = a.tocsr()
    a.sort_indices()
    n_cols = np.int64(a.shape[1])
    a_keys = np.repeat(np.arange(a.shape[0], dtype=np.int64), np.diff(a.indptr)) * n_cols + a.indices
    out = np.empty(len(ai), dtype=np.float32)
    for s in range(0, len(ai), chunk):
        bi_c, ai_c = bi[s:s + chunk], ai[s:s + chunk].astype(np.int64)
        starts, lens = b.indptr[bi_c], np.diff(b.indptr)[bi_c]
        rep = np.repeat(np.arange(len(bi_c)), lens)
        flat = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens) + np.repeat(starts, lens)
        av = _lookup(a_keys, a.data, ai_c[rep] * n_cols + b.indices[flat], default=0.0)
        out[s:s + chunk] = np.bincount(rep, weights=av * b.data[flat], minlength=len(bi_c))
    return out


class UserStats:
    """Агрегаты истории пользователей одного сплита (всё — до начала окна)."""

    def __init__(self, split: Split, codes: Codes):
        h = split.history
        self.user_ids = pd.Index(split.users["user_id"].to_numpy())
        u = self.user_ids.get_indexer(h["user_id"].to_numpy()).astype(np.int64)
        t = h["track_idx"].to_numpy(np.int64)
        cnt = h["count"].to_numpy(float)
        n_users = len(self.user_ids)

        self.ut = _keyed(u * codes.n_tracks + t, cnt)

        art = codes.artist[t].astype(np.int64)
        ua = pd.DataFrame({"k": u * codes.n_artist + art, "c": cnt}).groupby("k")["c"].sum()
        self.ua = (ua.index.to_numpy(), ua.to_numpy(float))

        alb = codes.album[t].astype(np.int64)
        m = alb >= 0
        ual = pd.DataFrame({"k": u[m] * codes.n_album + alb[m], "c": cnt[m]}).groupby("k")["c"].sum()
        self.ual = (ual.index.to_numpy(), ual.to_numpy(float))

        self.total = np.bincount(u, weights=cnt, minlength=n_users)
        lang = codes.lang[t].astype(np.int64)
        m = lang >= 0
        ul = pd.DataFrame({"k": u[m] * codes.n_lang + lang[m], "c": cnt[m]}).groupby("k")["c"].sum()
        lang_total = np.bincount(u[m], weights=cnt[m], minlength=n_users)
        self.ul = (ul.index.to_numpy(), ul.to_numpy(float) / lang_total[ul.index.to_numpy() // codes.n_lang])

        def wmean(x):
            ok = ~np.isnan(x[t])
            s = np.bincount(u[ok], weights=cnt[ok] * x[t][ok], minlength=n_users)
            n = np.bincount(u[ok], weights=cnt[ok], minlength=n_users)
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(n > 0, s / n, np.nan)
        self.mean_year = wmean(codes.year)

        # профиль тегов: прослушивания x теги трека, строки нормированы на 1
        listens = csr_matrix((cnt, (u, t)), shape=(n_users, codes.n_tracks))
        prof = (listens @ codes.tags).tocsr()
        sums = np.asarray(prof.sum(axis=1)).ravel()
        sums[sums == 0] = 1.0
        self.profile = csr_matrix(prof.multiply(1.0 / sums[:, None]))

        users = split.users.set_index("user_id").reindex(self.user_ids)
        self.age = users["age"].to_numpy(float)
        self.age_bucket = users["age_bucket"].fillna("unknown").to_numpy()
        self.gender = users["gender"].fillna("unknown").to_numpy()


def build_features(pool: pd.DataFrame, queries: pd.DataFrame, stats: UserStats, codes: Codes,
                   priors: Priors) -> pd.DataFrame:
    """pool: qid, track_idx, rrf_score, n_sources. queries: qid, user_id, words
    (+ target_idx, если есть -> колонка label). Возвращает qid, track_idx, [label], FEATURES."""
    q_index = pd.Index(queries["qid"].to_numpy())
    qi = q_index.get_indexer(pool["qid"].to_numpy())
    if (qi < 0).any():
        raise ValueError("в пуле есть qid, которых нет в queries")
    ui = stats.user_ids.get_indexer(queries["user_id"].to_numpy())[qi]
    if (ui < 0).any():
        raise ValueError("в queries есть пользователи без статистики истории")
    ui64 = ui.astype(np.int64)
    t = pool["track_idx"].to_numpy(np.int64)
    art = codes.artist[t].astype(np.int64)
    alb = codes.album[t].astype(np.int64)
    lang = codes.lang[t].astype(np.int64)

    out = pd.DataFrame({"qid": pool["qid"].to_numpy(), "track_idx": t.astype(np.int32)})
    if "target_idx" in queries:
        out["label"] = (queries["target_idx"].to_numpy()[qi] == t).astype(np.int8)

    out["rrf_score"] = pool["rrf_score"].to_numpy(np.float32)
    out["n_sources"] = pool["n_sources"].to_numpy(np.int8)
    out["age_bucket"] = stats.age_bucket[ui]
    out["gender"] = stats.gender[ui]
    pop = codes.popularity[t]
    out["popularity"] = pop

    out["ctr_artist"] = lookup(priors, codes.artist_name[t])

    year = codes.year[t]
    out["age_at_release"] = stats.age[ui] - (WINDOW_YEAR - year)
    out["year_diff_history"] = np.abs(year - stats.mean_year[ui])

    cnt = _lookup(stats.ut[0], stats.ut[1], ui64 * codes.n_tracks + t, default=0.0)
    out["in_history"] = (cnt > 0).astype(np.int8)
    out["log_history_count"] = np.log1p(cnt)

    a_cnt = _lookup(stats.ua[0], stats.ua[1], ui64 * codes.n_artist + art, default=0.0)
    total = stats.total[ui]
    with np.errstate(invalid="ignore", divide="ignore"):
        out["artist_share"] = np.where(total > 0, a_cnt / total, 0.0)
    out["log_album_listens"] = np.where(
        alb >= 0, np.log1p(_lookup(stats.ual[0], stats.ual[1], ui64 * codes.n_album + alb, default=0.0)), 0.0)

    out["lang_share"] = np.where(lang >= 0, _lookup(stats.ul[0], stats.ul[1], ui64 * codes.n_lang + lang, default=0.0),
                                 np.nan)

    qw = codes.words_matrix(list(queries["words"]))
    n_words = np.asarray(qw.sum(axis=1)).ravel()
    hits = _rowdot(qw, qi, codes.tags, t)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["query_tag_frac"] = np.where(n_words[qi] > 0, hits / n_words[qi], 0.0)
    out["profile_tag_score"] = _rowdot(stats.profile, ui, codes.tags, t)

    for col, cats in CATEGORICAL.items():
        out[col] = pd.Categorical(out[col], categories=cats)
    for col in FEATURES:
        if col not in CATEGORICAL and out[col].dtype == np.float64:
            out[col] = out[col].astype(np.float32)
    return out
