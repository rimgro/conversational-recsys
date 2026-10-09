"""LGBMRanker: обученный LightGBM LambdaRank поверх пула bm25_cards + hnsw (ranker.type: lgbm).

Обучение — ветка dev-ranker (ranker/pit.py, ranker/fit.py); здесь только инференс для одного запроса.
Признаки считаются так же, как ranker/pit.py в режиме test: срез истории на момент запроса без событий окна.

Папка модели (ranker.model_dir):
  model.txt       LightGBM (текст)
  features.json   порядок признаков и категории (age_bucket, gender, top_genre)
  tracks.parquet  m4a_id, artist, album, year, lang, popularity, genres, tags — как при обучении
  items.parquet   m4a_id, ctr_item, ctr_artist — айтемные признаки на срез тестового окна

Пул должен совпадать с обучением (configs/ranker_lgbm.yaml): bm25_cards + hnsw по 100, RRF k = 60 без весов, top-200.
Любая ошибка (нет файлов, нет lightgbm, неожиданные данные) -> порядок RRF и предупреждение.
"""
from __future__ import annotations

import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from recsys.ranking import MISSING_RANK, BaseRanker, StubRanker, with_scores
from recsys.schemas import Context

WINDOW_YEAR = 2020
_TOKEN = re.compile(r"[^\w']+")


def age_bucket(age: Optional[float]) -> str:
    if age is None or pd.isna(age) or age < 10 or age > 100:
        return "unknown"
    return "50+" if age >= 50 else f"{int(age) // 10 * 10}s"


def query_ngrams(text: str, vocab: set) -> set:
    toks = [t for t in _TOKEN.split((text or "").lower()) if t]
    grams = {" ".join(toks[i:i + n]) for n in (1, 2, 3) for i in range(len(toks) - n + 1)}
    return grams & vocab


class LGBMRanker(BaseRanker):
    def __init__(self, model_dir: str, score_sources: Optional[Dict[str, str]] = None,
                 fallback: Optional[BaseRanker] = None):
        import lightgbm as lgb

        path = Path(model_dir)
        self.booster = lgb.Booster(model_str=(path / "model.txt").read_text(encoding="utf-8"))
        meta = json.loads((path / "features.json").read_text(encoding="utf-8"))
        self.features: List[str] = meta["features"]
        self.categorical: Dict[str, List[str]] = meta["categorical"]
        # признак score_<x> модели -> имя источника в пайплайне
        self.score_sources = score_sources or {"score_hnsw": "hnsw", "score_bm25": "bm25_cards"}
        tracks = pd.read_parquet(path / "tracks.parquet")
        items = pd.read_parquet(path / "items.parquet")
        self.tracks = tracks.merge(items, on="m4a_id", how="left").set_index("m4a_id")
        self.vocab = {t for tags in self.tracks["tags"] for t in tags}
        self.fallback = fallback or StubRanker()
        self.last_error: Optional[str] = None

    # ---------------------------------------------------------------- пользователь

    def _user(self, ctx: Context) -> Dict[str, Any]:
        hist = [(h.track_id, float(h.count)) for h in ctx.request.history if h.track_id in self.tracks.index]
        h = self.tracks.loc[[t for t, _ in hist], ["artist", "album", "year", "lang", "genres", "tags"]].copy()
        h["cnt"] = [c for _, c in hist]
        genres: Counter = Counter()
        tags: Counter = Counter()
        for gs, ts, c in zip(h["genres"], h["tags"], h["cnt"]):
            for g in gs:
                genres[g] += c
            for t in ts:
                tags[t] += c
        known_year = h["year"].notna()
        known_lang = h["lang"] != ""
        top = sorted(genres.items(), key=lambda kv: (-kv[1], kv[0]))
        top_cats = set(self.categorical.get("top_genre", [])) - {"other", "unknown"}
        age = ctx.request.user_attrs.get("age")
        age = float(age) if age is not None and float(age) > 0 else None
        gender = ctx.request.user_attrs.get("gender")
        return {
            "track": dict(hist),
            "artist": h.groupby("artist")["cnt"].sum().to_dict(),
            "album": h[h["album"].notna()].groupby("album")["cnt"].sum().to_dict(),
            "lang": h[known_lang].groupby("lang")["cnt"].sum().to_dict(),
            "lang_total": float(h.loc[known_lang, "cnt"].sum()),
            "genre": genres, "tag": tags,
            "tag_total": float(sum(c * len(t) for t, c in zip(h["tags"], h["cnt"]))),
            "total": float(h["cnt"].sum()),
            "mean_year": float((h.loc[known_year, "year"] * h.loc[known_year, "cnt"]).sum()
                               / h.loc[known_year, "cnt"].sum()) if known_year.any() else np.nan,
            "age": age,
            "age_bucket": age_bucket(age),
            "gender": gender if gender in ("m", "f") else "unknown",
            "top_genre": "unknown" if not top else (top[0][0] if top[0][0] in top_cats else "other"),
        }

    # ---------------------------------------------------------------- признаки

    def compute_features(self, features: pd.DataFrame, ctx: Context) -> pd.DataFrame:
        """features пайплайна (track_id, rrf_score, n_sources, rank_<src>, score_<src>, ...) -> признаки модели."""
        u = self._user(ctx)
        ids = features["track_id"].astype(str).tolist()
        t = self.tracks.reindex(ids)
        x = pd.DataFrame(index=features.index)
        x["rrf_score"] = features["rrf_score"].astype(float).to_numpy()
        x["n_sources"] = features["n_sources"].astype(float).to_numpy()
        for feat, src in self.score_sources.items():
            if f"score_{src}" in features:
                found = features[f"rank_{src}"].to_numpy() != MISSING_RANK
                x[feat] = np.where(found, features[f"score_{src}"].astype(float).to_numpy(), np.nan)
            else:
                x[feat] = np.nan
        x["age_bucket"] = u["age_bucket"]
        x["gender"] = u["gender"]
        x["top_genre"] = u["top_genre"]
        x["popularity"] = t["popularity"].to_numpy(dtype=float)
        x["ctr_item"] = t["ctr_item"].to_numpy(dtype=float)
        x["ctr_artist"] = t["ctr_artist"].to_numpy(dtype=float)
        year = t["year"].to_numpy(dtype=float)
        x["age_at_release"] = (u["age"] if u["age"] is not None else np.nan) - (WINDOW_YEAR - year)
        x["year_diff_history"] = np.abs(year - u["mean_year"])
        cnt = np.array([u["track"].get(i, 0.0) for i in ids])
        x["in_history"] = (cnt > 0).astype(float)
        x["log_history_count"] = np.log1p(cnt)
        total = u["total"]
        a_cnt = np.array([u["artist"].get(a, 0.0) for a in t["artist"].fillna("")])
        x["artist_share"] = a_cnt / total if total > 0 else 0.0
        x["log_album_listens"] = np.log1p([u["album"].get(a, 0.0) if isinstance(a, str) else 0.0
                                           for a in t["album"]])
        langs = t["lang"].fillna("").tolist()
        x["lang_share"] = [np.nan if not lg else (u["lang"].get(lg, 0.0) / u["lang_total"] if u["lang_total"] > 0
                                                  else 0.0) for lg in langs]
        genres = [g if isinstance(g, (list, np.ndarray)) else [] for g in t["genres"]]
        tags = [g if isinstance(g, (list, np.ndarray)) else [] for g in t["tags"]]
        x["genre_share"] = [max((u["genre"].get(g, 0.0) for g in gs), default=0.0) / total if total > 0 else 0.0
                            for gs in genres]
        x["profile_tag_score"] = [sum(u["tag"].get(g, 0.0) for g in ts) / u["tag_total"] if u["tag_total"] > 0
                                  else 0.0 for ts in tags]
        messages = ctx.request.user_messages
        grams = query_ngrams(messages[-1] if messages else "", self.vocab)
        x["query_tag_frac"] = [len(grams.intersection(ts)) / len(grams) if grams else 0.0 for ts in tags]
        for col, cats in self.categorical.items():
            vals = x[col].where(x[col].isin(cats), "other" if "other" in cats else None)
            x[col] = pd.Categorical(vals, categories=cats)
        return x[self.features]

    def rank(self, features: pd.DataFrame, ctx: Context) -> pd.DataFrame:
        try:
            scores = self.booster.predict(self.compute_features(features, ctx))
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            warnings.warn(f"LGBMRanker: {self.last_error}; порядок RRF")
            return self.fallback.rank(features, ctx)
        return with_scores(features, scores)
