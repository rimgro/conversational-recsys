"""Шаг 4: признаки top-N кандидатов -> ранкер -> диверсификация (делает пайплайн).

Ранкер: rank(features, ctx) -> та же таблица с колонкой rank_score, по убыванию.
  StubRanker       заглушка: порядок RRF (на схеме «top-20 первых»)
  HeuristicRanker  взвешенная сумма признаков, бейзлайн до CatBoost
  APIRanker        «ранкер по API»: POST признаков на внешний сервис, при ошибке -> StubRanker
"""

from __future__ import annotations

import json
import math
import urllib.request
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from recsys.data.catalog import Catalog
from recsys.schemas import Context, FusedCandidate
from recsys.text import decade_tag, tag_matches


MISSING_RANK = 1000
_DECADES = {f"{d}0s" for d in range(10)}


def build_features(fused: List[FusedCandidate], ctx: Context, catalog: Catalog,
                   sources: Sequence[str]) -> pd.DataFrame:
    s, prof = ctx.summary, ctx.profile
    prof_total = sum(prof.tag_weights.values()) or 1.0
    seeds = {a.lower() for a in s.seed_artists}
    history = {h.track_id: h.count for h in ctx.request.history}
    countries = set(s.countries)
    decades = {t for t in s.include_tags if t in _DECADES}
    rows = []
    for fc in fused:
        p = catalog.pos(fc.track_id)
        row = catalog.df.iloc[p]
        tags = list(row["tags"]) + list(row["genres"])
        artist = row["artist"]
        year = row.get("release_year")
        has_year = year is not None and not pd.isna(year) and int(year) > 1900
        q_hits = sum(tag_matches(t, tags) for t in s.include_tags)
        u_hits = sum(tag_matches(t, tags) for t in s.user_tags)
        feat = {
            "track_id": fc.track_id,
            "rrf_score": fc.score,
            "n_sources": len(fc.ranks),
            "log_popularity": math.log1p(float(row["popularity"])),
            "query_tag_hits": q_hits,
            "query_tag_frac": q_hits / len(s.include_tags) if s.include_tags else 0.0,
            "user_tag_hits": u_hits,
            "profile_tag_score": sum(prof.tag_weights.get(t, 0.0) for t in tags) / prof_total,
            "artist_listened": float(prof.artist_weights.get(artist, 0.0) > 0) if artist else 0.0,
            "artist_seed": float(artist.lower() in seeds) if artist else 0.0,
            "n_tags": len(row["tags"]),
            "in_history": float(fc.track_id in history),
            "log_history_count": math.log1p(history.get(fc.track_id, 0.0)),
            "country_match": float(bool(countries) and row.get("artist_country") in countries),
            "decade_match": float(bool(decades) and has_year and decade_tag(int(year)) in decades),
        }
        for src in sources:
            feat[f"rank_{src}"] = fc.ranks.get(src, MISSING_RANK)
            feat[f"score_{src}"] = fc.scores.get(src, 0.0)
        rows.append(feat)
    return pd.DataFrame(rows)


def reasons_for(track_id: str, ctx: Context, catalog: Catalog, max_reasons: int = 3) -> List[str]:
    """Короткие причины для объяснения (по-английски): совпавшие теги, артист, история."""
    tags = catalog.tags(track_id) + catalog.genres(track_id)
    artist = catalog.artist(track_id)
    matched = [t for t in ctx.summary.include_tags if tag_matches(t, tags)]
    out = [f"matches: {', '.join(matched)}"] if matched else []
    if artist and artist.lower() in {a.lower() for a in ctx.summary.seed_artists}:
        out.append(f"{artist}, as you asked")
    elif any(h.track_id == track_id for h in ctx.request.history):
        out.append("you have listened to it before")
    elif artist and artist in ctx.profile.artist_weights:
        out.append(f"you listen to {artist}")
    if not out:
        common = [t for t in ctx.profile.top_tags(10) if t in tags][:2]
        if common:
            out.append("close to your taste: " + ", ".join(common))
    return out[:max_reasons]


def cap_per_artist(ranked: pd.DataFrame, catalog: Catalog, max_per_artist: int = 2) -> pd.DataFrame:
    """Не больше max_per_artist треков одного артиста; лишние уходят в конец, а не удаляются."""
    if not max_per_artist or ranked.empty:
        return ranked
    seen: Counter = Counter()
    head, tail = [], []
    for i, tid in enumerate(ranked["track_id"]):
        a = catalog.artist(tid) or tid
        seen[a] += 1
        (head if seen[a] <= max_per_artist else tail).append(i)
    return ranked.iloc[head + tail].reset_index(drop=True)


class BaseRanker(ABC):
    """Таблица признаков top-N кандидатов -> та же таблица с rank_score, по убыванию."""

    @abstractmethod
    def rank(self, features: pd.DataFrame, ctx: Context) -> pd.DataFrame:
        ...


def with_scores(features: pd.DataFrame, scores) -> pd.DataFrame:
    """Добавляет rank_score и сортирует по убыванию (порядок при равенстве сохраняется)."""
    out = features.assign(rank_score=list(scores))
    return out.sort_values("rank_score", ascending=False, kind="stable").reset_index(drop=True)


class StubRanker(BaseRanker):
    def rank(self, features: pd.DataFrame, ctx: Context) -> pd.DataFrame:
        return with_scores(features, features["rrf_score"])


DEFAULT_WEIGHTS: Dict[str, float] = {
    "rrf_score": 30.0,          # RRF ~ 0.01–0.05
    "query_tag_frac": 1.0,
    "user_tag_hits": 0.1,
    "profile_tag_score": 0.5,
    "artist_seed": 1.0,
    "artist_listened": 0.2,
    "log_popularity": 0.03,
    "n_sources": 0.05,
    "in_history": 0.3,
    "log_history_count": 0.05,
    "country_match": 0.3,
    "decade_match": 0.3,
}


class HeuristicRanker(BaseRanker):
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    def rank(self, features: pd.DataFrame, ctx: Context) -> pd.DataFrame:
        score = pd.Series(0.0, index=features.index)
        for f, w in self.weights.items():
            if f in features:
                score += w * features[f].astype(float)
        return with_scores(features, score)


class APIRanker(BaseRanker):
    def __init__(self, url: str, timeout: float = 2.0, api_key: Optional[str] = None,
                 fallback: Optional[BaseRanker] = None):
        self.url = url
        self.timeout = timeout
        self.api_key = api_key
        self.fallback = fallback or StubRanker()
        self.last_error: Optional[str] = None

    def rank(self, features: pd.DataFrame, ctx: Context) -> pd.DataFrame:
        payload = {
            "features": json.loads(features.to_json(orient="records", double_precision=15)),
            "context": {"summary": ctx.summary.text, "include_tags": ctx.summary.include_tags,
                        "user_id": ctx.request.user_id},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        try:
            req = urllib.request.Request(self.url, data=json.dumps(payload).encode("utf8"), headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                scores = json.loads(r.read().decode("utf8"))["scores"]
            if len(scores) != len(features):
                raise ValueError(f"ожидали {len(features)} скоров, получили {len(scores)}")
        except Exception as e:
            self.last_error = str(e)
            return self.fallback.rank(features, ctx)
        return with_scores(features, [float(x) for x in scores])


def build_ranker(cfg: Dict[str, Any]) -> BaseRanker:
    rcfg = cfg.get("ranker", {})
    kind = rcfg.get("type", "stub")
    if kind == "stub":
        return StubRanker()
    if kind == "heuristic":
        return HeuristicRanker(rcfg.get("weights"))
    if kind == "api":
        return APIRanker(rcfg["api_url"], timeout=rcfg.get("timeout", 2.0), api_key=rcfg.get("api_key"))
    raise ValueError(f"Неизвестный ranker.type: {kind}")
