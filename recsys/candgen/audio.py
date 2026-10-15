"""Кандгены по звучанию: косинус MuQ-эмбеддингов (Catalog.embeddings, нормированы) к вектору-запросу.

  audio_taste   центр истории с весом count                             (team1)
  audio_recent  центр истории с весом log1p(count) * 2^(-дней / 30)     (team1)
  audio_seed    образец из запроса: трек «like X by Y» (similar_to), иначе треки названных артистов и лайки
                (from_artists: false — только трек «like X by Y»); без образца — пусто
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from recsys.candgen.base import Candgen, Cands, register, top_positions
from recsys.candgen.history import _PerUser
from recsys.data.users import UserState
from recsys.schemas import Context


def nearest(embeddings: np.ndarray, weights: np.ndarray, positions: np.ndarray, k: int) -> Cands:
    """Треки с положительным косинусом к нормированной взвешенной сумме эмбеддингов positions, top k."""
    if not len(positions):
        return Cands.empty()
    with np.errstate(all="ignore"):   # numpy 2.0 + Accelerate (macOS) даёт ложные warnings в matmul
        v = weights.astype(np.float32) @ embeddings[positions]
        v /= max(float(np.linalg.norm(v)), 1e-8)
        scores = embeddings @ v
    return top_positions(scores, k)


class _Taste(_PerUser):
    half_life: Optional[float] = None
    log_count = False

    def __init__(self, cfg, res):
        super().__init__(cfg, res)
        if res.catalog.embeddings is None:
            raise ValueError(f"{self.name}: в каталоге нет эмбеддингов MuQ")

    def for_user(self, user: UserState) -> Cands:
        w = user.weights(self.cfg.get("half_life", self.half_life), self.cfg.get("log_count", self.log_count))
        return nearest(self.res.catalog.embeddings, w, user.positions, self.top_k)


@register("audio_taste", "audio")
class AudioTaste(_Taste):
    """Похожие по звучанию на всю историю (вес — число прослушиваний)."""


@register("audio_recent", "audio")
class AudioRecent(_Taste):
    """Похожие по звучанию на недавнее (log1p(count) * 2^(-дней / 30))."""
    half_life = 30
    log_count = True


@register("audio_seed", "audio")
class AudioSeed(Candgen):
    """Похожие по звучанию на образец из запроса: трек «like X by Y», иначе треки названных артистов и лайки."""

    def __init__(self, cfg, res):
        super().__init__(cfg, res)
        if res.catalog.embeddings is None:
            raise ValueError("audio_seed: в каталоге нет эмбеддингов MuQ")

    def seeds(self, ctx: Context) -> np.ndarray:
        catalog = self.res.catalog
        tracks = catalog.positions(ctx.summary.seed_track_ids)
        if len(tracks) or not self.cfg.get("from_artists", True):   # from_artists: false — только «like X by Y»
            return tracks
        extra = list(catalog.positions(ctx.request.liked_ids))
        names = {a.lower() for a in ctx.summary.seed_artists}
        if names:
            extra += list(np.flatnonzero(catalog.df["artist"].str.lower().isin(names).to_numpy()))
        return np.array(extra, dtype=np.int64)

    def retrieve(self, batch: List[Context]) -> List[Cands]:
        emb = self.res.catalog.embeddings
        out = []
        for ctx in batch:
            pos = self.seeds(ctx)
            out.append(nearest(emb, np.ones(len(pos)), pos, self.top_k))
        return out

    def cache_key(self):
        from recsys.dialog import rules_key
        return {**super().cache_key(), "summarizer": self.res.cfg.get("summarizer", {}).get("type"),
                "rules": rules_key(self.res.cfg)}
