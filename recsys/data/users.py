"""Состояние пользователя на момент запроса: что и сколько слушал до начала окна (Request.meta['cutoff']).

Общее для кандгенов истории (candgen/history.py, candgen/audio.py) и признаков ранкера (features.py); считается
один раз на пользователя (UserStates — кэш по user_id и cutoff).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, Optional, Tuple

import numpy as np

from recsys.data.catalog import Catalog
from recsys.data.parts import normalize_query
from recsys.schemas import Request

DAY = 86400


@dataclass
class UserState:
    positions: np.ndarray            # (n,) int32 — уникальные треки истории (позиции каталога)
    counts: np.ndarray               # (n,) float32 — сколько раз слушал
    days: np.ndarray                 # (n,) float64 — дней от последнего прослушивания до cutoff
    cutoff: int                      # начало окна (unix)
    artist_counts: Counter = field(default_factory=Counter)   # нормализованный артист -> прослушиваний
    centroid: Optional[np.ndarray] = None                     # MuQ: центр истории с весом counts (норм.)

    @property
    def total(self) -> float:
        return float(self.counts.sum())

    def weights(self, half_life: Optional[float] = None, log_count: bool = True) -> np.ndarray:
        """Вес трека истории: log1p(count) (или count) * 2^(-days / half_life)."""
        w = np.log1p(self.counts) if log_count else self.counts.astype(np.float64)
        return w * np.exp2(-self.days / half_life) if half_life else w

    @cached_property
    def retrieval_scores(self) -> Dict[int, float]:
        """Позиция -> log1p(count) * 2^(-days/180): «насколько трек свой» (признак ранкера)."""
        return dict(zip(self.positions.tolist(), self.weights(180).tolist()))


class UserStates:
    """Кэш состояний: (user_id, cutoff) -> UserState."""

    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.artists = artist_keys(catalog)
        self.embeddings = catalog.embeddings
        self._cache: Dict[Tuple[Optional[str], int], UserState] = {}

    def get(self, request: Request) -> UserState:
        cutoff = request.meta.get("cutoff")
        if cutoff is None:  # диалог: окна нет — после последнего прослушивания
            cutoff = max((h.timestamp or 0 for h in request.history), default=0) + 1
        key = (request.user_id, int(cutoff))
        if request.user_id is None or key not in self._cache:
            state = self._build(request, int(cutoff))
            if request.user_id is None:
                return state
            self._cache[key] = state
        return self._cache[key]

    def _build(self, request: Request, cutoff: int) -> UserState:
        pos, counts, last = [], [], []
        for h in request.history:
            p = self.catalog.pos(h.track_id)
            if p is None:
                continue
            pos.append(p)
            counts.append(h.count)
            last.append(h.timestamp if h.timestamp is not None else cutoff - 1)
        positions = np.array(pos, dtype=np.int32)
        counts_ = np.array(counts, dtype=np.float32)
        days = (cutoff - np.array(last, dtype=np.float64)) / DAY
        artist_counts: Counter = Counter()
        for p, c in zip(pos, counts):
            if self.artists[p]:
                artist_counts[self.artists[p]] += c
        centroid = None
        if self.embeddings is not None and len(positions):
            with np.errstate(all="ignore"):   # numpy 2.0 + Accelerate (macOS) даёт ложные warnings в matmul
                v = counts_ @ self.embeddings[positions]
            centroid = (v / max(float(np.linalg.norm(v)), 1e-8)).astype(np.float32)
        return UserState(positions, counts_, days, cutoff, artist_counts, centroid)


def artist_keys(catalog: Catalog) -> np.ndarray:
    """Нормализованное имя артиста каждого трека ('' — нет)."""
    return np.array([normalize_query(a) for a in catalog.text("artist")], dtype=object)
