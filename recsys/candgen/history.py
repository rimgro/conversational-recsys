"""Кандгены по истории пользователя (не смотрят на текст запроса). Большинство целей датасета — повторные
прослушивания, поэтому история — главный источник для них. Как у бейзлайна team1.

  history            свои треки по log1p(count) * 2^(-дней / 180)
  history_recent     то же с полураспадом 30 дней: недавнее
  history_frequency  по числу прослушиваний
  favorite_artists   все треки каталога любимых артистов (по прослушиваниям артиста, затем по популярности)

Состояние пользователя (data/users.py) общее; выдача считается один раз на пользователя.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from recsys.candgen.base import Candgen, Cands, ranked, register
from recsys.data.users import UserState
from recsys.schemas import Context


class _PerUser(Candgen):
    """Выдача зависит только от пользователя: кэш по (user_id, cutoff)."""

    def __init__(self, cfg, res):
        super().__init__(cfg, res)
        self._memo: Dict[Tuple[Optional[str], int], Cands] = {}

    def retrieve(self, batch: List[Context]) -> List[Cands]:
        out = []
        for ctx in batch:
            key = (ctx.request.user_id, ctx.user.cutoff)
            if ctx.request.user_id is None:
                out.append(self.for_user(ctx.user))
                continue
            if key not in self._memo:
                self._memo[key] = self.for_user(ctx.user)
            out.append(self._memo[key])
        return out

    def for_user(self, user: UserState) -> Cands:
        raise NotImplementedError


class _History(_PerUser):
    half_life: Optional[float] = None   # дней; None — без затухания
    log_count = True

    def for_user(self, user: UserState) -> Cands:
        half_life = self.cfg.get("half_life", self.half_life)
        log_count = self.cfg.get("log_count", self.log_count)
        return ranked(user.positions, user.weights(half_life, log_count), self.top_k)


@register("history", "history")
class History(_History):
    """Свои треки: log1p(count) * 2^(-дней / 180)."""
    half_life = 180


@register("history_recent", "history")
class HistoryRecent(_History):
    """Свои треки: log1p(count) * 2^(-дней / 30)."""
    half_life = 30


@register("history_frequency", "history")
class HistoryFrequency(_History):
    """Свои треки по числу прослушиваний."""
    log_count = False


@register("favorite_artists", "history")
class FavoriteArtists(_PerUser):
    """Треки артистов истории: сначала артист, которого слушали больше, внутри — по spotify_popularity."""

    def __init__(self, cfg, res):
        super().__init__(cfg, res)
        self.popularity = np.nan_to_num(res.catalog.numeric("spotify_popularity"), nan=-1.0)
        by_artist = defaultdict(list)
        for p, a in enumerate(res.users.artists):
            if a:
                by_artist[a].append(p)
        self.by_artist = {a: np.array(v, dtype=np.int32) for a, v in by_artist.items()}

    def for_user(self, user: UserState) -> Cands:
        if not user.artist_counts:
            return Cands.empty()
        parts = [(self.by_artist[a], w) for a, w in user.artist_counts.items() if a in self.by_artist]
        ids = np.concatenate([p for p, _ in parts])
        weight = np.concatenate([np.full(len(p), w, dtype=np.float32) for p, w in parts])
        order = np.lexsort((ids, -self.popularity[ids], -weight))[:self.top_k]
        return Cands(ids[order], weight[order])
