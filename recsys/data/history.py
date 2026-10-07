"""Профиль вкуса пользователя из истории прослушиваний."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List

from recsys.data.catalog import Catalog
from recsys.schemas import HistoryItem, UserProfile


def build_profile(history: List[HistoryItem], catalog: Catalog, max_tracks: int = 300,
                  max_tags_per_track: int = 10, recency_decay: float = 0.98) -> UserProfile:
    """Профиль = взвешенная сумма тегов/жанров/артистов последних max_tracks уникальных треков.

    Вес трека = log1p(count) * decay^(позиция по свежести); у 300-го трека он уже ~0.002.
    listened_ids содержит всю историю, включая неизвестные каталогу треки.
    """
    profile = UserProfile(listened_ids={h.track_id for h in history})
    if not history:
        return profile
    items = sorted(history, key=lambda h: h.timestamp or 0, reverse=True)[:max_tracks]
    tags_col, weights_col = catalog.df["tags"].to_numpy(), catalog.df["tag_weights"].to_numpy()
    genres_col, artist_col = catalog.df["genres"].to_numpy(), catalog.df["artist"].to_numpy()
    tags: Dict[str, float] = defaultdict(float)
    genres: Dict[str, float] = defaultdict(float)
    artists: Dict[str, float] = defaultdict(float)
    known = 0
    for h in items:
        p = catalog.pos(h.track_id)
        if p is None:
            continue
        w = math.log1p(max(h.count, 1.0)) * (recency_decay ** known)
        known += 1
        profile.track_weights[h.track_id] = w
        for t, tw in list(zip(tags_col[p], weights_col[p]))[:max_tags_per_track]:
            tags[t] += w * tw / 100.0
        for g in genres_col[p]:
            genres[g] += w
        if artist_col[p]:
            artists[artist_col[p]] += w
    profile.n_known_tracks = known
    profile.tag_weights = dict(tags)
    profile.genre_weights = dict(genres)
    profile.artist_weights = dict(artists)
    return profile
