"""История прослушиваний: хранилище взаимодействий и профиль вкуса пользователя."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from recsys.data.catalog import Catalog
from recsys.schemas import HistoryItem, UserProfile


class InteractionStore:
    """Быстрый доступ к истории пользователя по user_id."""

    def __init__(self, df: pd.DataFrame):
        sort_cols = ["user_id", "timestamp"] if "timestamp" in df else ["user_id", "count"]
        self.df = df.sort_values(sort_cols, ascending=[True, False]).reset_index(drop=True)
        self._groups: Dict[str, np.ndarray] = self.df.groupby("user_id").indices

    def __len__(self) -> int:
        return len(self.df)

    @property
    def users(self) -> List[str]:
        return list(self._groups.keys())

    def n_tracks(self, user_id: str) -> int:
        idx = self._groups.get(str(user_id))
        return 0 if idx is None else len(idx)

    def history(self, user_id: str, max_len: Optional[int] = 30) -> List[HistoryItem]:
        """Самые свежие (или самые частые, если нет timestamp) max_len треков."""
        idx = self._groups.get(str(user_id))
        if idx is None:
            return []
        rows = self.df.iloc[idx[:max_len] if max_len else idx]
        stamps = rows["timestamp"].astype(str) if "timestamp" in rows else [None] * len(rows)
        return [
            HistoryItem(track_id=str(tid), count=float(cnt), timestamp=ts)
            for tid, cnt, ts in zip(rows["track_id"], rows["count"], stamps)
        ]


def build_profile(history: List[HistoryItem], catalog: Catalog, max_tags_per_track: int = 10,
                  recency_decay: float = 0.97) -> UserProfile:
    """Профиль = взвешенная сумма тегов/жанров/артистов треков из истории.

    Вес трека = log1p(count) * decay^(позиция по свежести). Неизвестные каталогу треки
    попадают только в listened_ids.
    """
    profile = UserProfile(listened_ids={h.track_id for h in history})
    if not history:
        return profile
    items = list(history)
    if all(h.timestamp for h in items):
        items.sort(key=lambda h: h.timestamp, reverse=True)
    tags: Dict[str, float] = defaultdict(float)
    genres: Dict[str, float] = defaultdict(float)
    artists: Dict[str, float] = defaultdict(float)
    for age, h in enumerate(items):
        p = catalog.pos(h.track_id)
        if p is None:
            continue
        profile.n_known_tracks += 1
        w = math.log1p(max(h.count, 1.0)) * (recency_decay ** age)
        row = catalog.df.iloc[p]
        for t, tw in list(zip(row["tags"], row["tag_weights"]))[:max_tags_per_track]:
            tags[t] += w * tw / 100.0
        for g in row["genres"]:
            genres[g] += w
        if row["artist"]:
            artists[row["artist"]] += w
    profile.tag_weights = dict(tags)
    profile.genre_weights = dict(genres)
    profile.artist_weights = dict(artists)
    return profile
