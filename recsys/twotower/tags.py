"""Теги треков для двухбашенного кандгена (arch: v2): Last.fm-теги и жанры каталога -> словарь и веса.

Словарь — теги и жанры (общие названия — один элемент), встречающиеся хотя бы у min_count треков, по убыванию
частоты; элемент 0 — «<no_tags>» (у трека не осталось тегов из словаря). Вес тега — вес Last.fm / 100, жанра —
genre_weight (у тега-жанра — сумма); веса трека нормированы на сумму 1, чтобы вектор тегов не зависел от их числа.

  idx  (n_tracks, max_per_track) int64 — номера в словаре, пустые места — pad = len(vocab)
  w    (n_tracks, max_per_track) float32 — веса, у пустых мест 0
Строки — в порядке track_ids (номер трека = строка semantic_codes.npy).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

NO_TAGS = "<no_tags>"


@dataclass
class TrackTags:
    vocab: List[str]
    idx: np.ndarray
    w: np.ndarray

    @property
    def pad(self) -> int:
        return len(self.vocab)


def build_track_tags(catalog_df: pd.DataFrame, track_ids: Sequence[str], min_count: int = 5,
                     max_per_track: int = 32, genre_weight: float = 0.5) -> TrackTags:
    """catalog_df — Catalog.df (track_id, tags, tag_weights, genres; теги уже нормализованы)."""
    df = catalog_df.drop_duplicates("track_id").set_index("track_id").reindex([str(t) for t in track_ids])
    per_track: List[Dict[str, float]] = []
    for tags, weights, genres in zip(df["tags"], df["tag_weights"], df["genres"]):
        d: Dict[str, float] = {}
        if isinstance(tags, (list, np.ndarray)):
            for t, w in zip(tags, weights):
                d[t] = d.get(t, 0.0) + float(w) / 100.0
        if isinstance(genres, (list, np.ndarray)):
            for g in genres:
                d[g] = d.get(g, 0.0) + genre_weight
        per_track.append(d)
    count = Counter(t for d in per_track for t in d)
    vocab = [NO_TAGS] + sorted((t for t, c in count.items() if c >= min_count), key=lambda t: (-count[t], t))
    pos = {t: i for i, t in enumerate(vocab)}
    pad = len(vocab)
    idx = np.full((len(per_track), max_per_track), pad, dtype=np.int64)
    w = np.zeros((len(per_track), max_per_track), dtype=np.float32)
    for i, d in enumerate(per_track):
        items = sorted(((pos[t], x) for t, x in d.items() if t in pos and x > 0), key=lambda p: -p[1])[:max_per_track]
        if not items:
            items = [(0, 1.0)]
        total = sum(x for _, x in items)
        for j, (k, x) in enumerate(items):
            idx[i, j], w[i, j] = k, x / total
    return TrackTags(vocab, idx, w)
