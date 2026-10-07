from __future__ import annotations

from collections import Counter

import pandas as pd

from recsys.data.catalog import Catalog


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
