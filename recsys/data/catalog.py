"""Каталог треков: track_id -> название, артист, теги, жанры, популярность."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from recsys.text import normalize_tag, tokenize

LIST_COLUMNS = ["tags", "tag_weights", "genres"]
COLUMNS = ["track_id", "title", "artist", "tags", "tag_weights", "genres", "popularity"]


class Catalog:
    """Обёртка над DataFrame с колонками COLUMNS.

    tags отсортированы по убыванию веса, tag_weights в шкале Last.fm (0–100).
    """

    def __init__(self, df: pd.DataFrame, normalize: bool = True):
        df = df.copy()
        for col in ["title", "artist"]:
            if col not in df:
                df[col] = ""
            df[col] = df[col].fillna("").astype(str)
        for col in LIST_COLUMNS:
            if col not in df:
                df[col] = [[] for _ in range(len(df))]
            df[col] = df[col].map(_as_list)
        # теги без весов -> убывающие веса
        df["tag_weights"] = [
            [float(x) for x in w] if len(w) == len(t) else [100.0 * (0.9 ** i) for i in range(len(t))]
            for t, w in zip(df["tags"], df["tag_weights"])
        ]
        if normalize:
            pairs = [normalize_tag_list(t, w) for t, w in zip(df["tags"], df["tag_weights"])]
            df["tags"] = [p[0] for p in pairs]
            df["tag_weights"] = [p[1] for p in pairs]
            df["genres"] = [normalize_tag_list(g)[0] for g in df["genres"]]
        if "popularity" not in df:
            df["popularity"] = 0.0
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0).astype(float)
        df["track_id"] = df["track_id"].astype(str)
        df = df.drop_duplicates("track_id").reset_index(drop=True)
        self.df = df
        self._pos: Dict[str, int] = {tid: i for i, tid in enumerate(df["track_id"])}
        self._tag_index: Optional[Dict[str, np.ndarray]] = None
        self._artist_index: Optional[Dict[str, str]] = None

    # ------------------------------------------------------------ доступ

    def __len__(self) -> int:
        return len(self.df)

    def __contains__(self, track_id: str) -> bool:
        return track_id in self._pos

    @property
    def track_ids(self) -> np.ndarray:
        return self.df["track_id"].to_numpy()

    def pos(self, track_id: str) -> Optional[int]:
        return self._pos.get(str(track_id))

    def positions(self, track_ids: Iterable[str]) -> np.ndarray:
        return np.array([p for p in (self._pos.get(str(t)) for t in track_ids) if p is not None], dtype=np.int64)

    def row(self, track_id: str) -> Optional[dict]:
        p = self.pos(track_id)
        return None if p is None else self.df.iloc[p].to_dict()

    def tags(self, track_id: str) -> List[str]:
        p = self.pos(track_id)
        return [] if p is None else self.df.at[p, "tags"]

    def genres(self, track_id: str) -> List[str]:
        p = self.pos(track_id)
        return [] if p is None else self.df.at[p, "genres"]

    def artist(self, track_id: str) -> str:
        p = self.pos(track_id)
        return "" if p is None else self.df.at[p, "artist"]

    def get(self, track_ids: Iterable[str]) -> pd.DataFrame:
        return self.df.iloc[self.positions(track_ids)]

    @property
    def has_names(self) -> bool:
        return bool((self.df["artist"] != "").any())

    # ------------------------------------------------------------ индексы

    @property
    def tag_index(self) -> Dict[str, np.ndarray]:
        """тег/жанр -> позиции треков."""
        if self._tag_index is None:
            idx = defaultdict(list)
            for i, (tags, genres) in enumerate(zip(self.df["tags"], self.df["genres"])):
                for t in set(tags) | set(genres):
                    idx[t].append(i)
            self._tag_index = {t: np.array(v, dtype=np.int64) for t, v in idx.items()}
        return self._tag_index

    def tag_vocab(self, min_count: int = 2) -> Dict[str, int]:
        """Теги и жанры, встречающиеся хотя бы у min_count треков."""
        return {t: len(v) for t, v in self.tag_index.items() if len(v) >= min_count}

    @property
    def artist_index(self) -> Dict[str, str]:
        """нормализованное имя артиста ('the velvet owls') -> имя как в каталоге."""
        if self._artist_index is None:
            self._artist_index = {}
            for name in self.df["artist"].unique():
                key = " ".join(tokenize(name))
                if len(key) >= 2:
                    self._artist_index.setdefault(key, name)
        return self._artist_index

    def positions_with_any_tag(self, tags: Iterable[str]) -> np.ndarray:
        arrays = [self.tag_index[t] for t in tags if t in self.tag_index]
        return np.unique(np.concatenate(arrays)) if arrays else np.array([], dtype=np.int64)

    # ------------------------------------------------------------ сохранение

    def save(self, path: str) -> None:
        self.df[COLUMNS].to_parquet(path, index=False)

    @classmethod
    def load(cls, path: str) -> "Catalog":
        return cls(pd.read_parquet(path), normalize=False)

    def stats(self) -> pd.Series:
        n_tags = self.df["tags"].map(len)
        return pd.Series({
            "tracks": len(self),
            "with_names": int((self.df["artist"] != "").sum()),
            "with_tags": int((n_tags > 0).sum()),
            "with_genres": int((self.df["genres"].map(len) > 0).sum()),
            "mean_tags": round(float(n_tags.mean()), 2) if len(self) else 0.0,
            "with_popularity": int((self.df["popularity"] > 0).sum()),
            "unique_tags": len(self.tag_index),
        })


def _as_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    if isinstance(x, str):
        # "rock, indie" / "rock|indie"
        sep = "|" if "|" in x else ","
        return [s.strip() for s in x.split(sep) if s.strip()]
    return list(x)


def normalize_tag_list(tags: List[str], weights: Optional[List[float]] = None):
    """Нормализует теги, склеивает дубликаты (берёт максимум веса), сортирует по весу."""
    weights = weights if weights is not None and len(weights) == len(tags) else [100.0] * len(tags)
    merged: Counter = Counter()
    for t, w in zip(tags, weights):
        nt = normalize_tag(t)
        if nt:
            merged[nt] = max(merged[nt], float(w))
    items = merged.most_common()
    return [t for t, _ in items], [w for _, w in items]
