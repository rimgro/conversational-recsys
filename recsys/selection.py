"""Шаг 3: отбор кандидатов для ранкера — объединение списков кандгенов, фильтры, взвешенный RRF, первые limit.

  1. union — все треки всех кандгенов; all_rrf = сумма 1 / (union_rrf_k + место) по всем спискам (признак)
  2. фильтры (selection.filters, по умолчанию выключены): жёсткие ограничения запроса из разбора
  3. score = сумма weight / (rrf_k + место) по кандгенам с весом в selection.weights; в ранкер идут треки
     с score > 0, первые limit по score (при равенстве — меньшая позиция)
  4. фильтры оставили меньше min_results — добираем остальными треками union по all_rrf

Одни веса для всех запросов: тип запроса (query_type) модели не известен — в датасете он только для метрик.
Отбор ничего не знает о цели запроса. Веса подбирает python -m recsys tune-policy (recsys/tune.py).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from recsys.candgen.base import Cands
from recsys.data.catalog import Catalog
from recsys.schemas import Context
from recsys.text import tag_matches

FILTERS = ("banned", "exclude_artist", "new_artist", "exclude_tag")


@dataclass
class Selection:
    ids: np.ndarray            # отобранные позиции (≤ limit), по возрастанию
    score: np.ndarray          # скор отбора (взвешенный RRF) у ids
    all_rrf: np.ndarray        # RRF всех кандгенов у ids
    union: np.ndarray          # все кандидаты после фильтров, по возрастанию
    n_union: int               # кандидатов до фильтров
    removed: Dict[str, np.ndarray]   # фильтр -> выкинутые им позиции (для отчёта: не выкинул ли он цель)


class Selector:
    def __init__(self, cfg: Dict[str, Any], catalog: Catalog, sources: List[str]):
        scfg = cfg["selection"]
        self.catalog = catalog
        self.n = len(catalog)
        self.limit = int(scfg.get("limit", 150))
        self.union_k = float(scfg.get("union_rrf_k", 60))
        self.min_results = int(scfg.get("min_results", 20))
        self.sources = list(sources)
        self.rrf_k = float(scfg.get("rrf_k", 60))
        self.weights = {s: float(w) for s, w in (scfg.get("weights") or {}).items() if w}
        missing = sorted(set(self.weights) - set(self.sources))
        if missing:
            warnings.warn(f"selection.weights: кандгены {missing} выключены, их веса не используются")
        # кандгены без веса в отбор не идут, но остаются признаками ранкера (так у team1 с audio_taste)
        self.unused = sorted(set(self.sources) - set(self.weights))
        unknown = set(scfg.get("filters") or []) - set(FILTERS)
        if unknown:
            raise ValueError(f"selection.filters: неизвестные {sorted(unknown)}, есть: {', '.join(FILTERS)}")
        self.filters = list(scfg.get("filters") or [])
        self.exclude_top_tags = int(scfg.get("exclude_top_tags", 3))
        self._artists = np.char.lower(catalog.text("artist").astype(str))

    def scores(self, lists: Dict[str, Cands], weights: Dict[str, float], k: float) -> np.ndarray:
        """Сумма weight / (k + место) по спискам, по всему каталогу."""
        s = np.zeros(self.n, dtype=np.float32)
        for name, w in weights.items():
            c = lists.get(name)
            if c is not None and len(c) and w:
                s[c.ids] += w / (k + np.arange(1, len(c) + 1))
        return s

    def select(self, ctx: Context, lists: Dict[str, Cands]) -> Selection:
        arrays = [c.ids for c in lists.values() if len(c)]
        union = np.unique(np.concatenate(arrays)).astype(np.int32) if arrays else np.zeros(0, dtype=np.int32)
        n_union = len(union)
        all_rrf = self.scores(lists, {name: 1.0 for name in lists}, self.union_k)
        removed: Dict[str, np.ndarray] = {}
        if self.filters and len(union):
            keep = np.ones(len(union), dtype=bool)
            for name in self.filters:
                drop = keep & self.filter_mask(name, ctx, union)
                if drop.any():
                    removed[name] = union[drop]
                    keep &= ~drop
            union = union[keep]
        score = self.scores(lists, {s: w for s, w in self.weights.items() if s in lists}, self.rrf_k)
        eligible = union[score[union] > 0]
        if len(eligible) > self.limit:
            threshold = np.partition(score[eligible], -self.limit)[-self.limit]
            above, tied = eligible[score[eligible] > threshold], eligible[score[eligible] == threshold]
            eligible = np.concatenate([above, tied[:self.limit - len(above)]])
        ids = np.sort(eligible).astype(np.int32)
        if self.filters and len(ids) < self.min_results:   # фильтры выкинули почти всё: добираем из union
            rest = np.setdiff1d(union, ids)
            rest = rest[np.lexsort((rest, -all_rrf[rest]))][:self.min_results - len(ids)]
            ids = np.sort(np.concatenate([ids, rest])).astype(np.int32)
        return Selection(ids, score[ids], all_rrf[ids], union, n_union, removed)

    # ------------------------------------------------------------ фильтры

    def filter_mask(self, name: str, ctx: Context, pos: np.ndarray) -> np.ndarray:
        """True — трек выкидывается фильтром name.

        banned          ctx.banned_ids: показанное, скипнутое; прослушанное для similar_to / novelty (summary.new_tracks)
        exclude_artist  артист из summary.exclude_artists («like X by Y» -> Y, «no Coldplay»)
        new_artist      novelty (summary.new_artists): артист есть в истории пользователя
        exclude_tag     тег трека совпал с summary.exclude_tags («no rap»): только среди первых exclude_top_tags тегов
                        по весу (0 — любой тег или жанр): «no pop» не выкидывает прог-рок, у которого pop восьмой
        """
        s, catalog = ctx.summary, self.catalog
        if name == "banned":
            return np.isin(pos, catalog.positions(ctx.banned_ids)) if ctx.banned_ids else np.zeros(len(pos), bool)
        if name == "exclude_artist":
            names = {a.lower() for a in s.exclude_artists}
            return np.isin(self._artists[pos], list(names)) if names else np.zeros(len(pos), bool)
        if name == "new_artist":
            if not s.new_artists:
                return np.zeros(len(pos), bool)
            history = {self._artists[p] for p in catalog.positions(h.track_id for h in ctx.request.history)} - {""}
            return np.isin(self._artists[pos], list(history))
        if name == "exclude_tag":
            if not s.exclude_tags:
                return np.zeros(len(pos), bool)
            tags, genres = catalog.df["tags"].to_numpy(), catalog.df["genres"].to_numpy()
            n = self.exclude_top_tags
            out = np.zeros(len(pos), bool)
            for i, p in enumerate(pos):
                track_tags = list(tags[p][:n]) if n else list(tags[p]) + list(genres[p])
                out[i] = any(tag_matches(t, track_tags) for t in s.exclude_tags)
            return out
        raise ValueError(name)


def selection_rank(sel: Selection) -> np.ndarray:
    """Место трека в отборе (1 — лучший по скору отбора; при равенстве — меньшая позиция)."""
    order = np.lexsort((sel.ids, -sel.score))
    ranks = np.empty(len(sel.ids), dtype=np.float32)
    ranks[order] = np.arange(1, len(sel.ids) + 1)
    return ranks


__all__ = ["Selection", "Selector", "FILTERS", "selection_rank"]
