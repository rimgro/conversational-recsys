"""Интерфейс кандгена и реестр.

Кандген получает пачку запросов (Context: запрос, разбор, состояние пользователя) и для каждого возвращает
Cands — позиции треков каталога (int32) и скоры (float32) по убыванию скора, не больше top_k.
Один запрос из диалога — пачка из одного Context: офлайн-оценка, обучение ранкера и онлайн работают одним кодом.

Кандгены ничего не фильтруют (прослушанное, показанное, артист образца): это делает отбор (selection.py),
чтобы список каждого кандгена можно было оценить сам по себе.

Новый кандген: класс-наследник Candgen с @register("имя") в модуле recsys/candgen/, параметры — в
candgen.sources.<имя> конфига (enabled, top_k и свои).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple, Type

import numpy as np

from recsys.schemas import Context

if TYPE_CHECKING:
    from recsys.resources import Resources


class Cands(NamedTuple):
    ids: np.ndarray      # (n,) int32 — позиции каталога, без повторов
    scores: np.ndarray   # (n,) float32 — по убыванию

    @classmethod
    def empty(cls) -> "Cands":
        return cls(np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.ids)


class Candgen(ABC):
    name: str = ""
    kind: str = ""        # text | history | audio | dense — группировка в отчётах

    def __init__(self, cfg: Dict[str, Any], res: "Resources"):
        self.cfg = cfg
        self.res = res
        self.top_k = int(cfg.get("top_k", 1000))

    @abstractmethod
    def retrieve(self, batch: List[Context]) -> List[Cands]:
        """По Cands на каждый Context пачки, в том же порядке."""

    def cache_key(self) -> Dict[str, Any]:
        """Всё, от чего зависит выдача (кроме самих запросов): ключ кэша кандидатов (candgen/cache.py).
        Кандгены, обученные на части index, добавляют её описание."""
        return {"name": self.name, "cfg": {k: v for k, v in self.cfg.items() if k != "enabled"}}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name}, top_k={self.top_k})"


REGISTRY: Dict[str, Type[Candgen]] = {}


def register(name: str, kind: str) -> Callable[[Type[Candgen]], Type[Candgen]]:
    def deco(cls: Type[Candgen]) -> Type[Candgen]:
        if name in REGISTRY:
            raise ValueError(f"кандген {name} уже зарегистрирован")
        cls.name, cls.kind = name, kind
        REGISTRY[name] = cls
        return cls
    return deco


def top_positions(scores: np.ndarray, k: int, positive: bool = True) -> Cands:
    """Лучшие k позиций по скору. positive — только скор > 0. При равенстве — меньшая позиция раньше
    (детерминированно; так же, как top_indices у бейзлайна team1)."""
    ids = np.flatnonzero(scores > 0) if positive else np.flatnonzero(np.isfinite(scores))
    if len(ids) > k:
        values = scores[ids]
        threshold = np.partition(values, -k)[-k]
        larger, tied = ids[values > threshold], ids[values == threshold]
        ids = np.concatenate([larger, tied[:k - len(larger)]])
    ids = ids[np.lexsort((ids, -scores[ids]))]
    return Cands(ids.astype(np.int32), scores[ids].astype(np.float32))


def ranked(ids: np.ndarray, weights: np.ndarray, k: int) -> Cands:
    """Готовый набор позиций с весами -> по убыванию веса (при равенстве — меньшая позиция), первые k."""
    ids = np.asarray(ids, dtype=np.int32)
    order = np.lexsort((ids, -weights))[:k]
    return Cands(ids[order], np.asarray(weights, dtype=np.float32)[order])
