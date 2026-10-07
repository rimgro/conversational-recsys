from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from recsys.schemas import Context, RankedTrack


class BaseExplainer(ABC):
    """Шаг 5: top-k треков + контекст -> текстовый ответ пользователю."""

    @abstractmethod
    def describe(self, tracks: List[RankedTrack], ctx: Context) -> str:
        ...
