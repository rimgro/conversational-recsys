from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from recsys.schemas import Context


class BaseRanker(ABC):
    """Шаг 4: таблица признаков top-N кандидатов -> та же таблица с rank_score, по убыванию.

    Отрезать top-k и диверсифицировать будет пайплайн.
    """

    @abstractmethod
    def rank(self, features: pd.DataFrame, ctx: Context) -> pd.DataFrame:
        ...
