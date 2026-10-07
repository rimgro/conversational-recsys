from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from recsys.data.catalog import Catalog
from recsys.schemas import Candidate, Context


class BaseRetriever(ABC):
    """Шаг 2: один источник кандидатов. search(ctx) -> [Candidate], отсортированы по score."""
    name: str = "base"

    def __init__(self, catalog: Catalog, top_k: int = 200, exclude_listened: bool = True):
        self.catalog = catalog
        self.top_k = top_k
        self.exclude_listened = exclude_listened

    @abstractmethod
    def search(self, ctx: Context, top_k: Optional[int] = None) -> List[Candidate]:
        ...

    def excluded_positions(self, ctx: Context) -> np.ndarray:
        """Что не возвращать: показанное и скипнутое в диалоге (+ прослушанное, если включено)."""
        ids = set(ctx.request.shown_ids) | set(ctx.request.skipped_ids)
        if self.exclude_listened:
            ids |= ctx.profile.listened_ids
        return self.catalog.positions(ids)

    def to_candidates(self, positions: np.ndarray, scores: np.ndarray) -> List[Candidate]:
        ids = self.catalog.track_ids
        return [Candidate(track_id=str(ids[p]), source=self.name, score=float(s), rank=r)
                for r, (p, s) in enumerate(zip(positions, scores), start=1)]
