from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from recsys.schemas import DialogSummary, Request, UserProfile


class BaseSummarizer(ABC):
    """Шаг 1: сырой диалог + user_info (+ профиль истории) -> DialogSummary."""

    @abstractmethod
    def summarize(self, request: Request, profile: Optional[UserProfile] = None) -> DialogSummary:
        ...
