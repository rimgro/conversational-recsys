"""Контракты между шагами пайплайна.

Request -> DialogSummary -> {source: [Candidate]} -> [FusedCandidate] -> features -> [RankedTrack] -> Response
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import pandas as pd


# ---------------------------------------------------------------- вход

@dataclass
class Message:
    role: str  # "user" | "assistant"
    text: str


@dataclass
class HistoryItem:
    """Уникальный трек из истории: сколько раз слушали и когда последний раз (unix-время)."""
    track_id: str
    count: float = 1.0
    timestamp: Optional[int] = None


@dataclass
class Request:
    dialog: List[Message] = field(default_factory=list)
    history: List[HistoryItem] = field(default_factory=list)
    user_info: str = ""                                   # текстовый профиль (user_profile в датасете)
    user_attrs: Dict[str, Any] = field(default_factory=dict)  # age, gender, country (user_demographics)
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    shown_ids: List[str] = field(default_factory=list)    # уже показаны в этом диалоге
    liked_ids: List[str] = field(default_factory=list)
    skipped_ids: List[str] = field(default_factory=list)
    target_ids: List[str] = field(default_factory=list)   # только для офлайн-оценки
    meta: Dict[str, Any] = field(default_factory=dict)    # query_family, is_new, split, ...

    @property
    def user_messages(self) -> List[str]:
        return [m.text for m in self.dialog if m.role == "user"]

    def next_turn(self, response: "Response", user_text: str) -> "Request":
        """Следующая реплика: добавляем ответ ассистента, показанные треки и новое сообщение."""
        req = copy.deepcopy(self)
        req.dialog.append(Message("assistant", response.text))
        req.dialog.append(Message("user", user_text))
        req.shown_ids = list(dict.fromkeys(req.shown_ids + response.track_ids))
        return req


# ---------------------------------------------------------------- шаг 1: саммари диалога

@dataclass
class DialogSummary:
    text: str = ""                       # саммари на английском
    query: str = ""                      # короткий поисковый запрос для BM25 / HNSW
    include_tags: List[str] = field(default_factory=list)
    exclude_tags: List[str] = field(default_factory=list)
    seed_artists: List[str] = field(default_factory=list)
    exclude_artists: List[str] = field(default_factory=list)
    seed_track_ids: List[str] = field(default_factory=list)
    mood: Optional[str] = None
    energy: Optional[str] = None         # low | medium | high
    countries: List[str] = field(default_factory=list)   # коды стран артиста: US, GB, ...
    languages: List[str] = field(default_factory=list)   # язык текста: en, ru, ...
    years: List[int] = field(default_factory=list)       # конкретные годы релиза
    user_tags: List[str] = field(default_factory=list)       # предпочтения из user_info
    user_attrs: Dict[str, Any] = field(default_factory=dict)  # age, gender, country
    source: str = "rule"                 # rule | llm | llm+rule

    @property
    def has_query(self) -> bool:
        return bool(self.query.strip() or self.include_tags or self.seed_artists or self.seed_track_ids)


@dataclass
class UserProfile:
    """Профиль вкуса, собранный из истории прослушиваний."""
    listened_ids: Set[str] = field(default_factory=set)
    tag_weights: Dict[str, float] = field(default_factory=dict)
    genre_weights: Dict[str, float] = field(default_factory=dict)
    artist_weights: Dict[str, float] = field(default_factory=dict)
    track_weights: Dict[str, float] = field(default_factory=dict)  # вес каждого учтённого трека истории
    n_known_tracks: int = 0

    @staticmethod
    def _top(d: Dict[str, float], n: int) -> List[str]:
        return [k for k, _ in sorted(d.items(), key=lambda kv: -kv[1])[:n]]

    def top_tags(self, n: int = 10) -> List[str]:
        return self._top(self.tag_weights, n)

    def top_genres(self, n: int = 5) -> List[str]:
        return self._top(self.genre_weights, n)

    def top_artists(self, n: int = 5) -> List[str]:
        return self._top(self.artist_weights, n)


@dataclass
class Context:
    """Всё, что знают шаги 2–5 о текущем запросе."""
    request: Request
    summary: DialogSummary
    profile: UserProfile
    banned_ids: Set[str] = field(default_factory=set)  # не рекомендовать: показанное, скипнутое, прослушанное

    @classmethod
    def build(cls, request: Request, summary: DialogSummary, profile: UserProfile,
              exclude_listened: bool = True) -> "Context":
        banned = set(request.shown_ids) | set(request.skipped_ids)
        if exclude_listened:
            banned |= profile.listened_ids
        return cls(request, summary, profile, banned)


# ---------------------------------------------------------------- шаги 2–3: кандидаты и слияние

@dataclass
class Candidate:
    track_id: str
    source: str
    score: float
    rank: int  # с 1


@dataclass
class FusedCandidate:
    track_id: str
    score: float                                       # RRF
    ranks: Dict[str, int] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------- шаги 4–5: выдача

@dataclass
class RankedTrack:
    track_id: str
    rank: int
    score: float
    title: str = ""
    artist: str = ""
    tags: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        if self.artist or self.title:
            return f"{self.artist or '?'} — {self.title or '?'}"
        return self.track_id


@dataclass
class Response:
    text: str
    tracks: List[RankedTrack]
    summary: DialogSummary
    debug: Dict[str, Any] = field(default_factory=dict)

    @property
    def track_ids(self) -> List[str]:
        return [t.track_id for t in self.tracks]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "rank": t.rank,
            "track_id": t.track_id,
            "artist": t.artist,
            "title": t.title,
            "score": round(t.score, 4),
            "tags": ", ".join(t.tags[:6]),
            "reasons": "; ".join(t.reasons),
        } for t in self.tracks])
