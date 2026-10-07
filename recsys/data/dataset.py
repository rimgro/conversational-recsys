"""Адаптер нашего датасета с диалогами.

Датасета ещё нет, поэтому формат ниже ПРЕДПОЛОЖЕНИЕ. Если реальные поля называются иначе,
поменяйте только `data.dataset.fields` в configs/default.yaml (или эту функцию).

dialogs.jsonl, одна строка = один диалог:
{
  "dialog_id": "d_0001",
  "user_id": "92915",
  "user_info": "female, 23 years old, from Brazil. Favourite genres: indie rock",
  "history": [{"track_id": "0010xmHR6UICBOYT", "count": 3}, ...],   # или нет, тогда берём по user_id
  "messages": [{"role": "user", "text": "..."}, {"role": "assistant", "text": "..."}],
  "target_track_ids": ["..."]                                        # что пользователь в итоге выбрал
}

tracks.(csv|parquet|jsonl): track_id, title, artist [, tags, genres, popularity]
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from recsys.data.history import InteractionStore
from recsys.schemas import HistoryItem, Message, Request

DEFAULT_FIELDS: Dict[str, str] = {
    "dialog_id": "dialog_id",
    "user_id": "user_id",
    "user_info": "user_info",
    "history": "history",
    "messages": "messages",
    "role": "role",
    "text": "text",
    "targets": "target_track_ids",
}


def load_dialogs(path: str) -> List[Dict[str, Any]]:
    """jsonl (по строке на диалог) или json со списком диалогов."""
    with open(path, encoding="utf8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        data = json.load(f)
    return data if isinstance(data, list) else data.get("dialogs", [])


def record_to_request(
    rec: Dict[str, Any],
    fields: Optional[Dict[str, str]] = None,
    interactions: Optional[InteractionStore] = None,
    max_history: int = 30,
) -> Request:
    f = {**DEFAULT_FIELDS, **(fields or {})}
    messages = [
        Message(role=str(m.get(f["role"], "user")), text=str(m.get(f["text"], "")))
        for m in rec.get(f["messages"], []) or []
    ]
    user_id = rec.get(f["user_id"])
    user_id = None if user_id is None else str(user_id)

    history: List[HistoryItem] = []
    for h in rec.get(f["history"], []) or []:
        if isinstance(h, dict):
            history.append(HistoryItem(str(h["track_id"]), float(h.get("count", 1.0)), h.get("timestamp")))
        else:
            history.append(HistoryItem(str(h)))
    if not history and interactions is not None and user_id is not None:
        history = interactions.history(user_id, max_len=max_history)

    return Request(
        dialog=messages,
        history=history[:max_history] if max_history else history,
        user_info=str(rec.get(f["user_info"], "") or ""),
        user_id=user_id,
        request_id=rec.get(f["dialog_id"]),
        target_ids=[str(t) for t in rec.get(f["targets"], []) or []],
    )
