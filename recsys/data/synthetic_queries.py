"""Запросы для оценки из одних прослушиваний (режим onion, когда диалогов ещё нет).

Для пользователя часть истории откладывается в цели, а реплика строится шаблоном из тегов
целевого трека. Метрики на таких запросах завышены: запрос содержит теги цели.
"""
from __future__ import annotations

from typing import List

import numpy as np

from recsys.data.catalog import Catalog
from recsys.data.history import InteractionStore
from recsys.schemas import Message, Request
from recsys.text import GENERIC_WORDS

# Теги Last.fm, которые не описывают музыку.
JUNK_TAGS = {"seen live", "favorites", "favourite", "favorite", "albums i own", "under 2000 listeners",
             "spotify", "my favorite", "awesome", "beautiful", "love", "loved", "good", "best"}

_TEMPLATES = [
    "I'm looking for something {a} and {b}.",
    "Can you recommend some {a} music? Preferably {b}.",
    "Play me some {a} tracks, something {b}.",
    "I want {a}, maybe a bit {b}.",
]


def make_requests_from_interactions(
    catalog: Catalog,
    interactions: InteractionStore,
    n_requests: int = 200,
    n_history: int = 20,
    n_targets: int = 5,
    seed: int = 42,
) -> List[Request]:
    rng = np.random.default_rng(seed)
    users = [u for u in interactions.users if interactions.n_tracks(u) >= n_history + n_targets]
    rng.shuffle(users)
    requests: List[Request] = []
    for uid in users:
        if len(requests) >= n_requests:
            break
        items = [h for h in interactions.history(uid, max_len=None) if h.track_id in catalog]
        items = list({h.track_id: h for h in items}.values())
        if len(items) < n_history + n_targets:
            continue
        order = rng.permutation(len(items))
        targets = [items[i] for i in order[:n_targets]]
        history = [items[i] for i in order[n_targets:n_targets + n_history]]
        tags = [t for t in catalog.tags(targets[0].track_id) if t not in JUNK_TAGS and t not in GENERIC_WORDS]
        if len(tags) < 2:
            continue
        text = str(rng.choice(_TEMPLATES)).format(a=tags[0], b=tags[1])
        requests.append(Request(
            dialog=[Message("user", text)],
            history=history,
            user_id=uid,
            request_id=f"onion_{uid}",
            target_ids=[t.track_id for t in targets],
        ))
    return requests
