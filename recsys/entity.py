"""Кто стоит за фразой из запроса: поиск по ней отдельно и согласие источников (summarizer.resolve_entities).

Правила разбора (dialog.py) узнают артиста по точному имени, тегу-прозвищу и опечатке. Но прозвища вроде «rhcp»
или «nin» знают башни — они выучили их на запросах датасета. Поэтому фразу после отрицания, которую правила не
узнали, ищем отдельно, без истории и без остального запроса:

  metadata_bm25          BM25 по метаданным треков: полные и частичные имена («chili peppers», «maiden»)
  twotower, twotower_lazy  башни с пустой историей (если включены в пайплайне): прозвища («rhcp» -> RHCP 19/20)

Источник даёт топ-k треков. Фраза — артист, если один источник уверен (одного артиста ≥ single из k) или два
источника сходятся на одном артисте (≥ pair из k у каждого). Обычные слова и жанры («rock», «sad songs»,
«matter») дают разброс и артистом не становятся.

Башни берутся у кандгенов пайплайна (Resources.towers, их заполняет Pipeline): модели второй раз не грузятся.
Ответы кэшируются по фразе.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from recsys.schemas import Context, DialogSummary, Message, Request, UserProfile

DEFAULTS = {"k": 20, "single": 14, "pair": 8, "cache": 10000}


class EntityResolver:
    def __init__(self, res, cfg: Optional[Dict[str, Any]] = None):
        self.res = res
        c = {**DEFAULTS, **(cfg or {})}
        self.k, self.single, self.pair, self.cache_size = int(c["k"]), int(c["single"]), int(c["pair"]), int(c["cache"])
        self._artists = res.catalog.df["artist"].to_numpy()
        self._cache: "OrderedDict[str, Optional[str]]" = OrderedDict()
        self.last: Dict[str, Tuple[str, int]] = {}   # источник -> (главный артист, сколько из k) у последней фразы

    def artist(self, phrase: str) -> Optional[str]:
        """Артист каталога, которого называет фраза, или None."""
        phrase = " ".join(phrase.split()).lower()
        if not phrase:
            return None
        if phrase in self._cache:
            self._cache.move_to_end(phrase)
            return self._cache[phrase]
        votes = {name: self._top_artist(ids) for name, ids in self._search(phrase).items() if len(ids)}
        self.last = votes
        out = self.decide(votes)
        self._cache[phrase] = out
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return out

    def decide(self, votes: Dict[str, Tuple[str, int]]) -> Optional[str]:
        """Один источник уверен (≥ single) или два сходятся на одном артисте (≥ pair у каждого)."""
        for artist, n in votes.values():
            if n >= self.single:
                return artist
        agree = Counter(artist for artist, n in votes.values() if n >= self.pair)
        best = agree.most_common(1)
        return best[0][0] if best and best[0][1] >= 2 else None

    def _top_artist(self, ids: np.ndarray) -> Tuple[str, int]:
        artist, n = Counter(self._artists[ids]).most_common(1)[0]
        return str(artist), int(n)

    def _search(self, phrase: str) -> Dict[str, np.ndarray]:
        out = {"metadata_bm25": self.res.text_index("metadata_bm25", None).search([phrase], self.k)[0].ids}
        towers = getattr(self.res, "towers", {})
        if towers:
            ctx = _bare_context(phrase, self.res)
            for name, cg in towers.items():
                if cg._encoder is None:          # online: false — модель всё равно нужна для фразы
                    cg._encoder = cg.encoder()
                u = np.asarray(cg._encoder.encode(cg.encoder_input(ctx)), dtype=np.float32)
                scores = cg.vectors @ (u / max(float(np.linalg.norm(u)), 1e-8))
                scores[~cg.known] = -np.inf
                out[name] = np.argsort(-scores)[:self.k]
        return out


def _bare_context(phrase: str, res) -> Context:
    """Фраза как отдельный запрос без истории, профиля и пользователя."""
    req = Request(dialog=[Message("user", phrase)], listens=([], []))
    return Context(req, DialogSummary(), UserProfile(), user=res.users.get(req))
