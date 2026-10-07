"""Саммари диалога по правилам: заглушка вместо LLM и её фолбэк.

Ищет в репликах теги каталога и имена артистов (n-граммы), отрицания
('no rap', 'without vocals', 'not pop') превращает в исключения, позже сказанное
перекрывает ранее сказанное.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from recsys.data.catalog import Catalog
from recsys.dialog.base import BaseSummarizer
from recsys.schemas import DialogSummary, Request, UserProfile
from recsys.text import GENERIC_WORDS, NEGATIONS, STOPWORDS, content_tokens, split_clauses, tokenize

ENERGY_WORDS: Dict[str, str] = {
    **{w: "low" for w in "calm chill chilled relax relaxing relaxed quiet soft sleep sleeping study studying "
                         "focus slow slower mellow peaceful background gentle soothing".split()},
    **{w: "high" for w in "energetic energy upbeat fast faster workout gym running run party dance dancing "
                          "loud louder intense hype aggressive pump heavy".split()},
}
MOOD_WORDS: Dict[str, str] = {
    **{w: "happy" for w in "happy cheerful joyful uplifting sunny fun positive".split()},
    **{w: "sad" for w in "sad melancholic melancholy depressing heartbreak crying gloomy".split()},
    **{w: "romantic" for w in "romantic love date sexy".split()},
    **{w: "dark" for w in "dark sinister creepy".split()},
    **{w: "calm" for w in "calm peaceful serene".split()},
    **{w: "angry" for w in "angry aggressive rage".split()},
}
_ARTIST_CUES = {"like", "by", "from", "similar", "as", "love", "loves"}
_GENDERS = {"female": "f", "woman": "f", "girl": "f", "male": "m", "man": "m", "boy": "m"}


class RuleSummarizer(BaseSummarizer):
    def __init__(self, catalog: Catalog, min_tag_count: int = 2, max_ngram: int = 4):
        self.vocab = {
            t for t in catalog.tag_vocab(min_tag_count)
            if t not in GENERIC_WORDS and not all(w in STOPWORDS for w in t.split()) and len(t) >= 2
        }
        self.artists = catalog.artist_index
        self.max_ngram = max_ngram

    # ------------------------------------------------------------ разбор

    def _match(self, toks: List[str]) -> List[Tuple[int, int, str, str]]:
        """Непересекающиеся совпадения (start, end, kind, value), длинные первыми."""
        taken = [False] * len(toks)
        found = []
        for n in range(min(self.max_ngram, len(toks)), 0, -1):
            for i in range(len(toks) - n + 1):
                if any(taken[i:i + n]):
                    continue
                phrase = " ".join(toks[i:i + n])
                kind = None
                if phrase in self.artists and (n > 1 or (i > 0 and toks[i - 1] in _ARTIST_CUES)):
                    kind, value = "artist", self.artists[phrase]
                elif phrase in self.vocab:
                    kind, value = "tag", phrase
                if kind:
                    found.append((i, i + n, kind, value))
                    for j in range(i, i + n):
                        taken[j] = True
        return sorted(found)

    def _parse(self, text: str, state: dict) -> None:
        for clause in split_clauses(text):
            toks = tokenize(clause)
            neg_at = next((i for i, t in enumerate(toks) if t in NEGATIONS), None)
            for s, _, kind, value in self._match(toks):
                negated = neg_at is not None and s > neg_at
                inc, exc = ("include", "exclude") if kind == "tag" else ("seed_artists", "exclude_artists")
                if negated:
                    state[inc].pop(value, None)
                    state[exc][value] = True
                else:
                    state[exc].pop(value, None)
                    state[inc][value] = True
            for i, t in enumerate(toks):
                negated = neg_at is not None and i > neg_at
                if t in ENERGY_WORDS:
                    e = ENERGY_WORDS[t]
                    e = ({"low": "high", "high": "low"}[e]) if negated else e
                    if state["energy"] and state["energy"] != e:
                        # «а побыстрее»: снимаем теги, противоположные по энергии
                        for tag in [x for x in state["include"] if ENERGY_WORDS.get(x) not in (None, e)]:
                            state["include"].pop(tag)
                    state["energy"] = e
                if t in MOOD_WORDS and not negated:
                    state["mood"] = MOOD_WORDS[t]

    @staticmethod
    def _user_attrs(user_info: str) -> Dict[str, object]:
        attrs: Dict[str, object] = {}
        m = re.search(r"(\d{1,2})\s*(?:years?|y\.?o\.?|yo)\b", user_info, re.I) or \
            re.search(r"\bage[:\s]+(\d{1,2})\b", user_info, re.I)
        if m:
            attrs["age"] = int(m.group(1))
        for tok in tokenize(user_info):
            if tok in _GENDERS:
                attrs["gender"] = _GENDERS[tok]
                break
        m = re.search(r"\bfrom\s+([A-Z][\w-]*(?:\s+[A-Z][\w-]*)?)", user_info)
        if m:
            attrs["country"] = m.group(1)
        return attrs

    # ------------------------------------------------------------ API

    def summarize(self, request: Request, profile: Optional[UserProfile] = None) -> DialogSummary:
        state = {"include": {}, "exclude": {}, "seed_artists": {}, "exclude_artists": {},
                 "energy": None, "mood": None}
        for text in request.user_messages:
            self._parse(text, state)

        # предпочтения из user_info: отдельно, чтобы не путать с текущим запросом
        info_state = {"include": {}, "exclude": {}, "seed_artists": {}, "exclude_artists": {},
                      "energy": None, "mood": None}
        if request.user_info:
            self._parse(request.user_info, info_state)
        exclude = list(state["exclude"]) + [t for t in info_state["exclude"] if t not in state["include"]]

        include = list(state["include"])
        seeds = list(state["seed_artists"])
        last = request.user_messages[-1] if request.user_messages else ""
        query_words = [w for w in content_tokens(last) if w not in GENERIC_WORDS and w not in NEGATIONS]
        excluded_words = {w for t in exclude for w in t.split()}
        query_words = [w for w in query_words if w not in excluded_words]
        query = " ".join(dict.fromkeys(include + seeds + query_words))

        summary = DialogSummary(
            query=query,
            include_tags=include,
            exclude_tags=list(dict.fromkeys(exclude)),
            seed_artists=seeds,
            exclude_artists=list(state["exclude_artists"]),
            mood=state["mood"],
            energy=state["energy"],
            user_tags=list(info_state["include"]),
            user_attrs=self._user_attrs(request.user_info),
            source="rule",
        )
        summary.text = render_summary_text(summary)
        return summary


def render_summary_text(s: DialogSummary) -> str:
    parts = []
    if s.include_tags:
        parts.append("wants " + ", ".join(s.include_tags))
    if s.seed_artists:
        parts.append("similar to " + ", ".join(s.seed_artists))
    if s.mood:
        parts.append(f"mood: {s.mood}")
    if s.energy:
        parts.append(f"energy: {s.energy}")
    if s.exclude_tags or s.exclude_artists:
        parts.append("avoid " + ", ".join(s.exclude_tags + s.exclude_artists))
    if s.user_tags:
        parts.append("generally likes " + ", ".join(s.user_tags))
    return ("User " + "; ".join(parts) + ".") if parts else "User has no specific request."
