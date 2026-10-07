"""Шаг 1: сырой диалог + user_info (+ профиль истории) -> DialogSummary.

RuleSummarizer  заглушка по правилам: русский словарь (recsys/ru.py), теги каталога и имена артистов,
                отрицания ('без рэпа', 'но не танцевальная', 'no rap') -> исключения,
                позже сказанное перекрывает раннее.
LLMSummarizer   LLM возвращает JSON; любая ошибка -> результат RuleSummarizer.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from recsys.data.catalog import Catalog
from recsys.llm import BaseLLM, extract_json
from recsys.schemas import DialogSummary, Request, UserProfile
from recsys.ru import COUNTRIES_EN, NEGATIONS_RU, lookup_stem, match_era, match_phrase
from recsys.text import (GENERIC_WORDS, NEGATIONS, STOPWORDS, content_tokens, is_cyrillic, normalize_tag,
                         split_clauses, tokenize)


class BaseSummarizer(ABC):
    """Шаг 1: сырой диалог + user_info (+ профиль истории) -> DialogSummary."""

    @abstractmethod
    def summarize(self, request: Request, profile: Optional[UserProfile] = None) -> DialogSummary:
        ...


SUMMARY_SYSTEM = """You are the dialog-understanding module of a music recommender.
The user writes in Russian; the music catalog is described with English Last.fm tags.
Read the user profile, the listening-history tags and the dialog, and extract what music the user wants NOW.
Later messages override earlier ones. Negations ("без рэпа", "но не танцевальная") go to exclude lists,
but "без слов" / "без вокала" means the user WANTS instrumental music.
If the user names a specific track or artist (possibly transliterated or with typos), restore the original
Latin spelling in seed_artists / track_title. If the user quotes lyrics, copy them to lyrics as is.
Answer with ONE JSON object in English and nothing else:
{
  "summary": "one or two sentences in English",
  "include_tags": ["short lowercase Last.fm-style tags, e.g. indie rock, melancholic, instrumental, 80s"],
  "exclude_tags": [],
  "seed_artists": ["artists the user names or wants something similar to"],
  "exclude_artists": [],
  "track_title": "title if the user looks for a specific track, else null",
  "lyrics": "quoted lyrics if any, else null",
  "mood": "happy | sad | calm | romantic | dark | angry | null",
  "energy": "low | medium | high | null",
  "countries": ["ISO country codes of the artist, e.g. US, GB"],
  "languages": ["ISO language codes of the lyrics, e.g. en, ru"],
  "years": [release years if a specific year is named],
  "query": "short English search query, 3-8 words"
}"""

SUMMARY_USER = """User profile: {user_info}
Listening history (top tags): {history_tags}
Listening history (top artists): {history_artists}

Dialog:
{dialog}"""


ENERGY_WORDS: Dict[str, str] = {
    **{w: "low" for w in "calm chill chilled relax relaxing relaxed quiet soft sleep sleeping study studying "
                         "focus slow slower mellow peaceful background gentle soothing meditation".split()},
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
_ARTIST_CUES = {"like", "by", "from", "similar", "as", "love", "loves", "как", "типа", "похож", "похожее"}
_GENDERS = {"female": "f", "woman": "f", "girl": "f", "male": "m", "man": "m", "boy": "m"}
NEGATIONS_ALL = NEGATIONS | NEGATIONS_RU


@dataclass
class _Match:
    start: int
    end: int
    kind: str          # tag | artist | country | lang | year
    value: Any
    positive: bool = False  # полярность задана самой фразой ('без слов')


def _new_state() -> Dict[str, Any]:
    return {"include": {}, "exclude": {}, "seed_artists": {}, "exclude_artists": {},
            "countries": {}, "languages": {}, "years": {}, "energy": None, "mood": None}


class RuleSummarizer(BaseSummarizer):
    """Английские теги каталога и артисты (латиница) + русский словарь recsys/ru.py."""

    def __init__(self, catalog: Catalog, min_tag_count: int = 2, max_ngram: int = 4):
        self.vocab = {
            t for t in catalog.tag_vocab(min_tag_count)
            if t not in GENERIC_WORDS and not all(w in STOPWORDS for w in t.split()) and len(t) >= 2
            and not is_cyrillic(t)
        }
        self.artists = catalog.artist_index
        self.max_ngram = max_ngram

    # ------------------------------------------------------------ разбор

    def _match(self, toks: List[str]) -> List[_Match]:
        """Непересекающиеся совпадения, длинные первыми: артист > тег каталога > русская фраза > эпоха > основа."""
        taken = [False] * len(toks)
        found: List[_Match] = []

        def take(i: int, n: int) -> None:
            for j in range(i, i + n):
                taken[j] = True

        for n in range(min(self.max_ngram, len(toks)), 0, -1):
            for i in range(len(toks) - n + 1):
                if any(taken[i:i + n]):
                    continue
                phrase = " ".join(toks[i:i + n])
                if phrase in self.artists and (n > 1 or (i > 0 and toks[i - 1] in _ARTIST_CUES)):
                    found.append(_Match(i, i + n, "artist", self.artists[phrase]))
                    take(i, n)
                elif phrase in self.vocab:
                    found.append(_Match(i, i + n, "tag", phrase))
                    take(i, n)
                elif n == 1 or n == 2 or n == 3:
                    hit = match_phrase(toks, i)
                    if hit and hit[0] == n:
                        found.extend(self._entry_matches(i, i + n, hit[1]))
                        take(i, n)
        for i in range(len(toks)):  # эпохи и одиночные русские основы
            if taken[i]:
                continue
            era = match_era(toks, i)
            if era and not any(taken[i:i + era[0]]):
                found.append(_Match(i, i + era[0], "tag", era[1]))
                if era[2]:
                    found.append(_Match(i, i + era[0], "year", era[2]))
                take(i, era[0])
                continue
            entry = lookup_stem(toks[i]) if is_cyrillic(toks[i]) else None
            if entry:
                found.extend(self._entry_matches(i, i + 1, entry))
                take(i, 1)
            elif toks[i] in COUNTRIES_EN:
                found.append(_Match(i, i + 1, "country", COUNTRIES_EN[toks[i]]))
        for m in list(found):  # 'british' как тег каталога тоже задаёт страну
            if m.kind == "tag" and m.value in COUNTRIES_EN:
                found.append(_Match(m.start, m.end, "country", COUNTRIES_EN[m.value], m.positive))
        return sorted(found, key=lambda m: (m.start, m.kind))

    @staticmethod
    def _entry_matches(s: int, e: int, entry: Dict[str, Any]) -> List[_Match]:
        pos = bool(entry.get("positive"))
        out = [_Match(s, e, "tag", t, pos) for t in entry.get("tags", [])]
        if entry.get("country"):
            out.append(_Match(s, e, "country", entry["country"], pos))
        if entry.get("lang"):
            out.append(_Match(s, e, "lang", entry["lang"], pos))
        return out

    def _parse(self, text: str, state: Dict[str, Any]) -> None:
        for clause in split_clauses(text):
            toks = tokenize(clause)
            matches = self._match(toks)
            # отрицание, которое не входит в «положительную» фразу вроде 'без слов'
            covered = {j for m in matches if m.positive for j in range(m.start, m.end)}
            neg_at = next((i for i, t in enumerate(toks) if t in NEGATIONS_ALL and i not in covered), None)
            for m in matches:
                negated = neg_at is not None and m.start > neg_at and not m.positive
                if m.kind in ("tag", "artist"):
                    inc, exc = ("include", "exclude") if m.kind == "tag" else ("seed_artists", "exclude_artists")
                    if negated:
                        state[inc].pop(m.value, None)
                        state[exc][m.value] = True
                    else:
                        state[exc].pop(m.value, None)
                        state[inc][m.value] = True
                    if m.kind == "tag":
                        self._energy_mood(m.value, negated, state)
                elif not negated:
                    state[{"country": "countries", "lang": "languages", "year": "years"}[m.kind]][m.value] = True
            matched = {j for m in matches for j in range(m.start, m.end)}
            for i, t in enumerate(toks):  # английские слова настроения вне тегов каталога
                if i not in matched:
                    self._energy_mood(t, neg_at is not None and i > neg_at, state)

    @staticmethod
    def _energy_mood(word: str, negated: bool, state: Dict[str, Any]) -> None:
        # «не танцевальная» не значит «спокойная»: отрицание энергию не меняет
        if word in ENERGY_WORDS and not negated:
            e = ENERGY_WORDS[word]
            if state["energy"] and state["energy"] != e:
                # «а побыстрее»: снимаем теги, противоположные по энергии
                for tag in [x for x in state["include"] if ENERGY_WORDS.get(x) not in (None, e)]:
                    state["include"].pop(tag)
            state["energy"] = e
        if word in MOOD_WORDS and not negated:
            state["mood"] = MOOD_WORDS[word]

    @staticmethod
    def _user_attrs(user_info: str) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {}
        m = re.search(r"(\d{1,2})\s*(?:years?|y\.?o\.?|yo|лет|год)\b", user_info, re.I) or \
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
        state = _new_state()
        for text in request.user_messages:
            self._parse(text, state)

        # предпочтения из профиля пользователя: отдельно, чтобы не путать с текущим запросом
        info_state = _new_state()
        if request.user_info:
            self._parse(request.user_info, info_state)
        exclude = list(state["exclude"]) + [t for t in info_state["exclude"] if t not in state["include"]]

        include = list(state["include"])
        seeds = list(state["seed_artists"])
        last = request.user_messages[-1] if request.user_messages else ""
        # в поисковый запрос идут теги, артисты и латинские слова (каталог англоязычный)
        query_words = [w for w in content_tokens(last)
                       if not is_cyrillic(w) and w not in GENERIC_WORDS and w not in NEGATIONS_ALL]
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
            countries=list(state["countries"]),
            languages=list(state["languages"]),
            years=list(state["years"]),
            user_tags=list(info_state["include"]),
            user_attrs={**self._user_attrs(request.user_info), **request.user_attrs},
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
    if s.countries or s.languages or s.years:
        parts.append("origin: " + ", ".join(s.countries + [f"lang={x}" for x in s.languages]
                                              + [str(y) for y in s.years]))
    if s.exclude_tags or s.exclude_artists:
        parts.append("avoid " + ", ".join(s.exclude_tags + s.exclude_artists))
    if s.user_tags:
        parts.append("generally likes " + ", ".join(s.user_tags))
    return ("User " + "; ".join(parts) + ".") if parts else "User has no specific request."


_LIST_FIELDS = ["include_tags", "exclude_tags", "seed_artists", "exclude_artists", "countries", "languages"]


class LLMSummarizer(BaseSummarizer):
    def __init__(self, llm: BaseLLM, fallback: RuleSummarizer, max_dialog_messages: int = 12):
        self.llm = llm
        self.fallback = fallback
        self.max_dialog_messages = max_dialog_messages
        self.last_raw: str = ""  # для отладки в ноутбуке

    def build_messages(self, request: Request, profile: Optional[UserProfile]) -> List[Dict[str, str]]:
        dialog = "\n".join(f"{m.role}: {m.text}" for m in request.dialog[-self.max_dialog_messages:])
        user = SUMMARY_USER.format(
            user_info=request.user_info or "unknown",
            history_tags=", ".join(profile.top_tags(10)) if profile else "unknown",
            history_artists=", ".join(profile.top_artists(5)) if profile else "unknown",
            dialog=dialog or "(empty)",
        )
        return [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": user}]

    def summarize(self, request: Request, profile: Optional[UserProfile] = None) -> DialogSummary:
        base = self.fallback.summarize(request, profile)
        try:
            self.last_raw = self.llm.generate(self.build_messages(request, profile))
        except Exception as e:  # модель не загрузилась, OOM и т.п.
            self.last_raw = f"<error: {e}>"
            return base
        data = extract_json(self.last_raw)
        if not data:
            return base
        return self._merge(base, data)

    @staticmethod
    def _merge(base: DialogSummary, data: Dict[str, Any]) -> DialogSummary:
        """Поля LLM поверх правил; пустые поля LLM не затирают найденное правилами."""
        out = DialogSummary(**{**base.__dict__})
        for f in _LIST_FIELDS:
            vals = data.get(f)
            if isinstance(vals, list) and vals:
                vals = [str(v) for v in vals if str(v).strip()]
                if f.endswith("tags"):
                    vals = [normalize_tag(v) for v in vals]
                elif f == "countries":
                    vals = [v.upper() for v in vals]
                elif f == "languages":
                    vals = [v.lower() for v in vals]
                setattr(out, f, list(dict.fromkeys(vals)))
        for f in ["mood", "energy"]:
            v = data.get(f)
            if isinstance(v, str) and v.strip() and v.strip().lower() != "null":
                setattr(out, f, v.strip().lower())
        if isinstance(data.get("years"), list):
            out.years = [int(y) for y in data["years"] if str(y).isdigit()] or out.years
        extra = [str(data[k]).strip() for k in ("track_title", "lyrics")
                 if isinstance(data.get(k), str) and data[k].strip() and data[k].strip().lower() != "null"]
        if isinstance(data.get("query"), str) and data["query"].strip():
            out.query = data["query"].strip()
        if extra:  # название и строчка текста тоже ищутся через query
            out.query = " ".join([out.query] + extra).strip()
        out.text = str(data.get("summary") or "").strip() or render_summary_text(out)
        out.source = "llm+rule"
        return out


def build_summarizer(cfg: Dict[str, Any], catalog: Catalog, llm: Optional[BaseLLM] = None) -> BaseSummarizer:
    scfg = cfg.get("summarizer", {})
    rule = RuleSummarizer(catalog, min_tag_count=scfg.get("min_tag_count", 2))
    kind = scfg.get("type", "rule")
    if kind == "rule":
        return rule
    if kind == "llm":
        if llm is None:
            raise ValueError("summarizer.type=llm, но llm не передана")
        return LLMSummarizer(llm, fallback=rule)
    raise ValueError(f"Неизвестный summarizer.type: {kind}")
