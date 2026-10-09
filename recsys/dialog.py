"""Шаг 1: сырой диалог + user_info (+ профиль истории) -> DialogSummary.

RuleSummarizer  заглушка по правилам: теги каталога, имена артистов, эпохи, страны и язык текста,
                отрицания ('no rap', 'but not hard rock') -> исключения, позже сказанное перекрывает раннее;
                «like X by Y but from other artists» -> образец X и исключение артиста Y (similar_to),
                «new artists I haven't heard» -> только новые треки новых артистов (novelty).
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
from recsys.text import (COUNTRIES_EN, GENERIC_WORDS, LANGUAGE_CUES, LANGUAGES_EN, NEGATIONS, STOPWORDS,
                         content_tokens, is_cyrillic, match_era, normalize_tag, parse_constraints, split_clauses, tokenize)


class BaseSummarizer(ABC):
    """Шаг 1: сырой диалог + user_info (+ профиль истории) -> DialogSummary."""

    @abstractmethod
    def summarize(self, request: Request, profile: Optional[UserProfile] = None) -> DialogSummary:
        ...


SUMMARY_SYSTEM = """You are the dialog-understanding module of a music recommender.
The user writes in English; the music catalog is described with English Last.fm tags.
Read the user profile, the listening-history tags and the dialog, and extract what music the user wants NOW.
Later messages override earlier ones. Negations ("no rap", "but not too fast") go to exclude lists,
but "no vocals" / "without words" means the user WANTS instrumental music.
If the user names a specific track or artist (possibly with typos), restore the correct spelling in
seed_artists / track_title. If the user quotes lyrics, copy them to lyrics as is.
"Like X by Y but from other artists" means: Y goes to exclude_artists, not to seed_artists.
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
  "query": "short English search query, 3-8 words",
  "new_artists": true if the user wants artists they have not heard before, else false
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
_ARTIST_CUES = {"like", "by", "from", "similar", "as", "love", "loves"}
_GENDERS = {"female": "f", "woman": "f", "girl": "f", "male": "m", "man": "m", "boy": "m"}
# similar_to: «tracks like focus by charli xcx but from other artists»
_REFERENCE_RE = re.compile(
    r"\b(?:similar to|like|reminds? me of|in the vein of|close to|along the lines of|than|as \w+ as)\s+"
    r"(?P<title>.+?)\s+by\s+"
    r"(?P<artist>.+?)(?=\s+(?:but|from|with|and|that|which|only|more|less|just|or)\b|[,.;:!?]|$)", re.I)
_PROFILE_GENRES_RE = re.compile(r"favou?rite genres:\s*([^.]+)", re.I)
# «no vocals» / «without words» — это просьба об инструментальной музыке, а не исключение
_INSTRUMENTAL_RE = re.compile(r"\b(?:no|without)\s+(?:vocals?|words|lyrics|singing|voice)\b", re.I)
_OTHER_ARTISTS_RE = re.compile(
    r"\b(?:other|different|another|new)\s+(?:artists?|bands?|groups?|singers?|musicians?|people|acts?|names)\b", re.I)
# novelty: «new artists please», «something i haven't heard», «nothing like my usual»
_NEW = r"\bnew(?!\s+(?:age|wave|york|order|school|romantic|orleans|zealand|jersey|edition|found))"
_NOVELTY_RE = re.compile(
    _NEW + r"\s+(?:to me\s+)?(?:\w+\s+){0,3}?(?:artists?|bands?|music|stuff|names|acts?|sounds?|tracks?|songs?|"
    r"genres?|discover\w*)\b|\b(?:unfamiliar|undiscovered)\b|\bnew to me\b|"
    r"\b(?:haven'?t|have not|never|not)\s+(?:yet\s+)?(?:heard|listened)\b|"
    r"\b(?:different from|away from|not like|unlike|nothing like|break from|outside(?: of)?|beyond|out of|leave)\s+"
    r"(?:my|what i)\b|\b(?:totally|completely|something)\s+(?:new|different)\b|\bcomfort zone\b|"
    r"\bchange of pace\b|\bsurprise me\b|\bpalate cleanser\b", re.I)


@dataclass
class _Match:
    start: int
    end: int
    kind: str          # tag | artist | country | lang | year
    value: Any


def _new_state() -> Dict[str, Any]:
    return {"include": {}, "exclude": {}, "seed_artists": {}, "exclude_artists": {},
            "countries": {}, "languages": {}, "years": {}, "energy": None, "mood": None}


class RuleSummarizer(BaseSummarizer):
    """Теги каталога, артисты, эпохи, страны и язык текста в английском запросе."""

    def __init__(self, catalog: Catalog, min_tag_count: int = 2, max_ngram: int = 4):
        self.vocab = {
            t for t in catalog.tag_vocab(min_tag_count)
            if t not in GENERIC_WORDS and t not in NEGATIONS and not all(w in STOPWORDS for w in t.split()) and len(t) >= 2
            and not is_cyrillic(t)
        }
        self.artists = catalog.artist_index
        self.max_ngram = max_ngram
        self.catalog = catalog
        self._titles: Optional[Dict[Any, str]] = None

    # ------------------------------------------------------------ разбор

    def _match(self, toks: List[str]) -> List[_Match]:
        """Непересекающиеся совпадения, длинные первыми: артист > тег каталога; затем эпохи, язык, страна."""
        taken = [False] * len(toks)
        found: List[_Match] = []
        for n in range(min(self.max_ngram, len(toks)), 0, -1):
            for i in range(len(toks) - n + 1):
                if any(taken[i:i + n]):
                    continue
                phrase = " ".join(toks[i:i + n])
                if phrase in self.artists and (n > 1 or (i > 0 and toks[i - 1] in _ARTIST_CUES)):
                    found.append(_Match(i, i + n, "artist", self.artists[phrase]))
                elif phrase in self.vocab:
                    found.append(_Match(i, i + n, "tag", phrase))
                else:
                    continue
                for j in range(i, i + n):
                    taken[j] = True
        for i, tok in enumerate(toks):
            if taken[i]:
                continue
            decade, year = match_era(tok)
            if decade:
                found.append(_Match(i, i + 1, "tag", decade))
                if year:
                    found.append(_Match(i, i + 1, "year", year))
            elif tok in LANGUAGES_EN and ((i > 0 and toks[i - 1] in ("in",) + tuple(LANGUAGE_CUES)) or
                                          (i + 1 < len(toks) and toks[i + 1] in LANGUAGE_CUES)):
                found.append(_Match(i, i + 1, "lang", LANGUAGES_EN[tok]))
            elif tok in COUNTRIES_EN:
                found.append(_Match(i, i + 1, "country", COUNTRIES_EN[tok]))
        for m in list(found):  # 'british' как тег каталога тоже задаёт страну
            if m.kind == "tag" and m.value in COUNTRIES_EN:
                found.append(_Match(m.start, m.end, "country", COUNTRIES_EN[m.value]))
        return sorted(found, key=lambda m: (m.start, m.kind))

    def _parse(self, text: str, state: Dict[str, Any]) -> None:
        for clause in split_clauses(_INSTRUMENTAL_RE.sub("instrumental", text)):
            toks = tokenize(clause)
            matches = self._match(toks)
            # отрицание, которое не входит в «положительную» фразу вроде 'без слов'
            neg_at = next((i for i, t in enumerate(toks) if t in NEGATIONS), None)
            for m in matches:
                negated = neg_at is not None and m.start > neg_at
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

    @staticmethod
    def _profile_tags(user_info: str, info_state: Dict[str, Any]) -> List[str]:
        """Интересы из профиля: раздел «Favourite genres: a, b, c.»; если его нет — теги каталога из текста
        без чисел и кодов («listening since 2012», «en 99%» — это не вкус)."""
        m = _PROFILE_GENRES_RE.search(user_info or "")
        if m:
            return list(dict.fromkeys(normalize_tag(g) for g in m.group(1).split(",") if g.strip()))
        return [t for t in info_state["include"] if len(t) > 2 and not any(ch.isdigit() for ch in t)]

    def _reference(self, text: str):
        """«like X by Y» -> (id трека X или None, артист Y как в каталоге или None, слова X и Y для query)."""
        m = _REFERENCE_RE.search(text)
        if not m:
            return None, None, set()
        artist_key, title_key = " ".join(tokenize(m["artist"])), " ".join(tokenize(m["title"]))
        artist = self.artists.get(artist_key)
        if artist is None:
            return None, None, set()
        if self._titles is None:  # артист -> [(название, id)]
            self._titles = {}
            df = self.catalog.df
            for tid, a, t in zip(df["track_id"], df["artist"], df["title"]):
                self._titles.setdefault(a, []).append((" ".join(tokenize(t)), tid))
        titles = self._titles.get(artist, [])
        track = next((tid for t, tid in titles if t == title_key), None) or \
            next((tid for t, tid in titles if title_key and t.startswith(title_key)), None)
        return track, artist, set(artist_key.split()) | set(title_key.split())

    # ------------------------------------------------------------ API

    def summarize(self, request: Request, profile: Optional[UserProfile] = None) -> DialogSummary:
        state = _new_state()
        constraints: Dict[str, Any] = {}
        for text in request.user_messages:
            self._parse(text, state)
            constraints.update(parse_constraints(text))  # позже сказанное перекрывает раннее

        # предпочтения из профиля пользователя: отдельно, чтобы не путать с текущим запросом
        info_state = _new_state()
        if request.user_info:
            self._parse(request.user_info, info_state)
        exclude = list(state["exclude"]) + [t for t in info_state["exclude"] if t not in state["include"]]

        include = list(state["include"])
        seeds = list(state["seed_artists"])
        exclude_artists = list(state["exclude_artists"])
        last = request.user_messages[-1] if request.user_messages else ""
        # similar_to: «like X by Y but from other artists» — X образец (похожие по звучанию), Y исключается
        ref_track, ref_artist, ref_words = self._reference(last)
        if ref_artist:  # артист образца и «артисты» из слов названия («like focus by ...» — не группа Focus)
            seeds = [a for a in seeds if a != ref_artist and not set(tokenize(a)) <= ref_words]
            exclude_artists.append(ref_artist)
        elif _OTHER_ARTISTS_RE.search(last):  # «like Y but other artists»: Y — образец, а не ответ
            exclude_artists += seeds
        # в поисковый запрос идут теги, артисты и латинские слова (каталог англоязычный)
        query_words = [w for w in content_tokens(_INSTRUMENTAL_RE.sub("instrumental", last))
                       if not is_cyrillic(w) and w not in GENERIC_WORDS and w not in NEGATIONS]
        excluded_words = {w for t in exclude for w in t.split()} | ref_words
        query_words = [w for w in query_words if w not in excluded_words]
        if ref_words:  # слова названия образца — не теги («like without you by oh wonder»)
            include = [t for t in include if not set(t.split()) <= ref_words]
        query = " ".join(dict.fromkeys(include + seeds + query_words))

        summary = DialogSummary(
            query=query,
            include_tags=include,
            exclude_tags=list(dict.fromkeys(exclude)),
            seed_artists=seeds,
            exclude_artists=list(dict.fromkeys(exclude_artists)),
            seed_track_ids=[ref_track] if ref_track else [],
            new_tracks=bool(ref_track or ref_artist or _NOVELTY_RE.search(last)),
            new_artists=bool(_NOVELTY_RE.search(last)),
            mood=state["mood"],
            energy=state["energy"],
            countries=list(state["countries"]),
            languages=list(state["languages"]),
            years=list(state["years"]),
            constraints=constraints,
            user_tags=self._profile_tags(request.user_info, info_state),
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
    if s.new_artists:
        parts.append("only artists new to the user")
    elif s.new_tracks:
        parts.append("only tracks new to the user")
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
        if data.get("new_artists") is True:
            out.new_artists = out.new_tracks = True
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
