"""Шаг 1: сырой диалог + user_info (+ профиль истории) -> DialogSummary.

RuleSummarizer  заглушка по правилам: теги каталога, имена артистов, эпохи, страны и язык текста,
                отрицания ('no rap', 'but not hard rock') -> исключения, позже сказанное перекрывает раннее;
                «like X by Y but from other artists» -> образец X и исключение артиста Y (similar_to),
                «new artists I haven't heard» -> только новые треки новых артистов (novelty).
                search_text — реплика без исключённого: её ищут текстовые кандгены (candgen.text_query: positive).

Отрицания (_negation): исключается только распознанное — тег или артист каталога после «no / not / without /
except / avoid / hate / don't want»; в запросах о тексте песни «no / not» — содержание, а не запрет:
  - «not / no / nothing» после местоимения или вспомогательного глагола — грамматика («you are not alone»);
  - маркер текста песни раньше в клаузе («lyrics», «about», «goes», «says») — дальше содержание;
  - название трека важнее: «eyes without a face», «no buses» + arctic monkeys, «hear me no(w)» + hollywood undead;
  - «not too / so / very X» — мягкое ограничение: X не просим и не исключаем, только убираем из search_text.
Опечатка в длинном слове отрицания («noting», «witout») — то же отрицание (_cue).
Слово — и артист, и тег («nirvana», «queen», «rock»): артист, если большинство треков с этим тегом — его.
Тег-прозвище («rhcp», «nin», «bowie»: ≥ 80% треков у одного артиста, без жанровых слов) — тоже артист.
Опечатка в имени после отрицания («no nirnava») — summarizer.fuzzy_artists (по умолчанию выкл.), _typo_artist.
Прозвище, которое знают только модели («no rhcp» в башнях) — summarizer.resolve_entities (выкл.), recsys/entity.py.
LLMSummarizer   LLM возвращает JSON; любая ошибка -> результат RuleSummarizer.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

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


RULES_VERSION = "4"   # менять вместе с правилами: от разбора зависят кэши выдачи (text_query: positive) и признаков


def rules_key(cfg: Dict[str, Any]) -> str:
    """Версия разбора для ключей кэшей: RULES_VERSION и включённые необязательные правила."""
    scfg = cfg.get("summarizer", {})
    return (RULES_VERSION + ("+fuzzy" if scfg.get("fuzzy_artists") else "")
            + (f"+resolve{sorted((scfg.get('resolve') or {}).items())}" if scfg.get("resolve_entities") else ""))


NEG_CUES = frozenset("no not without except excluding exclude avoid nothing none hate dislike neither nor minus".split())
_DONT = frozenset("don't dont".split())
_WANT = frozenset("want like play include give need".split())
_SOFT = frozenset("too so very overly that super".split())                   # «not too sad» — мягко
_GRAMMAR = frozenset("""i i'm im you you're youre we we're they they're he he's she she's it it's is are am was
were be been being do does did will would can could should might may must i'll you'll we'll they'll won't can't
cannot there there's that's who which""".split())                               # «you are not alone» — предложение
_SENTENCE_CUES = frozenset("no not nothing none".split())
_LONG_CUES = ("nothing", "without", "except", "exclude", "excluding", "dislike")   # опечатка в 1 букву — то же слово                       # только им мешает грамматика
_LYRIC_MARKERS = frozenset("lyrics lyric about goes sings sang says said line lines words verse chorus phrase".split())


def _edit_distance(a: str, b: str, limit: int) -> int:
    """Левенштейн с ранним выходом: > limit -> limit + 1."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]


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

    def __init__(self, catalog: Catalog, min_tag_count: int = 2, max_ngram: int = 4, fuzzy_artists: bool = False,
                 resolver=None):
        self.vocab = {
            t for t in catalog.tag_vocab(min_tag_count)
            if t not in GENERIC_WORDS and t not in NEGATIONS and not all(w in STOPWORDS for w in t.split()) and len(t) >= 2
            and not is_cyrillic(t) and t.split()[0] not in NEG_CUES and t.split()[0] not in _DONT   # тег «not rock»
        }
        self.artists = catalog.artist_index
        self.max_ngram = max_ngram
        self.catalog = catalog
        self._titles: Optional[Dict[Any, str]] = None
        self._neg_titles: Optional[Dict[tuple, set]] = None   # названия с «no / not / without»: слова -> артисты
        self._tag_owner: Dict[str, Optional[str]] = {}
        self._aliases: Dict[str, Optional[str]] = {}
        self._genre_words: Optional[set] = None
        self.fuzzy_artists = fuzzy_artists
        self.resolver = resolver          # recsys.entity.EntityResolver: фраза после отрицания -> артист по поиску
        self._by_len: Optional[Dict[int, List[str]]] = None   # длина имени -> нормализованные имена артистов
        self._history_artists: set = set()
        self._common: Optional[set] = None                    # артисты истории текущего запроса (для опечаток)

    # ------------------------------------------------------------ разбор

    def _match(self, toks: List[str]) -> List[_Match]:
        """Непересекающиеся совпадения, длинные первыми: артист или тег каталога; затем эпохи, язык, страна.
        Одно слово — артист только после «like / by / from …» или отрицания («no nirvana»); и артист, и тег —
        артист, если тег почти только его (_artist_owns)."""
        taken = [False] * len(toks)
        found: List[_Match] = []
        for n in range(min(self.max_ngram, len(toks)), 0, -1):
            for i in range(len(toks) - n + 1):
                if any(taken[i:i + n]):
                    continue
                phrase = " ".join(toks[i:i + n])
                artist = phrase in self.artists and (
                    n > 1 or (i > 0 and (toks[i - 1] in _ARTIST_CUES or toks[i - 1] in NEG_CUES)))
                if artist and phrase in self.vocab:
                    artist = self._artist_owns(phrase, self.artists[phrase])
                alias = None if artist or phrase not in self.vocab else self._tag_alias(phrase)
                if artist:
                    found.append(_Match(i, i + n, "artist", self.artists[phrase]))
                elif alias:                  # тег-прозвище: «rhcp», «nin», «bowie» -> артист
                    found.append(_Match(i, i + n, "artist", alias))
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

    def _parse(self, text: str, state: Dict[str, Any]) -> List[str]:
        """Разбор реплики в state; -> её слова без исключённого и мягко отрицаемого (search_text)."""
        kept: List[str] = []
        text = _INSTRUMENTAL_RE.sub("instrumental", text)
        protected = self._title_cues(tokenize(text))
        for clause in split_clauses(text):
            toks = [self._cue(t) for t in tokenize(clause)]       # «noting like nirvana» -> nothing
            matches = self._match(toks)
            neg_at, soft = None, set()
            for i in range(len(toks)):
                kind = None if toks[i] in protected else self._negation(toks, i)
                if kind == "soft":
                    soft.add(i)
                elif kind == "hard" and neg_at is None:
                    neg_at = i
            if self.fuzzy_artists and neg_at is not None:
                typo = self._typo_artist(toks, neg_at, matches)
                if typo is not None:
                    matches.append(typo)
            if self.resolver is not None and neg_at is not None:
                found = self._resolved_artist(toks, neg_at, matches)
                if found is not None:
                    matches.append(found)
            drop = set()
            for m in matches:
                negated = neg_at is not None and m.start > neg_at
                softened = any(m.start == j + 2 for j in soft)   # «not too sad», «not so heavy»
                if softened and not negated:
                    drop.update(range(m.start, m.end))
                    continue
                if m.kind in ("tag", "artist"):
                    inc, exc = ("include", "exclude") if m.kind == "tag" else ("seed_artists", "exclude_artists")
                    if negated:
                        state[inc].pop(m.value, None)
                        state[exc][m.value] = True
                        drop.update(range(m.start, m.end))
                    else:
                        state[exc].pop(m.value, None)
                        state[inc][m.value] = True
                    if m.kind == "tag":
                        self._energy_mood(m.value, negated, state)
                elif not negated:
                    state[{"country": "countries", "lang": "languages", "year": "years"}[m.kind]][m.value] = True
                else:
                    drop.update(range(m.start, m.end))
            if drop and neg_at is not None:
                drop.add(neg_at)
            for j in soft:              # «not too fast»: не ищем и то, что смягчено (даже не тег)
                drop.update(range(j, min(j + 3, len(toks))))
            matched = {j for m in matches for j in range(m.start, m.end)}
            for i, t in enumerate(toks):  # английские слова настроения вне тегов каталога
                if i not in matched:
                    self._energy_mood(t, neg_at is not None and i > neg_at, state)
            kept += [t for i, t in enumerate(toks) if i not in drop]
        return kept

    def _cue(self, tok: str) -> str:
        """Опечатка в длинном слове отрицания («noting», «witout», «nothng») -> само слово; частые слова названий
        и тегов не трогаем (это обычные слова, а не опечатки)."""
        if tok in NEG_CUES or len(tok) < 5 or tok in STOPWORDS or tok in self._common_words():
            return tok
        return next((c for c in _LONG_CUES if abs(len(c) - len(tok)) <= 1 and _edit_distance(tok, c, 1) <= 1), tok)

    def _common_words(self) -> set:
        """Слова названий треков и тегов, встречающиеся ≥ 5 раз: обычные слова, а не опечатки."""
        if self._common is None:
            words: Dict[str, int] = {}
            for text in list(self.catalog.df["title"]) + list(self.vocab):
                for w in set(tokenize(text)):
                    words[w] = words.get(w, 0) + 1
            self._common = {w for w, c in words.items() if c >= 5}
        return self._common

    @staticmethod
    def _negation(toks: List[str], i: int) -> Optional[str]:
        """toks[i] — отрицание-исключение (hard), мягкое «not too» (soft) или нет (None)."""
        t, prev = toks[i], toks[i - 1] if i else ""
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if t in _DONT or (t == "not" and prev == "do"):         # «i don't want rap», «do not play metal»
            return "hard" if nxt in _WANT else None
        if t not in NEG_CUES:
            return None
        if t == "not" and nxt in _SOFT:
            return "soft"
        if t in _SENTENCE_CUES and prev in _GRAMMAR:            # «you are not alone», «i might not be»
            return None
        if any(w in _LYRIC_MARKERS for w in toks[:i]):          # «lyrics about not caring», «song about no one»
            return None
        return "hard"

    def _title_cues(self, toks: List[str]) -> set:
        """Слова отрицания, которые на самом деле часть названия трека из запроса:
          - название с этим словом стоит в запросе целиком и либо длинное (3+ слова, 2+ значимых: «eyes without
            a face»), либо в запросе есть его артист («no buses» + arctic monkeys);
          - «<начало названия> no <артист>»: слова перед отрицанием — начало названия трека этого артиста
            («hear me no hollywood undead» — Hear Me Now)."""
        if self._neg_titles is None:
            self._neg_titles = {}
            for t, a in zip(self.catalog.df["title"], self.catalog.df["artist"]):
                words = tuple(tokenize(t))
                if 2 <= len(words) <= 8 and any(w in NEG_CUES or w in _DONT for w in words):
                    self._neg_titles.setdefault(words, set()).add(a)
        out = set()
        text = f" {' '.join(toks)} "
        mentioned = lambda a: f" {' '.join(tokenize(a))} " in text
        for n in range(min(8, len(toks)), 1, -1):
            for i in range(len(toks) - n + 1):
                words = tuple(toks[i:i + n])
                artists = self._neg_titles.get(words)
                if not artists:
                    continue
                content = [w for w in words if w not in STOPWORDS and w not in NEG_CUES and w not in _DONT]
                if (n >= 3 and len(content) >= 2) or any(mentioned(a) for a in artists):
                    out.update(w for w in words if w in NEG_CUES or w in _DONT)
        for i, t in enumerate(toks):
            if t not in NEG_CUES or t in out or i == 0:
                continue
            artist = next((self.artists[p] for n in range(min(self.max_ngram, len(toks) - i - 1), 0, -1)
                           if (p := " ".join(toks[i + 1:i + 1 + n])) in self.artists), None)
            if artist is None:
                continue
            titles = [title for title, _ in self._artist_titles().get(artist, [])]
            for k in range(1, min(4, i) + 1):        # 1..4 слова перед отрицанием — начало названия
                before = toks[i - k:i]
                phrase = " ".join(before)
                if (k >= 2 or before[0] not in STOPWORDS) and any(title.startswith(phrase) for title in titles):
                    out.add(t)
                    break
        return out

    def _resolved_artist(self, toks: List[str], neg_at: int, matches: List[_Match]) -> Optional[_Match]:
        """«no rhcp» -> Red Hot Chili Peppers (summarizer.resolve_entities): после отрицания ничего не распознано,
        значимые слова подряд — ищем их отдельно (recsys/entity.py) и берём артиста, если источники согласны."""
        if any(m.start > neg_at for m in matches):
            return None
        rest = [i for i in range(neg_at + 1, len(toks)) if toks[i] not in STOPWORDS and toks[i] not in NEG_CUES]
        if not rest or rest != list(range(rest[0], rest[-1] + 1)) or len(rest) > self.max_ngram:
            return None
        artist = self.resolver.artist(" ".join(toks[rest[0]:rest[-1] + 1]))
        return None if artist is None else _Match(rest[0], rest[-1] + 1, "artist", artist)

    def _typo_artist(self, toks: List[str], neg_at: int, matches: List[_Match]) -> Optional[_Match]:
        """«rock, but no nirnava» -> Nirvana (summarizer.fuzzy_artists). Только когда после отрицания в клаузе
        ничего не распознано, а нераспознанные значимые слова — подряд, не частые слова названий и тегов («matter»,
        «pity»), не общие (song, band) и целиком похожи на имя артиста: расстояние Левенштейна ≤ 1 для 4–6 букв,
        ≤ 2 — для длинных; 4 буквы — только артисты истории. Ближайшее; при равных — артист из истории
        пользователя, затем у кого больше треков в каталоге."""
        if any(m.start > neg_at for m in matches):
            return None
        rest = [i for i in range(neg_at + 1, len(toks))
                if toks[i] not in STOPWORDS and toks[i] not in NEG_CUES and toks[i] not in GENERIC_WORDS]
        if not rest or rest != list(range(rest[0], rest[-1] + 1)) or len(rest) > self.max_ngram:
            return None
        if self._by_len is None:
            self._by_len = {}
            for key in self.artists:
                self._by_len.setdefault(len(key), []).append(key)
            self._n_tracks = self.catalog.df["artist"].value_counts().to_dict()
        if any(toks[i] in self._common_words() for i in rest):    # «no matter», «no pity», «nothing back»
            return None
        phrase = " ".join(toks[rest[0]:rest[-1] + 1])
        if len(phrase) < 4:
            return None
        limit = 1 if len(phrase) <= 6 else 2
        best = None
        for n in range(len(phrase) - limit, len(phrase) + limit + 1):
            for key in self._by_len.get(n, ()):
                d = _edit_distance(phrase, key, limit)
                if d <= limit:
                    name = self.artists[key]
                    if len(phrase) < 5 and name not in self._history_artists:   # «joel» -> Joe: коротко — только своё
                        continue
                    rank = (d, name not in self._history_artists, -self._n_tracks.get(name, 0))
                    if best is None or rank < best[0]:
                        best = (rank, name)
        if best is None:
            return None
        return _Match(rest[0], rest[-1] + 1, "artist", best[1])

    def _artist_titles(self) -> Dict[str, List]:
        if self._titles is None:  # артист -> [(название, id)]
            self._titles = {}
            df = self.catalog.df
            for tid, a, t in zip(df["track_id"], df["artist"], df["title"]):
                self._titles.setdefault(a, []).append((" ".join(tokenize(t)), tid))
        return self._titles

    def _tag_alias(self, tag: str) -> Optional[str]:
        """Тег Last.fm — прозвище артиста: ≥ 5 треков, ≥ 80% из них — одного артиста, и в теге нет жанровых слов
        каталога («rhcp», «nin», «bowie», «freddie mercury» -> Queen; не «swamp rock», не «depressive rock»)."""
        if self._genre_words is None:
            self._genre_words = {w for g in self.catalog.df["genres"] for x in g for w in x.split()}
        if tag not in self._aliases:
            pos = self.catalog.tag_index.get(tag, [])
            owner = None
            if len(pos) >= 5 and not set(tag.split()) & self._genre_words:
                names, counts = np.unique(self.catalog.df["artist"].to_numpy()[pos], return_counts=True)
                if counts.max() >= 0.8 * len(pos):
                    owner = names[counts.argmax()]
            self._aliases[tag] = owner
        return self._aliases[tag]

    def _artist_owns(self, tag: str, artist: str) -> bool:
        """Тег — по сути имя артиста: больше половины треков с ним — этого артиста («nirvana», «queen»)."""
        if tag not in self._tag_owner:
            pos = self.catalog.tag_index.get(tag, [])
            artists = self.catalog.df["artist"].to_numpy()[pos] if len(pos) else []
            owner = None
            if len(artists):
                names, counts = np.unique(artists, return_counts=True)
                if counts.max() * 2 > len(artists):
                    owner = names[counts.argmax()]
            self._tag_owner[tag] = owner
        return self._tag_owner[tag] == artist

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
        titles = self._artist_titles().get(artist, [])
        track = next((tid for t, tid in titles if t == title_key), None) or \
            next((tid for t, tid in titles if title_key and t.startswith(title_key)), None)
        return track, artist, set(artist_key.split()) | set(title_key.split())

    # ------------------------------------------------------------ API

    def summarize(self, request: Request, profile: Optional[UserProfile] = None) -> DialogSummary:
        state = _new_state()
        constraints: Dict[str, Any] = {}
        kept: List[str] = []
        self._history_artists = set(profile.artist_weights) if profile is not None else set()
        for text in request.user_messages:
            kept = self._parse(text, state)               # search_text — по последней реплике
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
        # поиск по тексту — без исключённого: артист и название образца similar_to — тоже исключённое
        excluded_names = {w for a in exclude_artists for w in tokenize(a)} | ref_words
        search_text = " ".join(t for t in kept if t not in excluded_names)

        summary = DialogSummary(
            query=query,
            search_text=search_text,
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


def build_summarizer(cfg: Dict[str, Any], catalog: Catalog, llm: Optional[BaseLLM] = None,
                     resolver=None) -> BaseSummarizer:
    scfg = cfg.get("summarizer", {})
    rule = RuleSummarizer(catalog, min_tag_count=scfg.get("min_tag_count", 2),
                          fuzzy_artists=bool(scfg.get("fuzzy_artists", False)), resolver=resolver)
    kind = scfg.get("type", "rule")
    if kind == "rule":
        return rule
    if kind == "llm":
        if llm is None:
            raise ValueError("summarizer.type=llm, но llm не передана")
        return LLMSummarizer(llm, fallback=rule)
    raise ValueError(f"Неизвестный summarizer.type: {kind}")
