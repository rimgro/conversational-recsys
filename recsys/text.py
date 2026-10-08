"""Текстовые утилиты: нормализация тегов, токенизация, сопоставление тегов, эпохи / страны / языки в запросе.

Теги Last.fm, запросы датасета и саммари — на английском. Названия и имена бывают на языке оригинала
(в том числе кириллицей), поэтому токенизатор оставляет и кириллицу.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

# Варианты написания, которые приводим к одному виду (по целой строке и по словам).
TAG_SYNONYMS = {
    "hiphop": "hip hop",
    "r&b": "rnb",
    "r and b": "rnb",
    "r n b": "rnb",
    "lofi": "lo fi",
    "edm": "edm",
    "drum and bass": "drum n bass",
    "dnb": "drum n bass",
    "d&b": "drum n bass",
}
_WORD_SYNONYMS = {"hiphop": "hip hop", "lofi": "lo fi", "dnb": "drum n bass"}

STOPWORDS = frozenset("""
a an the and or but if then so of to in on at by for with from as is are was were be been being
i me my we our you your he she it its they them their this that these those there here
am do does did doing have has had having can could would should will shall may might must
just also very really quite too more most some any all each every other such only own same
than up down out over under again further once about into through during before after above below
what which who whom when where why how please want wanna need looking look find give play
recommend recommendation recommendations suggest something anything thing things kind sort maybe
let lets let's me i'm im i'd id i've ive it's its yeah yes ok okay hi hello hey thanks thank
""".split())

# Слишком общие слова, которые есть среди тегов Last.fm, но в речи почти не несут смысла.
GENERIC_WORDS = frozenset("""
music song songs track tracks good best great love like favorite favourite favorites awesome cool
nice new old amazing beautiful listen listening tune tunes stuff artist artists band bands
""".split())

NEGATIONS = frozenset("""
no not without except avoid hate dislike dislikes don't dont never nothing exclude excluding
none neither nor
""".split())

_CLAUSE_SPLIT_RE = re.compile(r"[.,;!?\n]|\b(?:but|however|though|although)\b", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9а-я]+(?:'[a-z]+)?")
_CYRILLIC_RE = re.compile(r"[а-я]")


def normalize_tag(tag: str) -> str:
    """'Hip-Hop' -> 'hip hop', 'Lo-Fi' -> 'lo fi', 'R&B' -> 'rnb'."""
    t = str(tag).lower().replace("ё", "е").strip()
    if t in TAG_SYNONYMS:
        return TAG_SYNONYMS[t]
    t = re.sub(r"[-_/]+", " ", t)
    t = re.sub(r"[^a-z0-9а-я&' ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if t in TAG_SYNONYMS:
        return TAG_SYNONYMS[t]
    words = [_WORD_SYNONYMS.get(w, w) for w in t.split()]
    return " ".join(words)


def tokenize(text: str) -> List[str]:
    """Нижний регистр, дефисы как пробелы, синонимы слов. Стоп-слова НЕ удаляются."""
    t = str(text).lower().replace("ё", "е").replace("&", " and ")
    t = re.sub(r"[-_/]+", " ", t)
    tokens: List[str] = []
    for tok in _TOKEN_RE.findall(t):
        tokens.extend(_WORD_SYNONYMS.get(tok, tok).split())
    return tokens


def content_tokens(text: str) -> List[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def is_cyrillic(token: str) -> bool:
    return bool(_CYRILLIC_RE.search(token))


def split_clauses(text: str) -> List[str]:
    return [c.strip() for c in _CLAUSE_SPLIT_RE.split(str(text)) if c and c.strip()]


def tag_matches(query_tag: str, track_tags: Iterable[str]) -> bool:
    """Тег запроса совпадает с тегом трека целиком или как подфраза по словам.

    'rap' совпадает с 'gangsta rap', но не с 'trap'.
    """
    q = f" {query_tag} "
    return any(q in f" {t} " for t in track_tags)


# ---------------------------------------------------------------- страны, языки, эпохи в запросе

COUNTRIES_EN = {
    "usa": "US", "us": "US", "american": "US", "america": "US", "uk": "GB", "british": "GB", "english": "GB",
    "england": "GB", "scottish": "GB", "german": "DE", "germany": "DE", "french": "FR", "france": "FR",
    "swedish": "SE", "sweden": "SE", "norwegian": "NO", "finnish": "FI", "icelandic": "IS", "danish": "DK",
    "dutch": "NL", "belgian": "BE", "japanese": "JP", "japan": "JP", "korean": "KR", "russian": "RU",
    "spanish": "ES", "spain": "ES", "italian": "IT", "italy": "IT", "canadian": "CA", "canada": "CA",
    "irish": "IE", "australian": "AU", "brazilian": "BR", "brazil": "BR", "mexican": "MX", "polish": "PL",
    "argentinian": "AR", "colombian": "CO", "jamaican": "JM", "portuguese": "PT", "greek": "GR", "turkish": "TR",
}
# язык текста — только в явном контексте («in spanish», «spanish lyrics»), иначе это страна
LANGUAGES_EN = {
    "english": "en", "spanish": "es", "french": "fr", "german": "de", "italian": "it", "portuguese": "pt",
    "russian": "ru", "japanese": "ja", "korean": "ko", "swedish": "sv", "finnish": "fi", "polish": "pl",
    "dutch": "nl", "turkish": "tr", "greek": "el", "norwegian": "no", "danish": "da", "icelandic": "is",
}
LANGUAGE_CUES = frozenset("lyrics language vocals sung sing singing songs".split())
_DECADE_RE = re.compile(r"^(19|20)?([0-9])0s$")


def decade_tag(year: int) -> str:
    """1983 -> '80s', 2004 -> '00s', 2012 -> '10s' (как в тегах Last.fm)."""
    return f"{(int(year) // 10 % 10)}0s"


def match_era(token: str) -> Tuple[Optional[str], Optional[int]]:
    """'80s' / '1980s' -> ('80s', None); '2003' -> ('00s', 2003); иначе (None, None)."""
    if re.fullmatch(r"(19[5-9]|20[0-2])[0-9]", token):
        return decade_tag(int(token)), int(token)
    m = _DECADE_RE.match(token)
    return (f"{m.group(2)}0s", None) if m else (None, None)
