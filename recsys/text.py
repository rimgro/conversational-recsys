"""Текстовые утилиты: нормализация тегов, токенизация, сопоставление тегов.

Всё на английском: диалоги, саммари и теги Last.fm англоязычные.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable, Iterator, List, Tuple

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

_CLAUSE_SPLIT_RE = re.compile(r"[.,;!?\n]|\bbut\b|\bhowever\b|\bthough\b", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def normalize_tag(tag: str) -> str:
    """'Hip-Hop' -> 'hip hop', 'Lo-Fi' -> 'lo fi', 'R&B' -> 'rnb'."""
    t = str(tag).lower().strip()
    if t in TAG_SYNONYMS:
        return TAG_SYNONYMS[t]
    t = re.sub(r"[-_/]+", " ", t)
    t = re.sub(r"[^a-z0-9&' ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if t in TAG_SYNONYMS:
        return TAG_SYNONYMS[t]
    words = [_WORD_SYNONYMS.get(w, w) for w in t.split()]
    return " ".join(words)


def tokenize(text: str) -> List[str]:
    """Нижний регистр, дефисы как пробелы, синонимы слов. Стоп-слова НЕ удаляются."""
    t = str(text).lower().replace("&", " and ")
    t = re.sub(r"[-_/]+", " ", t)
    tokens: List[str] = []
    for tok in _TOKEN_RE.findall(t):
        tokens.extend(_WORD_SYNONYMS.get(tok, tok).split())
    return tokens


def content_tokens(text: str) -> List[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def split_clauses(text: str) -> List[str]:
    return [c.strip() for c in _CLAUSE_SPLIT_RE.split(str(text)) if c and c.strip()]


def ngrams(tokens: List[str], n_max: int) -> Iterator[Tuple[int, int, str]]:
    """Все n-граммы длиной от n_max до 1: (start, end, 'phrase'), длинные первыми."""
    for n in range(min(n_max, len(tokens)), 0, -1):
        for i in range(len(tokens) - n + 1):
            yield i, i + n, " ".join(tokens[i:i + n])


def tag_matches(query_tag: str, track_tags: Iterable[str]) -> bool:
    """Тег запроса совпадает с тегом трека целиком или как подфраза по словам.

    'rap' совпадает с 'gangsta rap', но не с 'trap'.
    """
    q = f" {query_tag} "
    return any(q in f" {t} " for t in track_tags)


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf8")).hexdigest()[:8], 16)
