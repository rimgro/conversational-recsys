"""Части запросов: на чём учатся кандгены, ранкер, на чём подбираем параметры и на чём проверяем итог.

Обучаемые кандгены (query_tfidf, query_document в BM25, two-tower) и ранкер учатся на разных запросах. Иначе на
своих же запросах кандген находит цель «по памяти», ранкер учится ему доверять, а на тесте этой памяти нет.
Поэтому train делится по тексту запроса (md5 нормализованного текста, 10 бакетов):

  index         бакеты 0–5   обучение кандгенов (тексты запросов приклеиваются к целям); без текстов, которые
                             встречаются в сплите index_excludes (test_public; exclude_test_texts)
  ranker_train  бакеты 6–8   обучение ранкера (сэмпл sizes.ranker_train)
  valid         бакет  9     подбор политики отбора, параметров, early stopping (сэмпл sizes.valid)
  test          test_public  только итоговая оценка (сэмпл sizes.test); параметры по нему не подбираем
  submit        test_public  все запросы: сабмит

Один и тот же текст всегда попадает в одну часть: перефразировок одного запроса между частями нет.
Схема и сэмплы как в бейзлайне team1 (reference/team1), чтобы метрики были сравнимы.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Optional

import pandas as pd

PARTS = ("index", "ranker_train", "valid", "test", "submit")
TRAIN_PARTS = ("index", "ranker_train", "valid")   # части из data.parts.train_split, остальные — из test_split


def normalize_query(text: Any) -> str:
    """Слова в нижнем регистре через пробел: 'Play  Song, by X!' -> 'play song by x'."""
    return " ".join(re.findall(r"\w+", str(text).casefold()))


def text_bucket(normalized: str) -> int:
    return int(hashlib.md5(normalized.encode()).hexdigest()[:8], 16) % 10


def sample(frame: pd.DataFrame, size: Optional[int], seed: int) -> pd.DataFrame:
    """Случайные size строк (распределение типов запросов сохраняется); None или больше строк нет — все."""
    if size is None or len(frame) <= size:
        return frame.reset_index(drop=True)
    return frame.sample(n=size, random_state=seed).reset_index(drop=True)


def part_split(part: str, pcfg: Dict[str, Any]) -> str:
    """Из какого сплита датасета берётся часть."""
    if part not in PARTS:
        raise ValueError(f"неизвестная часть {part!r}, есть: {', '.join(PARTS)}")
    return pcfg["train_split"] if part in TRAIN_PARTS else pcfg["test_split"]


def index_excludes(pcfg: Dict[str, Any]) -> str:
    """Сплит, тексты которого не идут в index (data.parts.index_excludes, по умолчанию test_split)."""
    return pcfg.get("index_excludes") or pcfg["test_split"]


def make_part(part: str, train_queries: pd.DataFrame, test_queries: pd.DataFrame,
              pcfg: Dict[str, Any], exclude_queries: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Запросы части (колонки *_queries.parquet + normalized), в фиксированном порядке.

    pcfg — data.parts: buckets {часть: [от, до)}, sizes {часть: N | null}, exclude_test_texts, seed.
    exclude_queries — запросы, тексты которых не идут в index (сплит index_excludes; по умолчанию test_queries):
    index не зависит от того, на каком тестовом сплите считаем ответы.
    """
    part_split(part, pcfg)
    seed, sizes = pcfg.get("seed", 42), pcfg.get("sizes", {})
    if part in ("test", "submit"):
        return sample(test_queries, sizes.get(part), seed)
    q = train_queries.copy()
    q["normalized"] = q["query"].map(normalize_query)
    lo, hi = pcfg["buckets"][part]
    buckets = q["normalized"].map(text_bucket)
    mask = (buckets >= lo) & (buckets < hi)
    if part == "index":
        if pcfg.get("exclude_test_texts", True):
            ex = test_queries if exclude_queries is None else exclude_queries
            mask &= ~q["normalized"].isin(set(ex["query"].map(normalize_query)))
        return q[mask].copy()
    return sample(q[mask], sizes.get(part), seed)


def check_disjoint(parts: Dict[str, pd.DataFrame]) -> None:
    """Части train не пересекаются по тексту запроса (иначе утечка между кандгенами, ранкером и валидацией)."""
    names = [p for p in TRAIN_PARTS if p in parts]
    texts = {p: set(parts[p]["normalized"]) for p in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = texts[a] & texts[b]
            if common:
                raise AssertionError(f"части {a} и {b} пересекаются по {len(common)} текстам запросов")
