from __future__ import annotations

import math
from typing import Iterable, Sequence


def hit_rate_at_k(rec: Sequence[str], targets: Iterable[str], k: int) -> float:
    t = set(targets)
    return float(any(r in t for r in rec[:k])) if t else 0.0


def recall_at_k(rec: Sequence[str], targets: Iterable[str], k: int) -> float:
    t = set(targets)
    return len(t & set(rec[:k])) / len(t) if t else 0.0


def ndcg_at_k(rec: Sequence[str], targets: Iterable[str], k: int) -> float:
    t = set(targets)
    if not t:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(rec[:k]) if r in t)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(t), k)))
    return dcg / idcg


def mrr_at_k(rec: Sequence[str], targets: Iterable[str], k: int) -> float:
    t = set(targets)
    for i, r in enumerate(rec[:k]):
        if r in t:
            return 1.0 / (i + 1)
    return 0.0
