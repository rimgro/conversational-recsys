"""Reciprocal Rank Fusion: score(d) = sum_s w_s / (k + rank_s(d))."""
from __future__ import annotations

from typing import Dict, List, Optional

from recsys.schemas import Candidate, FusedCandidate


def rrf(lists: Dict[str, List[Candidate]], k: int = 60,
        weights: Optional[Dict[str, float]] = None) -> List[FusedCandidate]:
    weights = weights or {}
    fused: Dict[str, FusedCandidate] = {}
    for source, cands in lists.items():
        w = float(weights.get(source, 1.0))
        if w <= 0:
            continue
        for c in cands:
            fc = fused.get(c.track_id)
            if fc is None:
                fc = fused[c.track_id] = FusedCandidate(track_id=c.track_id, score=0.0)
            if source in fc.ranks:  # дубль внутри одного источника: берём лучший ранг
                continue
            fc.ranks[source] = c.rank
            fc.scores[source] = c.score
            fc.score += w / (k + c.rank)
    return sorted(fused.values(), key=lambda f: -f.score)
