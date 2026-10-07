"""Шаг 3: слияние списков кандидатов (RRF), жёсткие фильтры, дедупликация.

RRF: score(d) = sum_s w_s / (k + rank_s(d)); веса источников зависят от контекста запроса.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from recsys.data.catalog import Catalog
from recsys.schemas import Candidate, Context, FusedCandidate
from recsys.text import tag_matches


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


def apply_filters(fused: List[FusedCandidate], ctx: Context, catalog: Catalog,
                  exclude_listened: bool = True) -> List[FusedCandidate]:
    banned = set(ctx.request.shown_ids) | set(ctx.request.skipped_ids)
    if exclude_listened:
        banned |= ctx.profile.listened_ids
    ex_tags = ctx.summary.exclude_tags
    ex_artists = {a.lower() for a in ctx.summary.exclude_artists}
    out = []
    for fc in fused:
        if fc.track_id in banned or fc.track_id not in catalog:
            continue
        if ex_artists and catalog.artist(fc.track_id).lower() in ex_artists:
            continue
        if ex_tags:
            track_tags = catalog.tags(fc.track_id) + catalog.genres(fc.track_id)
            if any(tag_matches(t, track_tags) for t in ex_tags):
                continue
        out.append(fc)
    return out


def source_weights(cfg: Dict[str, Any], ctx: Context) -> Dict[str, float]:
    """Базовые веса источников, поправленные по контексту запроса."""
    f = cfg.get("fusion", {})
    w = dict(f.get("weights", {}))
    if not ctx.summary.has_query:            # «ещё», «что-нибудь»: больше доверяем истории
        for s, m in f.get("no_query_boost", {}).items():
            w[s] = w.get(s, 1.0) * m
    if ctx.profile.n_known_tracks == 0:      # холодный старт
        for s, m in f.get("cold_start_boost", {}).items():
            w[s] = w.get(s, 1.0) * m
    return w
