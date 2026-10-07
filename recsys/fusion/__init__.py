from __future__ import annotations

from typing import Any, Dict

from recsys.fusion.filters import apply_filters
from recsys.fusion.rrf import rrf
from recsys.schemas import Context


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


__all__ = ["rrf", "apply_filters", "source_weights"]
