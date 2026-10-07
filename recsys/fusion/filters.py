"""Жёсткие фильтры после слияния: исключённые теги/артисты, показанное, прослушанное."""
from __future__ import annotations

from typing import List

from recsys.data.catalog import Catalog
from recsys.schemas import Context, FusedCandidate
from recsys.text import tag_matches


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
