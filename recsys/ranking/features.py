"""Признаки кандидатов для ранкера (сейчас для эвристики и отладки, позже для CatBoost)."""
from __future__ import annotations

import math
from typing import List, Sequence

import pandas as pd

from recsys.data.catalog import Catalog
from recsys.schemas import Context, FusedCandidate
from recsys.text import tag_matches

MISSING_RANK = 1000


def build_features(fused: List[FusedCandidate], ctx: Context, catalog: Catalog,
                   sources: Sequence[str]) -> pd.DataFrame:
    s, prof = ctx.summary, ctx.profile
    prof_total = sum(prof.tag_weights.values()) or 1.0
    seeds = {a.lower() for a in s.seed_artists}
    rows = []
    for fc in fused:
        p = catalog.pos(fc.track_id)
        row = catalog.df.iloc[p]
        tags = list(row["tags"]) + list(row["genres"])
        artist = row["artist"]
        q_hits = sum(tag_matches(t, tags) for t in s.include_tags)
        u_hits = sum(tag_matches(t, tags) for t in s.user_tags)
        feat = {
            "track_id": fc.track_id,
            "rrf_score": fc.score,
            "n_sources": len(fc.ranks),
            "log_popularity": math.log1p(float(row["popularity"])),
            "query_tag_hits": q_hits,
            "query_tag_frac": q_hits / len(s.include_tags) if s.include_tags else 0.0,
            "user_tag_hits": u_hits,
            "profile_tag_score": sum(prof.tag_weights.get(t, 0.0) for t in tags) / prof_total,
            "artist_listened": float(prof.artist_weights.get(artist, 0.0) > 0) if artist else 0.0,
            "artist_seed": float(artist.lower() in seeds) if artist else 0.0,
            "n_tags": len(row["tags"]),
        }
        for src in sources:
            feat[f"rank_{src}"] = fc.ranks.get(src, MISSING_RANK)
            feat[f"score_{src}"] = fc.scores.get(src, 0.0)
        rows.append(feat)
    return pd.DataFrame(rows)


def reasons_for(track_id: str, ctx: Context, catalog: Catalog, max_reasons: int = 3) -> List[str]:
    """Короткие причины для объяснения: совпавшие теги, сид-артист, знакомый артист."""
    tags = catalog.tags(track_id) + catalog.genres(track_id)
    artist = catalog.artist(track_id)
    out = [f"matches '{t}'" for t in ctx.summary.include_tags if tag_matches(t, tags)]
    if artist and artist.lower() in {a.lower() for a in ctx.summary.seed_artists}:
        out.append(f"by {artist}, as you asked")
    elif artist and artist in ctx.profile.artist_weights:
        out.append(f"you listen to {artist}")
    if not out:
        common = [t for t in ctx.profile.top_tags(10) if t in tags][:2]
        if common:
            out.append("close to your taste: " + ", ".join(common))
    return out[:max_reasons]
