"""ЗАГЛУШКА описания: приветствие + список треков с причинами."""
from __future__ import annotations

from typing import List

from recsys.explain.base import BaseExplainer
from recsys.schemas import Context, RankedTrack


class StubExplainer(BaseExplainer):
    def __init__(self, n_describe: int = 10):
        self.n_describe = n_describe

    def describe(self, tracks: List[RankedTrack], ctx: Context) -> str:
        if not tracks:
            return "Hi! Sorry, I couldn't find anything matching your request. Could you rephrase it?"
        greeting = "Hi! Here is what I picked for you"
        if ctx.summary.include_tags or ctx.summary.seed_artists:
            greeting += " (" + ", ".join(ctx.summary.include_tags + ctx.summary.seed_artists) + ")"
        lines = [greeting + ":"]
        for t in tracks[:self.n_describe]:
            tags = ", ".join(t.tags[:3])
            why = f" — {'; '.join(t.reasons)}" if t.reasons else ""
            lines.append(f"{t.rank}. {t.display_name} [{tags}]{why}")
        return "\n".join(lines)
