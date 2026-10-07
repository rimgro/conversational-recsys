"""Описание выдачи через LLM. Модель видит только переданные факты; ошибка -> шаблон."""
from __future__ import annotations

from typing import Dict, List

from recsys.explain.base import BaseExplainer
from recsys.explain.stub import StubExplainer
from recsys.llm import BaseLLM
from recsys.schemas import Context, RankedTrack

EXPLAIN_SYSTEM = """You are a friendly music assistant in a chat.
Write a short reply in English (3-6 sentences) presenting the recommended tracks to the user.
Mention a few tracks by artist and title and say briefly why they fit the request.
Use ONLY the facts given below (artist, title, tags, reasons). Do not invent tracks, facts or links.
Do not reorder or drop tracks from the list: the full list is shown to the user separately."""


class LLMExplainer(BaseExplainer):
    def __init__(self, llm: BaseLLM, n_describe: int = 5, max_new_tokens: int = 300):
        self.llm = llm
        self.n_describe = n_describe
        self.max_new_tokens = max_new_tokens
        self.fallback = StubExplainer()
        self.last_raw = ""

    def build_messages(self, tracks: List[RankedTrack], ctx: Context) -> List[Dict[str, str]]:
        last = ctx.request.user_messages[-1] if ctx.request.user_messages else ""
        items = "\n".join(
            f"{t.rank}. {t.display_name} | tags: {', '.join(t.tags[:5])} | reasons: {'; '.join(t.reasons) or '-'}"
            for t in tracks[:self.n_describe]
        )
        user = (f"User's last message: {last}\nRequest summary: {ctx.summary.text}\n\n"
                f"Recommended tracks:\n{items}")
        return [{"role": "system", "content": EXPLAIN_SYSTEM}, {"role": "user", "content": user}]

    def describe(self, tracks: List[RankedTrack], ctx: Context) -> str:
        if not tracks:
            return self.fallback.describe(tracks, ctx)
        try:
            self.last_raw = self.llm.generate(self.build_messages(tracks, ctx), max_new_tokens=self.max_new_tokens)
        except Exception as e:
            self.last_raw = f"<error: {e}>"
            return self.fallback.describe(tracks, ctx)
        return self.last_raw.strip() or self.fallback.describe(tracks, ctx)
