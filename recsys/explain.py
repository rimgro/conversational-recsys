"""Шаг 5: top-k треков + контекст -> текстовый ответ пользователю.

StubExplainer  заглушка: приветствие + список треков с причинами.
LLMExplainer   LLM пишет ответ только по переданным фактам; ошибка -> шаблон.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from recsys.llm import BaseLLM
from recsys.schemas import Context, RankedTrack


class BaseExplainer(ABC):
    """Шаг 5: top-k треков + контекст -> текстовый ответ пользователю."""

    @abstractmethod
    def describe(self, tracks: List[RankedTrack], ctx: Context) -> str:
        ...


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


def build_explainer(cfg: Dict[str, Any], llm: Optional[BaseLLM] = None) -> BaseExplainer:
    ecfg = cfg.get("explainer", {})
    kind = ecfg.get("type", "stub")
    if kind == "stub":
        return StubExplainer(n_describe=ecfg.get("n_describe", 10))
    if kind == "llm":
        if llm is None:
            raise ValueError("explainer.type=llm, но llm не передана")
        return LLMExplainer(llm, n_describe=ecfg.get("n_describe", 5),
                            max_new_tokens=ecfg.get("max_new_tokens", 300))
    raise ValueError(f"Неизвестный explainer.type: {kind}")
