"""Саммари диалога через LLM. Любая ошибка разбора -> фолбэк на правила."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from recsys.dialog.base import BaseSummarizer
from recsys.dialog.prompts import SUMMARY_SYSTEM, SUMMARY_USER
from recsys.dialog.rule import RuleSummarizer, render_summary_text
from recsys.llm import BaseLLM, extract_json
from recsys.schemas import DialogSummary, Request, UserProfile
from recsys.text import normalize_tag

_LIST_FIELDS = ["include_tags", "exclude_tags", "seed_artists", "exclude_artists"]


class LLMSummarizer(BaseSummarizer):
    def __init__(self, llm: BaseLLM, fallback: RuleSummarizer, max_dialog_messages: int = 12):
        self.llm = llm
        self.fallback = fallback
        self.max_dialog_messages = max_dialog_messages
        self.last_raw: str = ""  # для отладки в ноутбуке

    def build_messages(self, request: Request, profile: Optional[UserProfile]) -> List[Dict[str, str]]:
        dialog = "\n".join(f"{m.role}: {m.text}" for m in request.dialog[-self.max_dialog_messages:])
        user = SUMMARY_USER.format(
            user_info=request.user_info or "unknown",
            history_tags=", ".join(profile.top_tags(10)) if profile else "unknown",
            history_artists=", ".join(profile.top_artists(5)) if profile else "unknown",
            dialog=dialog or "(empty)",
        )
        return [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": user}]

    def summarize(self, request: Request, profile: Optional[UserProfile] = None) -> DialogSummary:
        base = self.fallback.summarize(request, profile)
        try:
            self.last_raw = self.llm.generate(self.build_messages(request, profile))
        except Exception as e:  # модель не загрузилась, OOM и т.п.
            self.last_raw = f"<error: {e}>"
            return base
        data = extract_json(self.last_raw)
        if not data:
            return base
        return self._merge(base, data)

    @staticmethod
    def _merge(base: DialogSummary, data: Dict[str, Any]) -> DialogSummary:
        """Поля LLM поверх правил; пустые поля LLM не затирают найденное правилами."""
        out = DialogSummary(**{**base.__dict__})
        for f in _LIST_FIELDS:
            vals = data.get(f)
            if isinstance(vals, list) and vals:
                vals = [str(v) for v in vals if str(v).strip()]
                if f.endswith("tags"):
                    vals = [normalize_tag(v) for v in vals]
                setattr(out, f, list(dict.fromkeys(vals)))
        for f in ["mood", "energy"]:
            v = data.get(f)
            if isinstance(v, str) and v.strip() and v.strip().lower() != "null":
                setattr(out, f, v.strip().lower())
        if isinstance(data.get("query"), str) and data["query"].strip():
            out.query = data["query"].strip()
        out.text = str(data.get("summary") or "").strip() or render_summary_text(out)
        out.source = "llm+rule"
        return out
