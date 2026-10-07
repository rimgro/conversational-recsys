from __future__ import annotations

from typing import Any, Dict, Optional

from recsys.data.catalog import Catalog
from recsys.dialog.base import BaseSummarizer
from recsys.dialog.llm_summarizer import LLMSummarizer
from recsys.dialog.rule import RuleSummarizer
from recsys.llm import BaseLLM


def build_summarizer(cfg: Dict[str, Any], catalog: Catalog, llm: Optional[BaseLLM] = None) -> BaseSummarizer:
    scfg = cfg.get("summarizer", {})
    rule = RuleSummarizer(catalog, min_tag_count=scfg.get("min_tag_count", 2))
    kind = scfg.get("type", "rule")
    if kind == "rule":
        return rule
    if kind == "llm":
        if llm is None:
            raise ValueError("summarizer.type=llm, но llm не передана")
        return LLMSummarizer(llm, fallback=rule)
    raise ValueError(f"Неизвестный summarizer.type: {kind}")


__all__ = ["BaseSummarizer", "RuleSummarizer", "LLMSummarizer", "build_summarizer"]
