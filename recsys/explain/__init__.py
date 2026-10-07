from __future__ import annotations

from typing import Any, Dict, Optional

from recsys.explain.base import BaseExplainer
from recsys.explain.llm_explainer import LLMExplainer
from recsys.explain.stub import StubExplainer
from recsys.llm import BaseLLM


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


__all__ = ["BaseExplainer", "StubExplainer", "LLMExplainer", "build_explainer"]
