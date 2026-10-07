from __future__ import annotations

from typing import Any, Dict

from recsys.ranking.api import APIRanker
from recsys.ranking.base import BaseRanker
from recsys.ranking.diversity import cap_per_artist
from recsys.ranking.features import build_features, reasons_for
from recsys.ranking.heuristic import HeuristicRanker
from recsys.ranking.stub import StubRanker


def build_ranker(cfg: Dict[str, Any]) -> BaseRanker:
    rcfg = cfg.get("ranker", {})
    kind = rcfg.get("type", "stub")
    if kind == "stub":
        return StubRanker()
    if kind == "heuristic":
        return HeuristicRanker(rcfg.get("weights"))
    if kind == "api":
        return APIRanker(rcfg["api_url"], timeout=rcfg.get("timeout", 2.0), api_key=rcfg.get("api_key"))
    raise ValueError(f"Неизвестный ranker.type: {kind}")


__all__ = ["BaseRanker", "StubRanker", "HeuristicRanker", "APIRanker", "build_ranker",
           "build_features", "reasons_for", "cap_per_artist"]
