"""Конфиг = обычный dict из YAML. Переопределения: load_config(path, {"ranker": {"type": "heuristic"}})."""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import yaml


def deep_update(base: Dict[str, Any], upd: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str = "configs/default.yaml", overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    with open(path, encoding="utf8") as f:
        cfg = yaml.safe_load(f) or {}
    return deep_update(cfg, overrides)
