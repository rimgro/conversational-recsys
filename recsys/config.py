"""Конфиг = обычный dict из YAML. Переопределения: load_config(path, {"ranker": {"type": "heuristic"}})."""
from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, Optional

import yaml


def deep_update(base: Dict[str, Any], upd: Optional[Dict[str, Any]], _path: str = "") -> Dict[str, Any]:
    """Рекурсивно накладывает upd на копию base.

    Ключ, которого нет в непустом разделе base, скорее всего опечатка: выдаём предупреждение
    (значение всё равно записывается, чтобы можно было добавлять новые параметры).
    """
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        key = f"{_path}.{k}" if _path else str(k)
        if out and k not in out:
            warnings.warn(f"config: ключа '{key}' нет в базовом конфиге (опечатка?)", stacklevel=2)
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v, key)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str = "configs/default.yaml", overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    with open(path, encoding="utf8") as f:
        cfg = yaml.safe_load(f) or {}
    return deep_update(cfg, overrides)
