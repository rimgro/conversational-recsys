"""Конфиг = обычный dict из YAML. Переопределения: load_config(path, {"ranker": {"type": "heuristic"}})."""
from __future__ import annotations

import copy
import os
import warnings
from typing import Any, Dict, List, Optional, Union

import yaml


def load_env(path: Optional[str] = None) -> None:
    """Переменные из .env (KEY=VALUE, # — комментарий) в окружение; уже заданные не перезаписываются.
    Файл — RECSYS_ENV_FILE или .env в текущей папке; RECSYS_ENV_FILE="" отключает (так в тестах)."""
    path = os.environ.get("RECSYS_ENV_FILE", ".env") if path is None else path
    if not path or not os.path.isfile(path):
        return
    with open(path, encoding="utf8") as f:
        for line in f:
            key, sep, value = line.strip().partition("=")
            if sep and key and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


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


def load_config(path: Union[str, List[str]] = "configs/default.yaml",
                overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """path — файл или список файлов: каждый следующий накладывается на предыдущий, затем overrides.
    Перед этим подхватывается .env (адреса и ключи сервисов, см. load_env)."""
    load_env()
    paths = [path] if isinstance(path, str) else list(path)
    cfg: Dict[str, Any] = {}
    for i, p in enumerate(paths):
        with open(p, encoding="utf8") as f:
            layer = yaml.safe_load(f) or {}
        cfg = layer if i == 0 else deep_update(cfg, layer)
    return deep_update(cfg, overrides)
