"""HTTP-клиент к сервисам-кандгенам по контракту docs/candgen_api.md (сервис HNSW, релиз hnsw-v1).

Пайплайн сервис не использует: кандген hnsw (dense.py) ищет по тем же векторам треков локально, точным поиском.
Клиент нужен для проверки сервиса (scripts/make_embed.py, examples/candgen.ipynb).
Значения вида ${VAR} и ${VAR:-по умолчанию} берутся из переменных окружения (и файла .env).
"""
from __future__ import annotations

import json
import os
import re
import socket
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: Any) -> Any:
    """'${HNSW_API_KEY}' -> значение переменной (пусто, если не задана); '${X:-d}' -> d, если X пуста."""
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), value)
    return value


class CandgenError(RuntimeError):
    pass


def _is_timeout(e: BaseException) -> bool:
    return isinstance(e, (socket.timeout, TimeoutError)) or isinstance(getattr(e, "reason", None),
                                                                       (socket.timeout, TimeoutError))


class CandgenClient:
    """JSON по HTTP к одному сервису; заголовок X-API-Key, если задан api_key.

    retries: сколько раз повторить при обрыве соединения (сервис перезапускается, сеть моргнула).
    Таймаут и ответы с HTTP-ошибкой не повторяются, чтобы не умножать задержку.
    """

    def __init__(self, url: str, timeout: float = 5.0, api_key: Optional[str] = None, retries: int = 1):
        self.url = expand_env(url).rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.headers = {"Content-Type": "application/json"}
        key = expand_env(api_key) if api_key else ""
        if key:
            self.headers["X-API-Key"] = key

    def post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._call(path, json.dumps(payload).encode("utf8"))

    def get(self, path: str) -> Dict[str, Any]:
        return self._call(path, None)

    def _call(self, path: str, data: Optional[bytes]) -> Dict[str, Any]:
        err: Optional[BaseException] = None
        for _ in range(self.retries + 1):
            req = urllib.request.Request(self.url + path, data=data, headers=self.headers,
                                         method="GET" if data is None else "POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    raw = r.read()
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf8", "replace")[:300]
                raise CandgenError(f"{self.url}{path}: HTTP {e.code} {detail}") from None
            except (urllib.error.URLError, OSError) as e:
                err = e
                if _is_timeout(e):
                    break
                continue
            try:
                return json.loads(raw.decode("utf8"))
            except ValueError as e:
                raise CandgenError(f"{self.url}{path}: ответ не JSON ({e})") from None
        raise CandgenError(f"{self.url}{path}: {err}")
