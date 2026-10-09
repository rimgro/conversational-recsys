"""Кандидаты из сервисов-кандгенов по HTTP: BM25 (ветка bm25-cards) и HNSW (релиз hnsw-v1), оба на сервере.

Кандгены обучаются и живут отдельно; здесь только predict-вызовы по контракту docs/candgen_api.md.
Тот же интерфейс, что у локальных источников: search(ctx) -> [Candidate]. Ошибка сети / сервиса ->
пустой список, предупреждение и last_error (пайплайн не падает, остальные источники работают).

  bm25       POST /bm25/search  слова из саммари -> индекс cards (вся карточка трека)  (candgen.bm25)
  hnsw       POST /hnsw/search  вектор последней реплики (EmbeddingGemma, строим сами) -> LanceDB  (candgen.hnsw);
                                сервис принимает только vector: без модели источник пропускается

Конфиг: раздел candgen (timeout, retries; bm25 / hnsw: url, api_key) + источники retrieval.bm25 / retrieval.hnsw.
Значения вида ${VAR} и ${VAR:-по умолчанию} берутся из переменных окружения (и файла .env).
"""
from __future__ import annotations

import json
import os
import re
import socket
import urllib.error
import urllib.request
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional

from recsys.data.catalog import Catalog
from recsys.retrieval.local_index import index_words
from recsys.retrieval.sources import BaseRetriever
from recsys.schemas import Candidate, Context

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_CLIENT_KEYS = ("url", "timeout", "retries", "api_key")


def expand_env(value: Any) -> Any:
    """'${BM25_API_KEY}' -> значение переменной (пусто, если не задана); '${X:-d}' -> d, если X пуста."""
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

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], service: str,
                    source_cfg: Optional[Dict[str, Any]] = None) -> "CandgenClient":
        """Общие настройки candgen + candgen.<service> (например, bm25) + ключи из самого источника."""
        common = cfg.get("candgen", {})
        c = {**{k: v for k, v in common.items() if k in _CLIENT_KEYS}, **common.get(service, {}),
             **{k: v for k, v in (source_cfg or {}).items() if k in _CLIENT_KEYS}}
        if not expand_env(c.get("url") or ""):
            raise ValueError(f"candgen.{service}.url не задан (для bm25 — переменная BM25_URL или файл .env)")
        return cls(c["url"], timeout=c.get("timeout", 5.0), api_key=c.get("api_key"),
                   retries=c.get("retries", 1))


class _RemoteRetriever(BaseRetriever):
    remote = True        # Pipeline опрашивает удалённые источники параллельно
    service = ""         # раздел candgen.<service> в конфиге
    path = ""
    health_path = ""

    def __init__(self, name: str, catalog: Catalog, client: CandgenClient, index: Optional[str] = None,
                 top_k: int = 200):
        super().__init__(catalog, top_k)
        self.name = name
        self.client = client
        self.index = index
        self.last_error: Optional[str] = None
        self.n_errors = 0  # сколько запросов прошло без кандидатов этого источника

    def payload(self, ctx: Context, k: int) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def search(self, ctx: Context, top_k: Optional[int] = None) -> List[Candidate]:
        body = self.payload(ctx, top_k or self.top_k)
        if body is None:
            return []
        body["exclude_ids"] = sorted(ctx.banned_ids)
        if self.index is not None:  # у HNSW одна таблица, поля index в его API нет
            body["index"] = self.index
        try:
            resp = self.client.post(self.path, body)
        except CandgenError as e:
            if self.last_error is None:  # одно предупреждение на источник, дальше только last_error
                warnings.warn(f"{self.name}: кандген недоступен, источник пропущен ({e})")
            self.last_error = str(e)
            self.n_errors += 1
            return []
        self.last_error = None
        return [Candidate(track_id=str(t), source=self.name, score=float(s), rank=r)
                for r, (t, s) in enumerate(zip(resp.get("ids", []), resp.get("scores", [])), start=1)]

    def health(self) -> Dict[str, Any]:
        """Состояние сервиса и версия нужного индекса (для ноутбука и логов)."""
        try:
            resp = self.client.get(self.health_path)
        except CandgenError as e:
            return {"url": self.client.url, "status": "unavailable", "error": str(e)}
        idx = resp.get("indexes", {}).get(self.index)
        return {"url": self.client.url, "status": resp.get("status") if idx else f"нет индекса {self.index!r}",
                "index_version": (idx or {}).get("index_version"), "n_items": (idx or {}).get("n_items")}


class BM25APIRetriever(_RemoteRetriever):
    """Слова запроса: слова include_tags и query из саммари, без слов исключённых тегов."""
    service = "bm25"
    path = "/bm25/search"
    health_path = "/health"
    MAX_WORDS = 256

    def __init__(self, name: str, catalog: Catalog, client: CandgenClient, index: str = "cards",
                 top_k: int = 200):
        super().__init__(name, catalog, client, index, top_k)

    def words(self, ctx: Context) -> List[str]:
        s = ctx.summary
        words = [w for t in s.include_tags for w in index_words(t)] + index_words(s.query)
        # слова исключённых тегов убираем, если они не входят в желаемые ('rock, but not hard rock')
        wanted = {w for t in s.include_tags for w in index_words(t)}
        banned = {w for t in s.exclude_tags for w in index_words(t)} - wanted
        return list(Counter(w for w in words if w not in banned))[:self.MAX_WORDS]

    def payload(self, ctx: Context, k: int) -> Optional[Dict[str, Any]]:
        words = self.words(ctx)
        return {"words": words, "k": k} if words else None


class HNSWAPIRetriever(_RemoteRetriever):
    """Текстовый семантический поиск в LanceDB по последней реплике пользователя (профиль, по замерам автора, вредит).
    Вектор EmbeddingGemma-2 строим сами (recsys/retrieval/query_embedder.py) и отправляем vector: текст сервис
    не принимает. Модель не загрузилась (нет sentence-transformers, GPU или доступа к ней) — предупреждение один раз,
    дальше источник пропускается, остальные работают."""
    service = "hnsw"
    path = "/hnsw/search"
    health_path = "/health"
    MAX_CHARS = 8192

    def __init__(self, name: str, catalog: Catalog, client: CandgenClient, top_k: int = 200, embedder: Any = None):
        super().__init__(name, catalog, client, top_k=top_k)
        self.embedder = embedder
        self.embed_error: Optional[str] = None

    def payload(self, ctx: Context, k: int) -> Optional[Dict[str, Any]]:
        messages = ctx.request.user_messages
        text = messages[-1].strip()[:self.MAX_CHARS] if messages else ""
        if not text:
            return None
        if self.embedder is None or self.embed_error is not None:
            return None
        try:
            return {"vector": self.embedder.encode(text), "k": k}
        except Exception as e:  # нет sentence-transformers / GPU / доступа к модели
            self.embed_error = f"{type(e).__name__}: {e}"
            warnings.warn(f"{self.name}: вектор запроса не построен, источник пропущен ({self.embed_error})")
            return None

    def health(self) -> Dict[str, Any]:
        try:
            resp = self.client.get(self.health_path)
        except CandgenError as e:
            return {"url": self.client.url, "status": "unavailable", "error": str(e)}
        return {"url": self.client.url, "status": resp.get("status"),
                "index_version": f"{resp.get('release')}/{resp.get('table_version')}", "n_items": resp.get("n_items")}
