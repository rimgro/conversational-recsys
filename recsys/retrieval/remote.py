"""Кандидаты из образа 1 (candgen/: BM25 + HNSW) по HTTP.

Кандгены обучаются и живут отдельно; здесь только predict-вызовы. Тот же интерфейс, что у
локальных источников: search(ctx) -> [Candidate]. Ошибка сети / сервиса -> пустой список,
предупреждение и last_error (пайплайн не падает, остальные источники работают).

  bm25_api   POST /bm25/search  слова из саммари -> индекс genres / tags / title
  hnsw_api   POST /hnsw/search  треки истории с весами (+ лайки, артисты-сиды) -> индекс audio

Конфиг: раздел candgen (url, timeout, api_key, headers) + источники с type: bm25_api | hnsw_api.
Значения вида ${VAR} и ${VAR:-по умолчанию} берутся из переменных окружения.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional

from recsys.data.catalog import Catalog
from recsys.retrieval.bm25 import index_words
from recsys.retrieval.sources import BaseRetriever, TitleRetriever, taste_track_weights
from recsys.schemas import Candidate, Context

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: Any) -> Any:
    """'${CANDGEN_API_KEY}' -> значение переменной (пусто, если не задана); '${X:-d}' -> d, если X пуста."""
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), value)
    return value


class CandgenError(RuntimeError):
    pass


class CandgenClient:
    """POST JSON в сервис кандгенов. Заголовок X-API-Key, если задан api_key; любые доп. заголовки."""

    def __init__(self, url: str, timeout: float = 3.0, api_key: Optional[str] = None,
                 headers: Optional[Dict[str, str]] = None):
        self.url = expand_env(url).rstrip("/")
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        self.headers.update({k: expand_env(v) for k, v in (headers or {}).items() if expand_env(v)})
        key = expand_env(api_key) if api_key else ""
        if key:
            self.headers["X-API-Key"] = key

    def post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(self.url + path, data=json.dumps(payload).encode("utf8"),
                                     headers=self.headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf8", "replace")[:300]
            raise CandgenError(f"{path}: HTTP {e.code} {detail}") from None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            raise CandgenError(f"{path}: {e}") from None

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], source_cfg: Optional[Dict[str, Any]] = None) -> "CandgenClient":
        c = {**cfg.get("candgen", {}), **{k: v for k, v in (source_cfg or {}).items()
                                          if k in ("url", "timeout", "api_key", "headers")}}
        return cls(c.get("url", "http://localhost:8000"), timeout=c.get("timeout", 3.0),
                   api_key=c.get("api_key"), headers=c.get("headers"))


class _RemoteRetriever(BaseRetriever):
    path = ""

    def __init__(self, name: str, catalog: Catalog, client: CandgenClient, index: str, top_k: int = 200):
        super().__init__(catalog, top_k)
        self.name = name
        self.client = client
        self.index = index
        self.last_error: Optional[str] = None
        self.last_response: Optional[Dict[str, Any]] = None

    def payload(self, ctx: Context, k: int) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def search(self, ctx: Context, top_k: Optional[int] = None) -> List[Candidate]:
        body = self.payload(ctx, top_k or self.top_k)
        if body is None:
            return []
        body.update(index=self.index, exclude_ids=sorted(ctx.banned_ids))
        try:
            resp = self.client.post(self.path, body)
        except CandgenError as e:
            if self.last_error is None:  # одно предупреждение на источник, дальше только last_error
                warnings.warn(f"{self.name}: кандген недоступен, источник пропущен ({e})")
            self.last_error = str(e)
            return []
        self.last_error = None
        self.last_response = {k: v for k, v in resp.items() if k not in ("ids", "scores")}
        return [Candidate(track_id=str(t), source=self.name, score=float(s), rank=r)
                for r, (t, s) in enumerate(zip(resp.get("ids", []), resp.get("scores", [])), start=1)]


class BM25APIRetriever(_RemoteRetriever):
    """query: 'tags' — теги, страна/эпоха и латинские слова саммари; 'title' — артист + название (с транслитом)."""
    path = "/bm25/search"
    MAX_WORDS = 256

    def __init__(self, name: str, catalog: Catalog, client: CandgenClient, index: str = "genres",
                 top_k: int = 200, query: str = "tags"):
        super().__init__(name, catalog, client, index, top_k)
        if query not in ("tags", "title"):
            raise ValueError(f"{name}: query должен быть tags или title, а не {query!r}")
        self.query = query

    def words(self, ctx: Context) -> List[str]:
        s = ctx.summary
        if self.query == "title":
            words = TitleRetriever.query_text(ctx).split()
        else:
            words = [w for t in s.include_tags for w in index_words(t)] + index_words(s.query)
            # слова исключённых тегов убираем, если они не входят в желаемые ('рок, но не хард-рок')
            wanted = {w for t in s.include_tags for w in index_words(t)}
            banned = {w for t in s.exclude_tags for w in index_words(t)} - wanted
            words = [w for w in words if w not in banned]
        return list(Counter(words))[:self.MAX_WORDS]

    def payload(self, ctx: Context, k: int) -> Optional[Dict[str, Any]]:
        words = self.words(ctx)
        return {"words": words, "k": k} if words else None


class HNSWAPIRetriever(_RemoteRetriever):
    path = "/hnsw/search"
    MAX_TRACKS = 10_000

    def payload(self, ctx: Context, k: int) -> Optional[Dict[str, Any]]:
        weights = taste_track_weights(ctx, self.catalog)
        if not weights:
            return None
        top = sorted(weights.items(), key=lambda kv: -kv[1])[:self.MAX_TRACKS]
        return {"track_ids": [t for t, _ in top], "weights": [float(w) for _, w in top], "k": k}
