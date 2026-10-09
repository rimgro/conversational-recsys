"""Наша часть <-> сервис HNSW по HTTP: клиент recsys/retrieval/remote.py против поддельного сервиса
по контракту docs/candgen_api.md (сам сервис — релиз hnsw-v1). BM25 локальный, сервиса у него нет."""
import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from recsys.config import deep_update, load_config
from recsys.pipeline import Pipeline
from recsys.schemas import Message, Request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(ROOT, "configs", "default.yaml")


def _api_cfg(cfg, url, **extra):
    base = load_config(DEFAULT, {"data": cfg["data"]})  # в default.yaml HNSW включён
    return deep_update(base, {"candgen": {"hnsw": {"url": url}, **extra}})


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _FakeEmbedder:
    """Вместо EmbeddingGemma: единичный вектор из 768 чисел; fail — модель не загрузилась."""

    def __init__(self, fail=False):
        self.fail, self.texts = fail, []

    def encode(self, text):
        if self.fail:
            raise ImportError("No module named 'sentence_transformers'")
        self.texts.append(text)
        return [0.0] * 767 + [1.0]


@pytest.fixture
def fake_hnsw(data):
    """Поддельный HNSW по контракту: запоминает запросы, отвечает треками каталога.
    Первое соединение к /hnsw/search обрывается без ответа — клиент должен повторить."""
    calls, dropped = [], []
    ids = [str(t) for t in data.catalog.track_ids[:50]]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._send({"status": "ok", "release": "test", "table_version": 1, "n_items": 50})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not dropped:
                dropped.append(1)
                self.close_connection = True
                return
            calls.append((self.path, body, dict(self.headers)))
            out = [t for t in ids if t not in set(body["exclude_ids"])][:body["k"]]
            self._send({"ids": out, "scores": [1.0 / (i + 1) for i in range(len(out))]})

        def _send(self, resp):
            raw = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", calls
    server.shutdown()


def test_pipeline_calls_hnsw(cfg, data, fake_hnsw, monkeypatch):
    url, calls = fake_hnsw
    monkeypatch.setenv("HNSW_API_KEY", "h456")
    pipe = Pipeline.from_config(_api_cfg(cfg, url), data.catalog)
    assert {r.name for r in pipe.retrievers} == {"bm25", "hnsw", "audio", "relisten"}
    hnsw = next(r for r in pipe.retrievers if r.name == "hnsw")
    hnsw.embedder = _FakeEmbedder()  # настоящую модель в тестах не грузим
    req = Request(dialog=[Message("user", "calm jazz, but not heavy metal")], history=data.requests[0].history,
                  shown_ids=[str(data.catalog.track_ids[0])])
    resp = pipe.run(req, debug=True)

    (path, body, headers), = calls  # один запрос: первый обрыв клиент повторил
    assert path == "/hnsw/search" and len(body["vector"]) == 768 and set(body) == {"vector", "k", "exclude_ids"}
    assert hnsw.embedder.texts == ["calm jazz, but not heavy metal"]
    assert body["exclude_ids"] == [str(data.catalog.track_ids[0])] and headers.get("X-Api-Key") == "h456"
    assert len(resp.debug["candidates"]["hnsw"]) == 49 and resp.tracks and hnsw.last_error is None
    assert hnsw.health() == {"url": url, "status": "ok", "index_version": "test/1", "n_items": 50}


def test_hnsw_down_does_not_break_pipeline(cfg, data):
    port = _free_port()  # на нём никто не слушает
    pipe = Pipeline.from_config(_api_cfg(cfg, f"http://127.0.0.1:{port}", timeout=0.5), data.catalog)
    hnsw = next(r for r in pipe.retrievers if r.name == "hnsw")
    hnsw.embedder = _FakeEmbedder()
    with pytest.warns(UserWarning, match="кандген недоступен"):
        resp = pipe.run(data.requests[0], debug=True)
    assert resp.tracks  # локальные relisten, audio и bm25 работают
    assert hnsw.last_error and hnsw.n_errors == 1 and hnsw.health()["status"] == "unavailable"


def test_expand_env(monkeypatch):
    from recsys.retrieval.remote import CandgenClient, expand_env
    monkeypatch.delenv("HNSW_URL", raising=False)
    assert expand_env("${HNSW_URL:-http://localhost:8001}") == "http://localhost:8001"
    monkeypatch.setenv("HNSW_URL", "http://10.0.0.7:8001/")
    assert expand_env("${HNSW_URL:-http://localhost:8001}") == "http://10.0.0.7:8001/"
    c = CandgenClient("${HNSW_URL}", api_key="${NO_SUCH_VAR}")
    assert c.url == "http://10.0.0.7:8001" and "X-API-Key" not in c.headers


def test_client_config(monkeypatch):
    from recsys.retrieval.remote import CandgenClient
    monkeypatch.setenv("HNSW_URL", "http://1.2.3.4:8000")
    cfg = load_config(DEFAULT)
    client = CandgenClient.from_config(cfg, "hnsw", {"timeout": 9})
    assert (client.url, client.timeout, client.retries) == ("http://1.2.3.4:8000", 9, 1)
    monkeypatch.delenv("HNSW_URL")
    with pytest.raises(ValueError, match="candgen.hnsw.url"):
        CandgenClient.from_config(cfg, "hnsw")


def test_load_env(tmp_path, monkeypatch):
    from recsys.config import load_env
    env = tmp_path / ".env"
    env.write_text("# комментарий\nRECSYS_T1=abc\nRECSYS_T2 = 'q'\nBROKEN\n", encoding="utf8")
    monkeypatch.setenv("RECSYS_T2", "уже задано")
    monkeypatch.delenv("RECSYS_T1", raising=False)
    load_env(str(env))
    assert os.environ["RECSYS_T1"] == "abc" and os.environ["RECSYS_T2"] == "уже задано"
    monkeypatch.delenv("RECSYS_T1")


def test_hnsw_sends_vector_built_locally(cfg, data, fake_hnsw):
    """В HNSW уходит вектор (по рецепту сервиса), а не текст; модель не загрузилась — источник пропускается."""
    from recsys.retrieval.query_embedder import query_text
    url, calls = fake_hnsw

    pipe = Pipeline.from_config(_api_cfg(cfg, url), data.catalog)
    hnsw = next(r for r in pipe.retrievers if r.name == "hnsw")
    hnsw.embedder = _FakeEmbedder()
    req = Request(dialog=[Message("user", "calm  jazz\nfor   night")], history=data.requests[0].history)
    pipe.run(req)
    body = next(b for p, b, h in calls if p == "/hnsw/search")
    assert len(body["vector"]) == 768 and "query" not in body and hnsw.embedder.texts == ["calm  jazz\nfor   night"]
    assert query_text("calm  jazz\nfor   night") == "task: search result | query: calm jazz for night"

    calls.clear()
    hnsw.embedder = _FakeEmbedder(fail=True)
    with pytest.warns(UserWarning, match="вектор запроса не построен"):
        resp = pipe.run(req, debug=True)
    assert not [b for p, b, h in calls if p == "/hnsw/search"] and hnsw.embed_error
    assert resp.debug["n_candidates"]["hnsw"] == 0 and resp.tracks
