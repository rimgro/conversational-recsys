"""Наша часть <-> сервис BM25 по HTTP: клиент recsys/retrieval/remote.py против поддельного сервиса
по контракту docs/candgen_api.md (сам сервис — ветка dev/bm25)."""
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
OVERLAY = os.path.join(ROOT, "configs", "candgen.yaml")


def _api_cfg(cfg, url, **extra):
    base = load_config([os.path.join(ROOT, "configs", "default.yaml"), OVERLAY], {"data": cfg["data"]})
    return deep_update(base, {"candgen": {"bm25": {"url": url}, **extra}})


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fake_bm25(data):
    """Поддельный BM25 по контракту: запоминает запросы, отвечает треками каталога.
    Первое соединение к индексу tags обрывается без ответа — клиент должен повторить."""
    calls, dropped = [], []
    ids = [str(t) for t in data.catalog.track_ids[:50]]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._send({"status": "ok", "indexes": {"genres": {"index_version": "test", "n_items": 50}}})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if body["index"] == "tags" and not dropped:
                dropped.append(1)
                self.close_connection = True
                return
            calls.append((self.path, body, dict(self.headers)))
            out = [t for t in ids if t not in set(body["exclude_ids"])][:body["k"]]
            self._send({"ids": out, "scores": [1.0 / (i + 1) for i in range(len(out))], "index": body["index"],
                        "index_version": "test"})

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


def test_pipeline_calls_bm25(cfg, data, fake_bm25, monkeypatch):
    url, calls = fake_bm25
    monkeypatch.setenv("BM25_API_KEY", "k123")
    pipe = Pipeline.from_config(_api_cfg(cfg, url), data.catalog)
    names = {r.name for r in pipe.retrievers}
    assert {"bm25_genres", "bm25_tags", "audio", "relisten"} <= names and "bm25" not in names
    req = Request(dialog=[Message("user", "спокойный джаз, но не хард-рок")], history=data.requests[0].history,
                  shown_ids=[str(data.catalog.track_ids[0])])
    resp = pipe.run(req, debug=True)

    by_index = {b["index"]: (path, b, h) for path, b, h in calls}
    path, body, headers = by_index["genres"]
    assert path == "/bm25/search" and "jazz" in body["words"] and "hard" not in body["words"]
    assert body["exclude_ids"] == [str(data.catalog.track_ids[0])]
    assert headers.get("X-Api-Key") == "k123"
    assert len(resp.debug["candidates"]["bm25_tags"]) == 49 and resp.tracks  # tags ответил после повтора
    bm25 = next(r for r in pipe.retrievers if r.name == "bm25_genres")
    assert bm25.last_error is None
    assert bm25.health() == {"url": url, "status": "ok", "index_version": "test", "n_items": 50}


def test_bm25_down_does_not_break_pipeline(cfg, data):
    port = _free_port()  # на нём никто не слушает
    pipe = Pipeline.from_config(_api_cfg(cfg, f"http://127.0.0.1:{port}", timeout=0.5), data.catalog)
    with pytest.warns(UserWarning, match="кандген недоступен"):
        resp = pipe.run(data.requests[0], debug=True)
    assert resp.tracks  # локальные relisten / title / history / audio / popular работают
    remote = [r for r in pipe.retrievers if getattr(r, "remote", False)]
    assert remote and all(r.last_error and r.n_errors == 1 for r in remote)
    assert remote[0].health()["status"] == "unavailable"


def test_expand_env(monkeypatch):
    from recsys.retrieval.remote import CandgenClient, expand_env
    monkeypatch.delenv("BM25_URL", raising=False)
    assert expand_env("${BM25_URL:-http://localhost:8001}") == "http://localhost:8001"
    monkeypatch.setenv("BM25_URL", "http://10.0.0.7:8001/")
    assert expand_env("${BM25_URL:-http://localhost:8001}") == "http://10.0.0.7:8001/"
    c = CandgenClient("${BM25_URL}", api_key="${NO_SUCH_VAR}", headers={"x-trace": "${TRACE:-abc}", "y": "${NONE}"})
    assert c.url == "http://10.0.0.7:8001" and "X-API-Key" not in c.headers
    assert c.headers["x-trace"] == "abc" and "y" not in c.headers


def test_client_config(monkeypatch):
    from recsys.retrieval.remote import CandgenClient
    monkeypatch.setenv("BM25_URL", "http://1.2.3.4:8000")
    cfg = load_config([os.path.join(ROOT, "configs", "default.yaml"), OVERLAY])
    client = CandgenClient.from_config(cfg, "bm25", {"timeout": 9})
    assert (client.url, client.timeout, client.retries) == ("http://1.2.3.4:8000", 9, 1)
    monkeypatch.delenv("BM25_URL")
    with pytest.raises(ValueError, match="candgen.bm25.url"):
        CandgenClient.from_config(cfg, "bm25")


def test_load_env(tmp_path, monkeypatch):
    from recsys.config import load_env
    env = tmp_path / ".env"
    env.write_text("# комментарий\nRECSYS_T1=abc\nRECSYS_T2 = 'q'\nBROKEN\n", encoding="utf8")
    monkeypatch.setenv("RECSYS_T2", "уже задано")
    monkeypatch.delenv("RECSYS_T1", raising=False)
    load_env(str(env))
    assert os.environ["RECSYS_T1"] == "abc" and os.environ["RECSYS_T2"] == "уже задано"
    monkeypatch.delenv("RECSYS_T1")
