"""Клиент сервиса HNSW (recsys/candgen/remote.py) против поддельного сервиса по контракту docs/candgen_api.md
и кандген hnsw с вектором запроса, построенным у нас (recsys/candgen/query_embedder.py)."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from recsys.candgen import REGISTRY
from recsys.candgen.remote import CandgenClient, CandgenError, expand_env
from recsys.schemas import Message, Request


@pytest.fixture
def fake_hnsw():
    """Поддельный HNSW: запоминает запросы. Первое соединение к /hnsw/search обрывается — клиент должен повторить."""
    calls, dropped = [], []

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
            self._send({"ids": ["a", "b"], "scores": [0.9, 0.5], "items": []})

        def _send(self, obj):
            out = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", calls
    server.shutdown()


def test_client_retries_and_sends_key(fake_hnsw):
    url, calls = fake_hnsw
    client = CandgenClient(url, api_key="secret", timeout=5)
    assert client.get("/health")["status"] == "ok"
    resp = client.post("/hnsw/search", {"vector": [0.0] * 768, "k": 2})
    assert resp["ids"] == ["a", "b"] and {k.lower(): v for k, v in calls[0][2].items()}["x-api-key"] == "secret"
    with pytest.raises(CandgenError):
        CandgenClient("http://127.0.0.1:9", timeout=0.5, retries=0).get("/health")


def test_expand_env(monkeypatch):
    monkeypatch.delenv("HNSW_URL", raising=False)
    assert expand_env("${HNSW_URL:-http://localhost:8001}") == "http://localhost:8001"
    monkeypatch.setenv("HNSW_URL", "http://10.0.0.7:8001/")
    assert expand_env("${HNSW_URL:-http://localhost:8001}") == "http://10.0.0.7:8001/"
    c = CandgenClient("${HNSW_URL}", api_key="${NO_SUCH_VAR}")
    assert c.url == "http://10.0.0.7:8001" and "X-API-Key" not in c.headers


def test_load_env(tmp_path, monkeypatch):
    from recsys.config import load_env
    env = tmp_path / ".env"
    env.write_text("# комментарий\nRECSYS_T1=abc\nRECSYS_T2 = 'q'\nBROKEN\n", encoding="utf8")
    monkeypatch.setenv("RECSYS_T2", "уже задано")
    monkeypatch.delenv("RECSYS_T1", raising=False)
    load_env(str(env))
    assert os.environ["RECSYS_T1"] == "abc" and os.environ["RECSYS_T2"] == "уже задано"
    monkeypatch.delenv("RECSYS_T1")


def test_query_text_recipe():
    from recsys.candgen.query_embedder import query_text
    assert query_text("calm  jazz\nfor   night") == "task: search result | query: calm jazz for night"


class _FakeEmbedder:
    """Вместо EmbeddingGemma: вектор трека 4 для любого текста."""

    def __init__(self, vector):
        self.vector, self.texts = vector, []

    def encode(self, text):
        self.texts.append(text)
        return self.vector.tolist()


def test_hnsw_online_encodes_new_reply(res, catalog, tmp_path):
    """Реплики нет в файлах векторов: online: true кодирует её моделью (здесь — подделка), поиск локальный."""
    v = np.random.default_rng(0).normal(size=(len(catalog), 16)).astype(np.float32)
    np.save(tmp_path / "item_vectors.npy", v)
    np.save(tmp_path / "item_ids.npy", catalog.track_ids.astype(str))
    hnsw = REGISTRY["hnsw"]({"top_k": 10, "dir": str(tmp_path), "online": True}, res)
    hnsw._encoder = _FakeEmbedder(v[4])
    ctx = res.context(Request(dialog=[Message("user", "calm jazz"), Message("user", "now faster")]))
    got = hnsw.retrieve([ctx])[0]
    assert got.ids[0] == 4 and hnsw._encoder.texts == ["now faster"]   # последняя реплика
