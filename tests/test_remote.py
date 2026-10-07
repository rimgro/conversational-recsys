"""Наша часть (образ 2) <-> кандгены (образ 1) по HTTP."""
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from recsys.config import deep_update, load_config
from recsys.pipeline import Pipeline
from recsys.schemas import Message, Request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OVERLAY = os.path.join(ROOT, "configs", "candgen.yaml")


def _api_cfg(cfg, url, **extra):
    base = load_config([os.path.join(ROOT, "configs", "default.yaml"), OVERLAY], {"data": cfg["data"]})
    return deep_update(base, {"candgen": {"url": url, **extra}})


@pytest.fixture
def fake_candgen(data):
    """Поддельный образ 1: запоминает запросы, отвечает треками каталога."""
    calls = []
    ids = [str(t) for t in data.catalog.track_ids[:50]]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, body, dict(self.headers)))
            out = [t for t in ids if t not in set(body["exclude_ids"])][:body["k"]]
            resp = {"ids": out, "scores": [1.0 / (i + 1) for i in range(len(out))], "index": body["index"],
                    "index_version": "test"}
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


def test_pipeline_calls_candgen(cfg, data, fake_candgen, monkeypatch):
    url, calls = fake_candgen
    monkeypatch.setenv("CANDGEN_API_KEY", "k123")
    pipe = Pipeline.from_config(_api_cfg(cfg, url), data.catalog)
    assert {r.name for r in pipe.retrievers} >= {"bm25_genres", "bm25_tags", "bm25_title", "hnsw_audio"}
    assert "bm25" not in {r.name for r in pipe.retrievers}
    req = Request(dialog=[Message("user", "спокойный джаз, но не хард-рок")], history=data.requests[0].history,
                  shown_ids=[str(data.catalog.track_ids[0])])
    resp = pipe.run(req, debug=True)

    by_index = {b["index"]: (path, b, h) for path, b, h in calls}
    path, body, headers = by_index["genres"]
    assert path == "/bm25/search" and "jazz" in body["words"] and "hard" not in body["words"]
    assert body["exclude_ids"] == [str(data.catalog.track_ids[0])]
    assert headers.get("X-Api-Key") == "k123"
    path, body, _ = by_index["audio"]
    assert path == "/hnsw/search" and len(body["track_ids"]) == len(body["weights"]) > 0
    assert set(body["track_ids"]) <= {h.track_id for h in req.history}
    assert len(resp.debug["candidates"]["hnsw_audio"]) == 49 and resp.tracks


def test_candgen_down_does_not_break_pipeline(cfg, data):
    with socket.socket() as s:  # свободный порт, на котором никто не слушает
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    pipe = Pipeline.from_config(_api_cfg(cfg, f"http://127.0.0.1:{port}", timeout=0.5), data.catalog)
    with pytest.warns(UserWarning, match="кандген недоступен"):
        resp = pipe.run(data.requests[0], debug=True)
    assert resp.tracks  # локальные relisten / title / history / popular работают
    remote = [r for r in pipe.retrievers if r.name.startswith(("bm25_", "hnsw_"))]
    assert remote and all(r.last_error for r in remote)


def test_real_candgen_service(cfg, data, tmp_path, monkeypatch):
    """Настоящий candgen (bm25/ + hnsw/) на uvicorn, индексы из синтетики теми же скриптами, что в образе."""
    pytest.importorskip("fastapi")
    uvicorn = pytest.importorskip("uvicorn")
    import importlib
    import subprocess
    import sys

    from recsys.data.synthetic import make_synthetic
    meta, splits = make_synthetic(n_tracks=1500, n_users=60, seed=cfg["data"]["synthetic"]["seed"])
    meta["m4a_tags_full"] = meta["lastfm_tag_weights"].map(lambda s: " ".join(json.loads(s)))  # без запятых: см. candgen/RUN.md
    src = tmp_path / "tracks_meta.parquet"
    meta.to_parquet(src)
    subprocess.run(["sh", os.path.join(ROOT, "candgen", "build_indexes.sh"), str(src), str(tmp_path / "idx")],
                   check=True, capture_output=True, env={**os.environ, "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]})
    monkeypatch.setenv("BM25_INDEXES_DIR", str(tmp_path / "idx" / "bm25"))
    monkeypatch.setenv("HNSW_INDEXES_DIR", str(tmp_path / "idx" / "hnsw"))
    monkeypatch.setenv("BM25_API_KEY", "")
    monkeypatch.syspath_prepend(os.path.join(ROOT, "candgen"))
    main = importlib.import_module("main")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    try:
        c = _api_cfg(cfg, f"http://127.0.0.1:{port}")
        pipe = Pipeline.from_config(c, data.catalog)
        r = next(q for q in data.requests if q.meta["query_family"] == "genre")
        resp = pipe.run(r, debug=True)
        n = resp.debug["n_candidates"]
        assert n["bm25_tags"] > 0 and n["hnsw_audio"] == 200 and n["bm25_genres"] > 0
        assert all(x.last_error is None for x in pipe.retrievers if hasattr(x, "last_error"))
        top = resp.debug["candidates"]["hnsw_audio"][0].track_id
        assert top in data.catalog  # id из образа 1 совпадают с каталогом образа 2 (m4a_id)
    finally:
        server.should_exit = True


def test_expand_env(monkeypatch):
    from recsys.retrieval.remote import CandgenClient, expand_env
    monkeypatch.delenv("CANDGEN_URL", raising=False)
    assert expand_env("${CANDGEN_URL:-http://localhost:8000}") == "http://localhost:8000"
    monkeypatch.setenv("CANDGEN_URL", "http://node:9000/")
    assert expand_env("${CANDGEN_URL:-http://localhost:8000}") == "http://node:9000/"
    c = CandgenClient("${CANDGEN_URL}", api_key="${NO_SUCH_VAR}", headers={"x-node-id": "${NODE:-abc}", "y": "${NONE}"})
    assert c.url == "http://node:9000" and "X-API-Key" not in c.headers
    assert c.headers["x-node-id"] == "abc" and "y" not in c.headers
