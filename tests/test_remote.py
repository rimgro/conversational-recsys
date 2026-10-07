"""Наша часть (DataSphere) <-> кандгены BM25 и HNSW (два сервиса на VPS) по HTTP, контракт docs/candgen_api.md."""
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from recsys.config import deep_update, load_config
from recsys.pipeline import Pipeline
from recsys.schemas import Message, Request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OVERLAY = os.path.join(ROOT, "configs", "candgen.yaml")


def _api_cfg(cfg, bm25_url, hnsw_url=None, **extra):
    base = load_config([os.path.join(ROOT, "configs", "default.yaml"), OVERLAY], {"data": cfg["data"]})
    return deep_update(base, {"candgen": {"bm25": {"url": bm25_url}, "hnsw": {"url": hnsw_url or bm25_url}, **extra}})


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fake_candgen(data):
    """Поддельный сервис по контракту (оба пути на одном порту): запоминает запросы, отвечает треками каталога.
    Первое соединение к /hnsw/search обрывается без ответа — клиент должен повторить."""
    calls = []
    dropped = []
    ids = [str(t) for t in data.catalog.track_ids[:50]]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._send({"status": "ok", "indexes": {"genres": {"index_version": "test", "n_items": 50}}})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/hnsw/search" and not dropped:
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


def test_pipeline_calls_candgen(cfg, data, fake_candgen, monkeypatch):
    url, calls = fake_candgen
    monkeypatch.setenv("BM25_API_KEY", "k123")  # ключ только у BM25: у сервисов свои настройки
    monkeypatch.delenv("HNSW_API_KEY", raising=False)
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
    path, body, headers = by_index["audio"]
    assert path == "/hnsw/search" and len(body["track_ids"]) == len(body["weights"]) > 0
    assert "X-Api-Key" not in headers
    assert set(body["track_ids"]) <= {h.track_id for h in req.history}
    assert len(resp.debug["candidates"]["hnsw_audio"]) == 49 and resp.tracks  # после повтора
    hnsw = next(r for r in pipe.retrievers if r.name == "hnsw_audio")
    assert hnsw.last_error is None
    bm25 = next(r for r in pipe.retrievers if r.name == "bm25_genres")
    assert bm25.health() == {"url": url, "status": "ok", "index_version": "test", "n_items": 50}


def test_vps_config_skips_embeddings(cfg):
    from recsys.data import load_data
    data_cfg = {k: v for k, v in cfg["data"].items() if k != "load_embeddings"}
    c = load_config([os.path.join(ROOT, "configs", "default.yaml"), OVERLAY], {"data": data_cfg})
    assert c["data"]["load_embeddings"] is False
    assert load_data(c, verbose=False).catalog.embeddings is None


def test_candgen_down_does_not_break_pipeline(cfg, data):
    port = _free_port()  # на нём никто не слушает
    pipe = Pipeline.from_config(_api_cfg(cfg, f"http://127.0.0.1:{port}", timeout=0.5), data.catalog)
    with pytest.warns(UserWarning, match="кандген недоступен"):
        resp = pipe.run(data.requests[0], debug=True)
    assert resp.tracks  # локальные relisten / title / history / popular работают
    remote = [r for r in pipe.retrievers if r.name.startswith(("bm25_", "hnsw_"))]
    assert remote and all(r.last_error for r in remote)
    assert remote[0].health()["status"] == "unavailable"


def _start(service, port, env):
    """Сервис так же, как в deploy/recsys-<service>.service: свой процесс, свой порт, свои переменные."""
    module = {"bm25": "app:app", "hnsw": "service:app"}[service]
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", module, "--host", "127.0.0.1", "--port", str(port)],
                            cwd=os.path.join(ROOT, service), env={**os.environ, **env},
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    health = f"http://127.0.0.1:{port}" + {"bm25": "/health", "hnsw": "/hnsw/health"}[service]
    for _ in range(300):
        if proc.poll() is not None:
            raise RuntimeError(f"{service} не запустился: {proc.stderr.read().decode()[-2000:]}")
        try:
            urllib.request.urlopen(health, timeout=1).close()
            return proc
        except OSError:
            time.sleep(0.05)
    proc.kill()
    raise RuntimeError(f"{service} не ответил на {health}")


@pytest.fixture
def vps(cfg, tmp_path):
    """Настоящие bm25/ и hnsw/ двумя процессами, индексы — скриптом deploy/build_indexes.sh из синтетики."""
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    import pandas as pd

    from recsys.data.synthetic import make_synthetic
    meta, _ = make_synthetic(n_tracks=1500, n_users=60, seed=cfg["data"]["synthetic"]["seed"])
    meta["m4a_tags_full"] = meta["lastfm_tag_weights"].map(lambda s: ",".join(json.loads(s)))  # как в tracks_meta
    src = tmp_path / "tracks_meta.parquet"
    meta.to_parquet(src)
    run = dict(check=True, capture_output=True)
    subprocess.run(["sh", os.path.join(ROOT, "deploy", "build_indexes.sh"), "all", str(src), str(tmp_path / "idx")],
                   env={**os.environ, "BM25_PYTHON": sys.executable, "HNSW_PYTHON": sys.executable}, **run)
    # маленький индекс с тегами через запятую, как в настоящем tracks_meta.parquet
    pd.DataFrame({"m4a_id": ["a", "b"], "m4a_tags_full": ["guitar,loud", "calm"]}).to_parquet(tmp_path / "commas.parquet")
    subprocess.run([sys.executable, os.path.join(ROOT, "bm25", "build_index.py"), "--input", str(tmp_path / "commas.parquet"),
                    "--field", "m4a_tags_full", "--out", str(tmp_path / "idx" / "bm25" / "commas")], **run)

    ports = {"bm25": _free_port(), "hnsw": _free_port()}
    procs = {}
    try:
        procs["bm25"] = _start("bm25", ports["bm25"], {"BM25_INDEXES_DIR": str(tmp_path / "idx" / "bm25"), "BM25_API_KEY": ""})
        procs["hnsw"] = _start("hnsw", ports["hnsw"], {"HNSW_INDEXES_DIR": str(tmp_path / "idx" / "hnsw"), "HNSW_API_KEY": ""})
        yield {s: f"http://127.0.0.1:{p}" for s, p in ports.items()}, procs
    finally:
        for p in procs.values():
            p.terminate()
            p.wait(timeout=10)


def test_real_services(cfg, data, vps):
    urls, procs = vps
    pipe = Pipeline.from_config(_api_cfg(cfg, urls["bm25"], urls["hnsw"]), data.catalog)
    r = next(q for q in data.requests if q.meta["query_family"] == "genre")
    resp = pipe.run(r, debug=True)
    n = resp.debug["n_candidates"]
    assert n["bm25_tags"] > 0 and n["hnsw_audio"] == 200 and n["bm25_genres"] > 0
    assert all(x.last_error is None for x in pipe.retrievers if hasattr(x, "last_error"))
    top = resp.debug["candidates"]["hnsw_audio"][0].track_id
    assert top in data.catalog  # id с VPS совпадают с каталогом нашей части (m4a_id)
    health = {x.name: x.health() for x in pipe.retrievers if getattr(x, "remote", False)}
    assert all(h["status"] == "ok" and h["index_version"] for h in health.values())

    # сервисы независимы: HNSW остановлен -> BM25 работает, hnsw_audio пропускается
    procs["hnsw"].terminate()
    procs["hnsw"].wait(timeout=10)
    with pytest.warns(UserWarning, match="hnsw_audio: кандген недоступен"):
        resp = pipe.run(r, debug=True)
    n = resp.debug["n_candidates"]
    assert n["hnsw_audio"] == 0 and n["bm25_tags"] > 0 and resp.tracks


def test_bm25_splits_comma_separated_tags(cfg, vps):
    """В tracks_meta теги одной строкой через запятую: 'guitar,loud' -> слова guitar и loud."""
    urls, _ = vps
    from recsys.retrieval.remote import CandgenClient
    resp = CandgenClient(urls["bm25"]).post("/bm25/search", {"words": ["loud"], "k": 5, "index": "commas"})
    assert resp["ids"] == ["a"]


def test_expand_env(monkeypatch):
    from recsys.retrieval.remote import CandgenClient, expand_env
    monkeypatch.delenv("BM25_URL", raising=False)
    assert expand_env("${BM25_URL:-http://localhost:8001}") == "http://localhost:8001"
    monkeypatch.setenv("BM25_URL", "http://10.0.0.7:8001/")
    assert expand_env("${BM25_URL:-http://localhost:8001}") == "http://10.0.0.7:8001/"
    c = CandgenClient("${BM25_URL}", api_key="${NO_SUCH_VAR}", headers={"x-trace": "${TRACE:-abc}", "y": "${NONE}"})
    assert c.url == "http://10.0.0.7:8001" and "X-API-Key" not in c.headers
    assert c.headers["x-trace"] == "abc" and "y" not in c.headers


def test_client_config_per_service(monkeypatch):
    from recsys.retrieval.remote import CandgenClient
    monkeypatch.setenv("HNSW_URL", "http://1.2.3.4:8002")
    cfg = load_config([os.path.join(ROOT, "configs", "default.yaml"), OVERLAY])
    bm25, hnsw = CandgenClient.from_config(cfg, "bm25"), CandgenClient.from_config(cfg, "hnsw", {"timeout": 9})
    assert (bm25.url, bm25.timeout, bm25.retries) == ("http://localhost:8001", 5.0, 1)
    assert (hnsw.url, hnsw.timeout) == ("http://1.2.3.4:8002", 9)
