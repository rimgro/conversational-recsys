import asyncio
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from recsys.serving.app import RequestGuard, create_app
from recsys.serving.artifacts import load_subset, release_id, split_bundle
from recsys.serving.client import CandidateAPIClient, CandidateAPIError, RemoteQwenRetriever
from recsys.serving.index import CandidateIndex
from serving_fixture import make_export

KEY = "synthetic-test-api-key-0123456789"
VECTOR = [1.] + [0.] * 255


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    root = tmp_path_factory.mktemp("serving")
    source = make_export(root / "source")
    split_bundle(source, root / "split")
    return root / "split"


@pytest.fixture(scope="module")
def index(release):
    return CandidateIndex(release / "right")


@pytest.fixture
def api(release, index):
    with TestClient(create_app(release / "right", KEY, index_factory=lambda _: index)) as app:
        yield app


def body(index, **kwargs):
    return {"bundle_id": index.manifest["bundle_id"], "release_id": index.manifest["release_id"],
        "vector": VECTOR, **kwargs}


def test_split_checks_full_source_and_separates_large_components(release, tmp_path):
    left, source = load_subset(release / "left", "left")
    right, _ = load_subset(release / "right", "right")
    assert left["release_id"] == right["release_id"] == release_id(source)
    assert not (release / "left/tracks.hnsw").exists()
    assert not (release / "left/item_tower.pt").exists()
    assert not (release / "right/history_features.npz").exists()
    assert not (release / "right/query_trainable.pt").exists()
    with pytest.raises(ValueError, match="exists"):
        split_bundle(release.parent / "source", release)
    copied = tmp_path / "corrupt"
    shutil.copytree(release.parent / "source", copied)
    (copied / "item_tower.pt").write_bytes(b"changed even though not copied to the server")
    with pytest.raises(ValueError, match="Source integrity"):
        split_bundle(copied, tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_manifest_traversal_and_serving_tampering_fail_closed(release, tmp_path):
    copied = tmp_path / "bad"
    shutil.copytree(release / "right", copied)
    source = json.loads((copied / "source_manifest.json").read_text())
    source["sha256"]["../outside"] = "0" * 64
    (copied / "source_manifest.json").write_text(json.dumps(source))
    with pytest.raises(ValueError, match="identity"):
        CandidateIndex(copied)
    (copied / "source_manifest.json").write_text((release / "right/source_manifest.json").read_text())
    (copied / "catalog.json").write_text('[]')
    with pytest.raises(ValueError, match="integrity"):
        CandidateIndex(copied)


def test_startup_rejects_wrong_index_vectors_even_with_matching_file_hashes(release, tmp_path):
    import hnswlib
    from recsys.serving.artifacts import sha256
    copied = tmp_path / "bad-index"
    shutil.copytree(release / "right", copied)
    vectors = np.load(copied / "track_vectors.npy")[:, ::-1].copy()
    alternate = hnswlib.Index(space="cosine", dim=256)
    alternate.init_index(max_elements=len(vectors))
    alternate.add_items(vectors, np.arange(len(vectors)), num_threads=1)
    alternate.save_index(str(copied / "tracks.hnsw"))
    source = json.loads((copied / "source_manifest.json").read_text())
    source["sha256"]["tracks.hnsw"] = sha256(copied / "tracks.hnsw")
    serving = json.loads((copied / "serving_manifest.json").read_text())
    serving["sha256"]["tracks.hnsw"] = source["sha256"]["tracks.hnsw"]
    serving["release_id"] = release_id(source)
    (copied / "source_manifest.json").write_text(json.dumps(source))
    (copied / "serving_manifest.json").write_text(json.dumps(serving))
    with pytest.raises(ValueError, match="Index vectors"):
        CandidateIndex(copied)


def test_right_api_imports_without_torch_or_transformers():
    probe = 'import sys; import recsys.serving.app; assert "torch" not in sys.modules; assert "transformers" not in sys.modules'
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_health_auth_and_release_mismatch(api, index):
    health = api.get("/health")
    assert health.status_code == 200 and health.json()["dimension"] == 256
    assert api.post("/v1/search", json=body(index)).status_code == 401
    assert api.post("/v1/search", headers={"X-API-Key": "wrong"}, json=body(index)).status_code == 401
    response = api.post("/v1/search", headers={"X-API-Key": KEY}, json=body(index, release_id="0" * 64))
    assert response.status_code == 409
    with pytest.raises(ValueError, match="API key"):
        create_app("unused", "short")


@pytest.mark.parametrize("patch", [{"vector": [0.] * 256}, {"vector": VECTOR[:-1]},
    {"vector": [True] + VECTOR[1:]}, {"vector": ["1"] + VECTOR[1:]},
    {"k": 201}, {"k": True}, {"query": "raw text is not accepted"},
    {"discovery": "true"}, {"history": [123]}, {"mode": "other"}])
def test_invalid_contract(api, index, patch):
    result = api.post("/v1/search", headers={"X-API-Key": KEY}, json=body(index, **patch))
    assert result.status_code == 422


def test_nonfinite_and_oversized_input(api, index):
    payload = body(index, vector=[float("nan")] + VECTOR[1:])
    result = api.post("/v1/search", headers={"X-API-Key": KEY, "Content-Type": "application/json"},
        content=json.dumps(payload))
    assert result.status_code == 422 and "NaN" not in result.text
    result = api.post("/v1/search", headers={"X-API-Key": KEY}, content=b" " * (4 * 1024**2 + 1))
    assert result.status_code == 413


def test_chunked_request_limit_before_json_parsing():
    async def run():
        messages = iter([{"type": "http.request", "body": b"1234", "more_body": True},
                         {"type": "http.request", "body": b"5678", "more_body": False}])
        sent = []
        async def receive(): return next(messages)
        async def send(message): sent.append(message)
        async def unexpected(*args): pytest.fail("Oversized body reached the application")
        await RequestGuard(unexpected, KEY, max_body_bytes=7)(
            {"type": "http", "method": "POST", "path": "/v1/search", "headers": [(b"x-api-key", KEY.encode())]},
            receive, send)
        assert sent[0]["status"] == 413
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["exact", "hnsw"])
def test_discovery_filters_full_history_and_fills_200(api, index, mode):
    history = [str(i * 2) for i in range(450)]  # Older than the left tower's 256-event window.
    result = api.post("/v1/search", headers={"X-API-Key": KEY},
        json=body(index, history=history, discovery=True, mode=mode)).json()
    assert result["returned"] == 200
    keys = [int(t) // 2 for t in result["ids"]]
    assert len(set(keys)) == 200 and min(keys) >= 450
    assert all(a >= b for a, b in zip(result["scores"], result["scores"][1:]))


def test_reference_alias_artist_and_banned_ids(api, index):
    result = api.post("/v1/search", headers={"X-API-Key": KEY}, json=body(index, mode="exact",
        query_type="similar_to", exclude_m4a_id="0", exclude_artist="ARTIST0", exclude_ids=["2"])).json()
    assert result["returned"] == 200
    groups = [int(t) // 2 for t in result["ids"]]
    assert 0 not in groups and 1 not in groups and all(g % 5 != 0 for g in groups)
    result = api.post("/v1/search", headers={"X-API-Key": KEY}, json=body(index,
        discovery=True, history=index.ids)).json()
    assert result["returned"] == 0 and not result["ids"]


def test_client_and_remote_left_preserve_full_history(api, index):
    def dispatch(request):
        return api.request(request.method, request.url.path, content=request.content,
            headers=dict(request.headers))
    client = CandidateAPIClient("http://testserver", KEY, index.manifest["bundle_id"],
        index.manifest["release_id"], transport=httpx.MockTransport(dispatch))
    class Encoder:
        serving_manifest = index.manifest
        def encode_query(self, query, history):
            self.history = history
            return np.array(VECTOR, dtype=np.float32)
    encoder = Encoder()
    history = [str(i * 2) for i in range(450)]
    with RemoteQwenRetriever(encoder, client) as remote:
        result = remote.search("query", history, discovery=True)
    assert encoder.history == history and result["returned"] == 200
    assert min(int(t) // 2 for t in result["ids"]) >= 450


@pytest.mark.parametrize("kind", ["wrong-release", "wrong-length", "unordered", "timeout", "redirect", "huge"])
def test_client_rejects_bad_peer(index, kind):
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json=index.health())
        if kind == "timeout": raise httpx.ReadTimeout("secret must not be echoed")
        if kind == "redirect": return httpx.Response(307, headers={"Location": "http://other.example"})
        if kind == "huge": return httpx.Response(200, content=b" " * (1024**2 + 1))
        result = index.search(VECTOR, k=2)
        result["elapsed_ms"] = 1.
        if kind == "wrong-release": result["release_id"] = "0" * 64
        if kind == "wrong-length": result["returned"] = 1
        if kind == "unordered":
            result["candidates"].reverse();result["ids"].reverse();result["scores"].reverse()
        return httpx.Response(200, json=result)
    with CandidateAPIClient("http://example", KEY, index.manifest["bundle_id"],
            index.manifest["release_id"], transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(CandidateAPIError) as error:
            client.search(VECTOR, k=2)
        assert KEY not in str(error.value) and "secret" not in str(error.value)


def test_cli_uses_remote_query_bundle_and_options(monkeypatch, tmp_path, capsys):
    from recsys.serving import client
    script = Path(__file__).resolve().parents[1] / "tools/predict_qwen.py"
    spec = importlib.util.spec_from_file_location("remote_cli", script)
    cli = importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    received = {}
    class Remote:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def search(self, query, **kwargs): received.update(kwargs);return {"returned": 0}
    def factory(*args): received['key'] = args[2];return Remote()
    monkeypatch.setattr(client.RemoteQwenRetriever, "from_bundle", factory)
    monkeypatch.setenv("QWEN_API_KEY", KEY)
    monkeypatch.setattr(sys, "argv", ["predict", "--bundle", str(tmp_path), "--query", "query",
        "--api-url", "https://example", "--exact", "--discovery", "--device", "cpu"])
    cli.main()
    assert received['mode'] == "exact" and received['discovery'] and received['k'] == 200
    assert received['key'] == KEY and json.loads(capsys.readouterr().out)['returned'] == 0
