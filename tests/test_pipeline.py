from recsys.config import deep_update
from recsys.eval import evaluate
from recsys.llm import StubLLM
from recsys.pipeline import Pipeline
from recsys.schemas import Message, Request


def test_pipeline_runs_and_respects_filters(cfg, data):
    pipe = Pipeline.from_config(deep_update(cfg, {"fusion": {"exclude_listened": True}}), data.catalog)
    req = data.requests[0]
    resp = pipe.run(req, debug=True)
    assert 0 < len(resp.tracks) <= cfg["ranker"]["top_k"]
    assert resp.text
    heard = {h.track_id for h in req.history}
    assert not heard & set(resp.track_ids)
    for t in resp.tracks:
        tags = data.catalog.tags(t.track_id)
        assert not any(ex in tags for ex in resp.summary.exclude_tags)


def test_next_turn_excludes_shown(cfg, data):
    pipe = Pipeline.from_config(cfg, data.catalog)
    r1 = pipe.run(data.requests[1])
    r2 = pipe.run(data.requests[1].next_turn(r1, "more please"))
    assert not set(r1.track_ids) & set(r2.track_ids)


def test_cold_start_empty_request(cfg, data):
    resp = Pipeline.from_config(cfg, data.catalog).run(Request())
    assert len(resp.tracks) > 0


def test_llm_stub_falls_back(cfg, data):
    c = deep_update(cfg, {"summarizer": {"type": "llm"}, "explainer": {"type": "llm"}})
    pipe = Pipeline.from_config(c, data.catalog, llm=StubLLM(""))
    resp = pipe.run(Request(dialog=[Message("user", "some calm jazz")]))
    assert resp.summary.source == "rule" and resp.text.startswith("Привет!")


def test_evaluate(cfg, data):
    pipe = Pipeline.from_config(deep_update(cfg, {"ranker": {"type": "heuristic"}}), data.catalog)
    per_req, mean = evaluate(pipe, data.requests[:20], verbose=False)
    assert len(per_req) > 0 and 0.0 <= mean["hit@10"] <= 1.0


def test_api_ranker_and_fallback(cfg, data):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            # обратный порядок: чем ниже RRF, тем выше скор
            scores = [-f["rrf_score"] for f in body["features"]]
            out = json.dumps({"scores": scores}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/rank"
        req = data.requests[0]
        stub = Pipeline.from_config(cfg, data.catalog).run(req, debug=True)
        api_cfg = deep_update(cfg, {"ranker": {"type": "api", "api_url": url}, "diversity": {"max_per_artist": 0}})
        api = Pipeline.from_config(api_cfg, data.catalog).run(req, debug=True)
        # API вернул обратный порядок RRF (допуск: JSON передаёт до 15 знаков)
        assert (api.debug["features"]["rrf_score"].diff().dropna() > -1e-12).all()
        assert set(api.track_ids) != set(stub.track_ids)
    finally:
        server.shutdown()

    # сервер недоступен -> ранкер-заглушка
    down_cfg = deep_update(cfg, {"ranker": {"type": "api", "api_url": "http://127.0.0.1:9/rank", "timeout": 0.5}})
    pipe = Pipeline.from_config(down_cfg, data.catalog)
    resp = pipe.run(data.requests[0])
    assert len(resp.tracks) > 0 and pipe.ranker.last_error


def test_metrics_by_family_and_shared_index(cfg, data):
    from recsys.eval import compare_configs, metrics_by
    pipe = Pipeline.from_config(cfg, data.catalog)
    per, _ = evaluate(pipe, data.requests[:30], verbose=False)
    by = metrics_by(per, "query_family")
    assert by["n"].sum() == 30 and set(by.index) <= {r.meta["query_family"] for r in data.requests}
    # без bm25 общий индекс берётся у relisten, а не у title/lyrics
    no_bm25 = Pipeline.from_config(deep_update(cfg, {"retrieval": {"bm25": {"enabled": False}}}), data.catalog)
    assert no_bm25.bm25_index is no_bm25.retrievers[0].index and no_bm25.retrievers[0].name == "relisten"
    table = compare_configs(cfg, {"a": {}, "b": {"ranker": {"type": "heuristic"}}}, data.catalog,
                            data.requests[:20], by="query_family")
    assert list(table.columns) == ["a", "b"] and "ALL" in table.index
