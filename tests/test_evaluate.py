"""evaluate.py: прогон валидации пишет метрики и не начинается без сервисов."""
import json
import os
import socket
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import evaluate  # noqa: E402


def test_evaluate_writes_results(tmp_path):
    out = tmp_path / "run"
    code = evaluate.main(["--synthetic", "--n-users", "20", "--max-queries", "2", "--run-dir", str(out),
                          "--chunk", "15", "--set", "ranker.type=heuristic"])
    assert code == 0
    assert {p.name for p in out.iterdir()} == {"metrics.json", "submission.parquet", "by_query_type.csv", "by_is_new.csv",
                                                "per_request.csv", "config.yaml", "by_source.csv", "by_source_query_type.csv"}
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert m["n_users"] == 20 and 20 <= m["n_requests"] <= 40 and m["ranker"] == "heuristic"  # до 2 на пользователя
    assert 0 <= m["metrics"]["ndcg@20"] <= 1
    import pandas as pd
    src = pd.read_csv(out / "by_source.csv", index_col=0)
    assert {"recall", "only_this", "median_rank", "mean_candidates"} <= set(src.columns) and "relisten" in src.index
    assert (src["only_this"].dropna() <= src["recall"].dropna().reindex(src["only_this"].dropna().index)).all()
    sub = pd.read_parquet(out / "submission.parquet")
    assert list(sub.columns) == ["query_id", "top20", "response"] and len(sub) == m["n_requests"]
    assert all(len(t) == 20 for t in sub["top20"]) and sub["response"].str.startswith("Hi!").all()
    lines = (out / "per_request.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == m["n_requests"] + 1 and lines[0].startswith("request_id")  # заголовок один, хотя писали частями


def test_evaluate_stops_when_candgen_down(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{s.getsockname()[1]}"  # никто не слушает
    code = evaluate.main(["--synthetic", "--n-users", "5", "--candgen", "--run-dir", str(tmp_path / "run"),
                          "--set", f"candgen.bm25.url={url}", "--set", f"candgen.hnsw.url={url}",
                          "--set", "candgen.timeout=0.5"])
    assert code == 2 and not (tmp_path / "run").exists()
