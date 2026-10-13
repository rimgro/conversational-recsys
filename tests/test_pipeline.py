"""Пайплайн целиком: одна реплика (диалог) и прогон частей датасета (python -m recsys)."""
import json

import numpy as np

import pandas as pd
import pytest

from recsys.__main__ import apply_sources, main
from recsys.config import deep_update
from recsys.llm import StubLLM
from recsys.pipeline import Pipeline
from recsys.resources import Resources
from recsys.schemas import Message, Request


@pytest.fixture(scope="module")
def pipe(cfg, res):
    return Pipeline(cfg, res)


def test_run_one_request(pipe, cfg, requests):
    resp = pipe.run(requests[0], debug=True)
    assert 0 < len(resp.tracks) <= cfg["ranker"]["top_k"] and resp.text
    assert [t.rank for t in resp.tracks] == list(range(1, len(resp.tracks) + 1))
    assert resp.debug["n_selected"] <= cfg["selection"]["limit"] and set(resp.debug["timings"]) >= {"2_candgen"}
    assert resp.to_frame()["track_id"].tolist() == resp.track_ids


def test_next_turn_excludes_shown(pipe, requests):
    r1 = pipe.run(requests[1])
    r2 = pipe.run(requests[1].next_turn(r1, "more please"))
    assert r2.tracks and not set(r1.track_ids) & set(r2.track_ids)


def test_cold_start_empty_request(pipe):
    """Ни истории, ни запроса: кандидатов нет, но пайплайн отвечает (в датасете такого не бывает)."""
    resp = pipe.run(Request())
    assert resp.tracks == [] and resp.text


def test_filters_in_dialog(cfg, res, requests):
    c = deep_update(cfg, {"selection": {"filters": ["banned", "exclude_artist", "new_artist", "exclude_tag"]}})
    req = requests[0]
    resp = Pipeline(c, res).run(Request(dialog=[Message("user", "some jazz, but no piano")], history=req.history))
    assert resp.tracks
    for t in resp.tracks:
        assert "piano" not in res.catalog.tags(t.track_id)[:3]


def test_llm_stub_falls_back(cfg, res):
    c = deep_update(cfg, {"summarizer": {"type": "llm"}, "explainer": {"type": "llm"}})
    pipe = Pipeline(c, Resources(c, dataset=res.dataset, llm=StubLLM(""), verbose=False))
    resp = pipe.run(Request(dialog=[Message("user", "some calm jazz")]))
    assert resp.summary.source == "rule" and resp.text.startswith("Hi!")


def test_template_response(pipe, requests):
    resp = pipe.run(requests[2])
    assert resp.text.startswith("I recommend") and resp.tracks[0].title in resp.text and len(resp.text.split()) <= 120


def test_apply_sources(cfg):
    c = deep_update(cfg, {})
    apply_sources(c, "+hnsw,-bm25")
    assert c["candgen"]["sources"]["hnsw"]["enabled"] and not c["candgen"]["sources"]["bm25"]["enabled"]
    apply_sources(c, "history,bm25")
    assert {n for n, s in c["candgen"]["sources"].items() if s["enabled"]} == {"history", "bm25"}
    with pytest.raises(SystemExit):
        apply_sources(c, "+nope")


def test_cli_train_evaluate_submit(cfg, res, tmp_path):
    """python -m recsys run / submit на синтетике: ранкер учится, отчёты и сабмит пишутся."""
    common = ["--synthetic", "--run", "t", "--set", f"run.dir={tmp_path}", "--set",
              f"data.cache_dir={cfg['data']['cache_dir']}", "--set", "data.synthetic={n_tracks: 1500, n_users: 60}",
              "--set", "ranker.catboost.iterations=20"]
    assert main(["run", "--parts", "test"] + common) == 0
    out = tmp_path / "t"
    assert (out / "ranker" / "model.cbm").exists() and (out / "ranker" / "importance.csv").exists()
    report = (out / "test" / "report.md").read_text(encoding="utf8")
    assert "Ранкер (catboost) nDCG@20" in report and "| ranker |" in report and "| union |" in report
    m = json.loads((out / "test" / "metrics.json").read_text(encoding="utf8"))
    assert 0 <= m["ndcg@20"] <= 1 and m["recall@150"] <= m["recall_union"]
    lists = pd.read_csv(out / "test" / "lists.csv", index_col=0)
    assert {"bm25", "history", "union", "selection", "ranker"} <= set(lists.index)
    assert main(["submit"] + common) == 0
    sub = pd.read_parquet(out / "submission_test_public.parquet")
    assert list(sub.columns) == ["query_id", "top20", "response"] and sub["query_id"].is_unique
    assert sub["top20"].map(lambda x: len(set(x)) == len(x)).all() and sub["response"].str.len().gt(0).all()
    assert sorted(sub["query_id"]) == sorted(res.dataset.part("submit")["query_id"])


def test_features_cached_for_training_and_ranking(cfg, res):
    """Признаки сохраняются блоками на все запросы части: обучение и ранжирование берут их с диска, те же числа."""
    from recsys import run
    pipe = Pipeline(cfg, res, load_model=False)
    x1, y1, g1, s1 = run.training_data(pipe, "valid")
    x2, y2, g2, s2 = run.training_data(pipe, "valid")
    assert s1 == s2 and (y1 == y2).all() and (g1 == g2).all() and x1.equals(x2)
    assert y1.sum() == s1["groups"] and (pd.Series(g1).diff().dropna() >= 0).all()   # по цели на группу, подряд
    files = list((__import__("pathlib").Path(cfg["data"]["cache_dir"]) / "features" / "valid").glob("*.parquet"))
    assert files
    ranked = Pipeline(cfg, res)                     # stub-ранкер; признаки — из того же кэша
    chunk = next(run.recommend_part(ranked, "valid", rank=True))
    for r in chunk[:5]:
        direct = ranked.rank(r.ctx, r.lists, r.selection)
        assert (direct.top == r.top).all() and np.allclose(direct.scores, r.scores)
