"""LGBMRanker (ranker.type: lgbm) с моделью из ranker_assets/v3."""
import os
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from recsys.config import deep_update  # noqa: E402
from recsys.pipeline import Pipeline  # noqa: E402
from recsys.ranker_lgbm import LGBMRanker, age_bucket, query_ngrams  # noqa: E402
from recsys.ranking import MISSING_RANK, build_ranker  # noqa: E402
from recsys.schemas import Context, DialogSummary, HistoryItem, Message, Request, UserProfile  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "ranker_assets", "v3")


@pytest.fixture(scope="module")
def ranker():
    return LGBMRanker(MODEL_DIR)


def _ctx(query, history=(), attrs=None):
    req = Request(dialog=[Message("user", query)], history=list(history), user_attrs=attrs or {})
    return Context(req, DialogSummary(), UserProfile())


def _pool(track_ids):
    n = len(track_ids)
    return pd.DataFrame({"track_id": track_ids, "rrf_score": np.linspace(0.03, 0.01, n), "n_sources": [2] * n,
                         "rank_hnsw": range(1, n + 1), "score_hnsw": np.linspace(0.8, 0.6, n),
                         "rank_bm25_cards": [1] + [MISSING_RANK] * (n - 1), "score_bm25_cards": [9.0] + [0.0] * (n - 1)})


def test_helpers():
    assert [age_bucket(a) for a in (None, 5, 17, 33, 64, 150)] == ["unknown", "unknown", "10s", "30s", "50+", "unknown"]
    assert query_ngrams("Some Hard Rock, please", {"hard rock", "rock", "jazz"}) == {"hard rock", "rock"}


def test_features_from_history(ranker):
    ids = ranker.tracks.index[:30].tolist()
    heard, artist = ids[0], ranker.tracks.loc[ids[0], "artist"]
    ctx = _ctx("some rock", [HistoryItem(heard, count=5)], {"age": 25, "gender": "f"})
    x = ranker.compute_features(_pool(ids), ctx)
    assert list(x.columns) == ranker.features
    assert x.loc[0, "in_history"] == 1 and x.loc[1:, "in_history"].sum() == 0
    assert x.loc[0, "log_history_count"] == pytest.approx(np.log1p(5))
    assert x.loc[0, "artist_share"] == pytest.approx(1.0)       # вся история — этот артист
    assert np.isnan(x.loc[1, "score_bm25"]) and x.loc[0, "score_bm25"] == 9.0   # не найден источником -> NaN
    assert set(x["age_bucket"].astype(str)) == {"20s"} and set(x["gender"].astype(str)) == {"f"}
    same_artist = [i for i, t in enumerate(ids) if ranker.tracks.loc[t, "artist"] == artist]
    assert (x.loc[same_artist, "artist_share"] == 1.0).all()


def test_rank_sorts_and_falls_back(ranker):
    ids = ranker.tracks.index[:50].tolist()
    ranked = ranker.rank(_pool(ids), _ctx("calm jazz"))
    assert len(ranked) == 50 and ranked["rank_score"].is_monotonic_decreasing
    broken = _pool(ids).drop(columns=["rrf_score"]).assign(rrf_score=None).drop(columns=["n_sources"])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        out = ranker.rank(broken.assign(rrf_score=np.linspace(1, 0, 50)), _ctx("calm jazz"))
    assert ranker.last_error and list(out["track_id"]) == ids        # порядок RRF


def test_pipeline_with_lgbm_ranker(cfg, data):
    c = deep_update(cfg, {"ranker": {"type": "lgbm", "model_dir": MODEL_DIR}})
    assert isinstance(build_ranker(c), LGBMRanker)
    resp = Pipeline.from_config(c, data.catalog).run(data.requests[0])
    assert 0 < len(resp.tracks) <= c["ranker"]["top_k"]
