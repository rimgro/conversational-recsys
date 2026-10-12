"""Отбор кандидатов (recsys/selection.py) и признаки ранкера (recsys/features.py)."""
import numpy as np
import pandas as pd

from recsys.candgen import Cands, build_candgens, retrieve
from recsys.data.catalog import Catalog
from recsys.data.history import build_profile
from recsys.features import CONSTRAINTS, FeatureBuilder, _rank_lookup, constraint_violations
from recsys.schemas import Context, DialogSummary, HistoryItem, Request, UserProfile
from recsys.selection import Selector, selection_rank


def _cands(ids):
    ids = np.asarray(ids, dtype=np.int32)
    return Cands(ids, np.linspace(1, 0, len(ids), dtype=np.float32))


def _selector(catalog, weights, limit=3, filters=(), sources=("a", "b"), rrf_k=60):
    cfg = {"selection": {"limit": limit, "rrf_k": rrf_k, "weights": weights, "filters": list(filters),
                         "union_rrf_k": 60, "exclude_top_tags": 2, "min_results": 2}}
    return Selector(cfg, catalog, list(sources))


def _tiny_catalog():
    return Catalog(pd.DataFrame({
        "track_id": ["a", "b", "c", "d", "e"], "artist": ["A", "B", "C", "D", "A"],
        "tags": [["rock", "pop"], ["rock"], ["jazz"], ["rock"], ["pop"]],
        "release_year": [1985, 2001, 1987, None, 1990], "mode": [1, 1, 0, 0, 1],
        "tempo": [120.0, 60.0, 90.0, 125.0, 100.0], "energy": [0.8, 0.9, 0.2, 0.7, 0.5],
        "artist_gender": ["Female", "Male", None, None, "Female"],
    }))


def test_weighted_rrf_and_limit():
    cat = _tiny_catalog()
    lists = {"a": _cands([4, 3, 2, 1]), "b": _cands([1, 0])}
    ctx = Context(Request(), DialogSummary(), UserProfile())
    sel = _selector(cat, {"a": 1, "b": 1}).select(ctx, lists)
    # 1 — в обоих списках; 4 — первый в a; у 3 (второй в a) и 0 (второй в b) поровну — берём меньшую позицию
    assert sel.ids.tolist() == [0, 1, 4] and sel.n_union == 5
    assert np.isclose(sel.score[1], 1 / 64 + 1 / 61) and selection_rank(sel).tolist() == [3, 1, 2]
    # вес 0 и кандген без веса в отбор не идут, но в union остаются
    sel = _selector(cat, {"a": 0, "b": 1}, rrf_k=20).select(ctx, lists)
    assert sel.ids.tolist() == [0, 1] and len(sel.union) == 5


def test_query_type_is_not_used(cfg, res, requests):
    """Тип запроса модели не известен: подмена meta['query_type'] не меняет ни отбор, ни признаки, ни выдачу."""
    from dataclasses import replace
    from recsys.pipeline import Pipeline
    pipe = Pipeline(cfg, res)
    for r in requests[:5]:
        a = pipe.run(r, debug=True)
        b = pipe.run(replace(r, meta={**r.meta, "query_type": "exact"}), debug=True)
        c = pipe.run(replace(r, meta={k: v for k, v in r.meta.items() if k != "query_type"}), debug=True)
        assert a.track_ids == b.track_ids == c.track_ids
        assert (a.debug["result"].selection.ids == c.debug["result"].selection.ids).all()
    fb = pipe.features
    ctx = res.context(requests[0])
    lists = {n: got[0] for n, got in pipe.retrieve([ctx]).items()}
    assert not [c for c in fb.build(ctx, lists, pipe.selector.select(ctx, lists)).columns if "type" in c]


def test_filters_and_min_results():
    cat = _tiny_catalog()
    lists = {"a": _cands([0, 1, 2, 3, 4]), "b": _cands([])}
    weights = {"a": 1}
    s = DialogSummary(exclude_tags=["pop"], exclude_artists=["A"])
    ctx = Context(Request(shown_ids=["b"]), s, UserProfile(), banned_ids={"b"})
    sel = _selector(cat, weights, limit=5, filters=["banned", "exclude_artist", "exclude_tag"]).select(ctx, lists)
    assert sel.ids.tolist() == [2, 3] and set(sel.removed) == {"banned", "exclude_artist"}
    # «no pop»: у a pop второй тег, у e — первый; exclude_top_tags = 2 — выкидываем обоих
    ctx = Context(Request(), DialogSummary(exclude_tags=["pop"]), UserProfile())
    sel = _selector(cat, weights, limit=5, filters=["exclude_tag"]).select(ctx, lists)
    assert sel.ids.tolist() == [1, 2, 3]
    # отбор ничего не дал (вес только у пустого b), а фильтры включены — добираем из union до min_results
    sel = _selector(cat, {"b": 1}, filters=["banned"]).select(ctx, lists)
    assert len(sel.ids) == 2


def test_novelty_filter_uses_whole_history():
    """novelty: артисты всей истории, а не только последних треков профиля (build_profile берёт 300)."""
    cat = Catalog(pd.DataFrame({"track_id": ["new", "old", "x", "y"], "artist": ["New", "Old", "Old", "Fresh"]}))
    history = [HistoryItem("new", 1.0, 2)] + [HistoryItem("old", 1.0, 1)]
    profile = build_profile(history, cat, max_tracks=1)  # в профиль попал только свежий трек
    assert "Old" not in profile.artist_weights
    ctx = Context(Request(history=history), DialogSummary(new_artists=True), profile)
    sel = _selector(cat, {"a": 1}, limit=5, filters=["new_artist"]) \
        .select(ctx, {"a": _cands(cat.positions(["x", "y"])), "b": Cands.empty()})
    assert cat.track_ids[sel.ids].tolist() == ["y"]


def test_constraint_violations():
    cat = _tiny_catalog()
    s = DialogSummary(constraints={"year_min": 1980, "year_max": 2000, "mode": 1, "bpm": 120.0, "vocals": "female"})
    ctx = Context(Request(), s, UserProfile())
    # b: год, bpm (60 — половина 120, это тот же бит) и пол вокала -> 2; c: тональность и bpm; d: тональность
    # (год неизвестен — не нарушение)
    assert list(constraint_violations(cat.positions("abcd"), ctx, cat, CONSTRAINTS)) == [0, 2, 2, 1]


def test_rank_lookup():
    c = Cands(np.array([7, 3, 9], dtype=np.int32), np.zeros(3, dtype=np.float32))
    assert _rank_lookup(c, np.array([3, 5, 9, 7]), 1001).tolist() == [2, 1001, 3, 1]
    assert _rank_lookup(Cands.empty(), np.array([1]), 11).tolist() == [11]


def test_features_for_selected(cfg, res, requests):
    candgens = build_candgens(cfg, res)
    sources = [c.name for c in candgens]
    ctxs = res.contexts(requests[:5])
    lists = retrieve(candgens, ctxs, res)
    selector = Selector(cfg, res.catalog, sources)
    groups = ["text", "history", "item", "rrf", "sources", "selection", "summary"]
    fb = FeatureBuilder({**cfg, "features": {"groups": groups}}, res, sources)
    for i, ctx in enumerate(ctxs):
        own = {n: lists[n][i] for n in sources}
        sel = selector.select(ctx, own)
        f = fb.build(ctx, own, sel)
        assert len(f) == len(sel.ids) and np.isfinite(f.to_numpy()).all()
        assert {f"{s}_source_rank" for s in sources} <= set(f.columns) and "n_violations" in f
        mine = np.isin(sel.ids, ctx.user.positions)                 # свои треки: есть в истории
        assert (f.loc[mine, "history_track_log_count"] > 0).all() and (f.loc[~mine, "history_track_recency_days"] == -1).all()
        hit = f["history_source_hit"].to_numpy() == 1
        assert (hit == mine).all()


def test_tuner_matches_selector(cfg, res):
    """Подбор весов считает recall@limit так же, как отбор, и не хуже текущих весов."""
    from recsys.metrics import per_list
    from recsys.pipeline import Pipeline
    from recsys.run import recommend_part
    from recsys.tune import PolicyTuner, collect
    pipe = Pipeline(cfg, res, load_model=False)
    queries = collect(pipe, "valid")
    tuner = PolicyTuner(pipe.sources, queries, pipe.selector.limit)
    current = np.array([pipe.selector.weights.get(s, 0.0) for s in pipe.sources])
    per = pd.concat([per_list(chunk, pipe.selector) for chunk in recommend_part(pipe, "valid", rank=False)])
    sel = per[per["list"] == "selection"]
    assert np.isclose(tuner.recall(queries, current, pipe.selector.rrf_k), sel["rank"].notna().mean())
    tuned = tuner.tune(queries)
    w = np.array([float(tuned["weights"].get(s, 0.0)) for s in pipe.sources])
    assert tuned["weights"] and np.isclose(tuner.recall(queries, w, tuned["rrf_k"]), tuned["recall"])
