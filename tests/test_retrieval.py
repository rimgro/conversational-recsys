"""Кандгены (recsys/candgen/): каждый возвращает уникальные позиции по убыванию скора, не больше top_k."""
import numpy as np
import pandas as pd
import pytest

from recsys.candgen import REGISTRY, Cands, build_candgens, retrieve, top_positions
from recsys.candgen.card import CardIndex, build_card_index, query_terms
from recsys.data import build_profile
from recsys.dialog import RuleSummarizer
from recsys.schemas import HistoryItem, Message, Request


def _ctx(res, text, history=(), **kw):
    req = Request(dialog=[Message("user", text)], history=list(history), **kw)
    return res.context(req)


def test_top_positions_ties_and_positive():
    s = np.array([0.0, 2.0, 1.0, 2.0, -1.0, 1.0], dtype=np.float32)
    c = top_positions(s, 3)
    assert c.ids.tolist() == [1, 3, 2] and c.scores.tolist() == [2.0, 2.0, 1.0]   # равные — меньшая позиция раньше
    assert top_positions(s, 10).ids.tolist() == [1, 3, 2, 5]                         # только скор > 0
    assert top_positions(s, 10, positive=False).ids.tolist() == [1, 3, 2, 5, 0, 4]


def test_all_candgens_return_sorted_unique(cfg, res, requests):
    candgens = build_candgens(cfg, res)
    names = {c.name for c in candgens}
    assert names == {n for n, c in cfg["candgen"]["sources"].items() if c.get("enabled")}
    assert not {"hnsw", "twotower"} & names and names <= set(REGISTRY)
    ctxs = res.contexts(requests[:20])
    out = retrieve(candgens, ctxs, res)
    for name, lists in out.items():
        assert len(lists) == len(ctxs)
        for c in lists:
            assert len(c) <= candgens[[g.name for g in candgens].index(name)].top_k
            assert len(np.unique(c.ids)) == len(c.ids) and (np.diff(c.scores) <= 1e-6).all(), name
    # история: только свои треки; favorite_artists: только артисты истории
    for ctx, c in zip(ctxs, out["history"]):
        assert set(c.ids) == set(ctx.user.positions)
    hist_artists = {res.users.artists[p] for p in ctxs[0].user.positions}
    assert {res.users.artists[p] for p in out["favorite_artists"][0].ids} <= hist_artists


def test_candidates_cache(cfg, res, requests):
    candgens = build_candgens(cfg, res)[:2]
    ctxs = res.contexts(requests[:10])
    first = retrieve(candgens, ctxs, res, part="test_cache")
    again = retrieve(candgens, ctxs, res, part="test_cache")
    for name in first:
        assert all((a.ids == b.ids).all() and np.allclose(a.scores, b.scores) for a, b in zip(first[name], again[name]))


def test_relisten_ranks_own_tracks_by_query(cfg, res, catalog):
    jazz = catalog.df.index[catalog.df["genres"].map(lambda g: g == ["jazz"])][:3]
    metal = catalog.df.index[catalog.df["genres"].map(lambda g: g == ["metal"])][:3]
    history = [HistoryItem(catalog.df.at[i, "track_id"], count=1, timestamp=100 - n)
               for n, i in enumerate(list(metal) + list(jazz))]
    relisten = REGISTRY["relisten"]({"top_k": 10}, res)
    out = relisten.retrieve([_ctx(res, "jazz", history)])[0]
    assert set(out.ids) == set(catalog.positions(h.track_id for h in history))
    assert set(out.ids[:3]) == set(jazz)


def test_query_terms_use_country_and_decade(res):
    """Запрос к карточке: страна и эпоха — отдельными токенами."""
    assert {"c:us", "t:80s"} <= set(query_terms(_ctx(res, "american rock from the 80s").summary))


def test_audio_seed_uses_reference_track(res, catalog, requests):
    """similar_to: похожие по звучанию на трек-образец; без образца — пусто."""
    row = catalog.df.iloc[3]
    seed = REGISTRY["audio_seed"]({"top_k": 50}, res)
    ctx = _ctx(res, f"tracks like {row['title'].lower()} by {row['artist'].lower()} but from other artists",
               requests[0].history)
    got = seed.retrieve([ctx])[0]
    with np.errstate(all="ignore"):  # numpy + Accelerate (macOS): ложные предупреждения matmul
        sims = catalog.embeddings @ catalog.embeddings[3]
    assert got.ids[0] == 3 and set(got.ids[:5]) <= set(np.argsort(-sims)[:50])
    assert len(seed.retrieve([_ctx(res, "calm jazz", requests[0].history)])[0]) == 0


def test_card_index_fields_and_cache(res, catalog, tmp_path):
    """Поля карточки складываются с весами; сохранённый индекс даёт те же скоры."""
    row = catalog.df.iloc[5]
    meta = pd.DataFrame({"m4a_id": [row["track_id"]], "lyrics": ["purple submarine dancing on the moon"]})
    index = build_card_index(catalog, meta)
    assert set(index.fields) == {"meta", "lyrics"}  # остальных колонок CARD_FIELDS в meta нет
    q = query_terms(_ctx(res, "purple submarine").summary)
    lyr = index.scores(q, {"lyrics": 1.0})
    assert lyr.argmax() == 5 and (lyr > 0).sum() == 1
    assert not index.scores(q, {"meta": 1.0})[5] and not index.scores(q, {"lyrics": 0.0}).any()  # вес 0 — поля нет
    path = str(tmp_path / "idx.npz")
    index.save(path)
    again = CardIndex.load(path)
    assert np.allclose(again.scores(q), index.scores(q)) and set(again.fields) == set(index.fields)


def test_text_candgen_finds_exact_title(res, catalog):
    row = catalog.df.iloc[7]
    ctx = _ctx(res, f"play {row['title']} by {row['artist']}")
    for name in ("bm25", "metadata_char"):
        got = REGISTRY[name]({"top_k": 10}, res).retrieve([ctx])[0]
        assert 7 in got.ids[:3], name


def test_dense_reads_vectors(cfg, res, catalog, tmp_path):
    ids = np.array([str(t) for t in catalog.track_ids[:30]])
    rng = np.random.default_rng(0)
    v = rng.normal(size=(30, 8)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    np.save(tmp_path / "item_vectors.npy", v)
    np.save(tmp_path / "item_ids.npy", ids)
    np.savez(tmp_path / "query_vectors_test.npz", query_id=np.array(["q1"]), vectors=v[[4]].astype(np.float16))
    for name in ("twotower_lazy", "twotower"):   # lazy-башня — те же файлы векторов
        dense = REGISTRY[name]({"top_k": 5, "dir": str(tmp_path)}, res)
        got = dense.retrieve([_ctx(res, "x", request_id="q1")])[0]
        assert got.ids[0] == 4 and len(got) == 5 and set(got.ids) <= set(range(30))   # треки без вектора не берём
    with pytest.warns(UserWarning, match="нет вектора запроса"):
        assert len(dense.retrieve([_ctx(res, "x", request_id="нет")])[0]) == 0
    assert dense.n_missing == 1
    ok, missing = _ctx(res, "x", request_id="q1"), _ctx(res, "x", request_id="нет")
    dense.check_coverage([ok], "test")
    with pytest.raises(ValueError, match="нет векторов для 1 из 2"):
        dense.check_coverage([ok, missing], "test")
    with pytest.raises(FileNotFoundError, match="encode_hnsw"):
        REGISTRY["hnsw"]({"dir": str(tmp_path / "нет")}, res)


def test_cands_empty():
    assert len(Cands.empty()) == 0


def test_profile_and_summary_for_dialog(res, catalog):
    """Реплика без пользователя датасета: разбор и профиль считаются, состояние — по истории запроса."""
    hist = [HistoryItem(catalog.df.at[0, "track_id"], 2.0, 10)]
    ctx = _ctx(res, "calm jazz", hist)
    assert ctx.user.cutoff == 11 and ctx.summary.include_tags
    assert build_profile(hist, catalog).n_known_tracks == 1
    assert RuleSummarizer(catalog).summarize(Request(dialog=[Message("user", "jazz")])).include_tags == ["jazz"]


def test_precomputed_lists(res, catalog, tmp_path):
    """Готовые списки: по query_id, в порядке файла, без треков не из каталога; нет списка — ошибка покрытия."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    ids = [str(t) for t in catalog.track_ids[:5]]
    pq.write_table(pa.table({"query_id": ["q1"], "top200": [[ids[3], "нет_такого", ids[1], ids[3]]],
                             "scores": [[0.9, 0.8, 0.7, 0.6]]}), tmp_path / "x_top200.parquet")
    pc = REGISTRY["twotower_pavel"]({"top_k": 200, "dir": str(tmp_path)}, res)
    got = pc.retrieve([_ctx(res, "x", request_id="q1")])[0]
    assert got.ids.tolist() == [3, 1] and np.allclose(got.scores, [0.9, 0.7])
    pc.check_coverage([_ctx(res, "x", request_id="q1")], "test")
    with pytest.raises(ValueError, match="нет списков для 1 из 2"):
        pc.check_coverage([_ctx(res, "x", request_id="q1"), _ctx(res, "x", request_id="q2")], "test")


def test_best_config():
    """configs/best.yaml: без предупреждений, все кандгены есть, голосуют только включённые, башни — признаки."""
    import os
    import warnings
    from recsys.candgen import enabled_sources
    from recsys.config import load_config
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = load_config([os.path.join(root, "configs", f) for f in ("default.yaml", "best.yaml")])
    names = enabled_sources(cfg)
    assert len(names) == 14 and set(names) <= set(REGISTRY)
    voters = {n for n, w in cfg["selection"]["weights"].items() if w}
    assert voters == {"bm25", "history", "twotower", "twotower_pavel", "twotower_lazy"} and voters <= set(names)
    assert all(cfg["candgen"]["sources"][n].get("features", True) for n in names)
