import os

import pandas as pd
import pytest

from recsys.config import deep_update
from recsys.data import Dataset
from recsys.data.crs import aggregate_history, parse_demographics
from recsys.data.parts import check_disjoint, make_part, normalize_query, text_bucket
from recsys.data.synthetic import make_synthetic


def test_aggregate_history():
    items = aggregate_history(["a", "b", "a", "c", "a"], [1, 2, 3, 4, 5])
    assert [(h.track_id, h.count, h.timestamp) for h in items] == [("a", 3.0, 5), ("c", 1.0, 4), ("b", 1.0, 2)]


def test_parse_demographics():
    assert parse_demographics(None) == {}
    assert parse_demographics({"age": 0.0, "country": "AQ", "gender": "m"}) == {"gender": "m"}
    assert parse_demographics('{"age": 25, "country": "DE", "gender": "f"}') == {"age": 25, "gender": "f",
                                                                                  "country": "DE"}


def test_parts_split_by_text_without_overlap():
    """Перефразировки одного текста не попадают в разные части; тексты теста не идут в index."""
    assert normalize_query("Play  Song, by X!") == "play song by x"
    texts = [f"query number {i}" for i in range(400)] + ["Shared Text"] * 5
    train = pd.DataFrame({"query_id": [f"q{i}" for i in range(len(texts))], "user_id": 1, "query": texts,
                          "query_type": "genre", "split": "train"})
    test = pd.DataFrame({"query_id": ["t1"], "user_id": 2, "query": ["shared   text"], "query_type": "genre",
                         "split": "test_public"})
    pcfg = {"train_split": "train", "test_split": "test_public", "seed": 1,
            "buckets": {"index": [0, 6], "ranker_train": [6, 9], "valid": [9, 10]},
            "sizes": {"ranker_train": 50, "valid": None, "test": None}}
    parts = {p: make_part(p, train, test, pcfg) for p in ("index", "ranker_train", "valid")}
    check_disjoint(parts)
    assert len(parts["ranker_train"]) <= 50
    assert "shared text" not in set(parts["index"]["normalized"])   # есть в тесте -> не в index
    assert 0 <= text_bucket("shared text") < 10
    with pytest.raises(AssertionError, match="пересекаются"):
        check_disjoint({"index": parts["index"], "valid": parts["index"].head(3)})


def test_dataset_reads_files(cfg, tmp_path):
    """Синтетика -> parquet в формате датасета -> Dataset(source=crs): тот же путь, что у настоящих файлов."""
    meta, splits = make_synthetic(n_tracks=300, n_users=20, seed=1)
    meta.to_parquet(os.path.join(tmp_path, "tracks_meta.parquet"), index=False)
    for name, tables in splits.items():
        tables["users"].to_parquet(os.path.join(tmp_path, f"{name}.parquet"), index=False)
        tables["queries"].to_parquet(os.path.join(tmp_path, f"{name}_queries.parquet"), index=False)
        tables["qrels"].to_parquet(os.path.join(tmp_path, f"{name}_qrels.parquet"), index=False)
    c = deep_update(cfg, {"data": {"source": "crs", "dir": str(tmp_path), "cache_dir": str(tmp_path / "cache")}})
    ds = Dataset(c, verbose=False)
    assert len(ds.catalog) == 300 and ds.catalog.embeddings.shape == (300, 128)
    assert list(ds.catalog.track_ids) == sorted(ds.catalog.track_ids)     # номер трека = место по m4a_id
    test = ds.requests("test")
    queries = splits["test_public"]["queries"]
    assert sorted(r.request_id for r in test) == sorted(queries.query_id)
    r = test[0]
    assert len(r.target_ids) == 1 and r.dialog[0].role == "user" and r.meta["query_type"] and r.meta["artist"]
    assert r.meta["cutoff"] == 1582156800 and all(h.timestamp < r.meta["cutoff"] for h in r.history)
    assert not ds.requests("test", labels=False)[0].target_ids          # сабмит: без ответов
    valid = ds.requests("valid")
    assert {q.meta["split"] for q in valid} == {"train"} and valid[0].meta["cutoff"] == 1579564800
    sim = next(q for q in ds.requests("index") if q.meta["query_type"] == "similar_to")
    assert sim.meta["exclude_ids"] and sim.meta["exclude_artist"] and sim.meta["is_new"]
    meta_cols = ds.tracks_meta(["m4a_id", "lyrics", "no_such_column"])
    assert list(meta_cols["m4a_id"]) == list(ds.catalog.track_ids) and (meta_cols["no_such_column"] == "").all()
    # второй раз каталог берётся из кэша вместе с эмбеддингами
    again = Dataset(c, verbose=False).catalog
    assert (again.embeddings == ds.catalog.embeddings).all()
    assert again.df["tags"].tolist() == ds.catalog.df["tags"].tolist()


def test_synthetic_targets_mostly_from_history(catalog, requests):
    pos = [r for r in requests if r.meta["source"] == "positive"]
    is_new = [r.meta["is_new"] for r in pos]
    in_hist = [r.target_ids[0] in {h.track_id for h in r.history} for r in pos]
    assert 0.7 < 1 - sum(is_new) / len(is_new) and is_new == [not x for x in in_hist]
    for r in requests:  # discovery: цель — новый трек; similar_to — другого артиста, чем образец
        if r.meta["source"] == "discovery":
            assert r.meta["is_new"] and r.target_ids[0] not in {h.track_id for h in r.history}
            if r.meta["query_type"] == "similar_to":
                assert catalog.artist(r.target_ids[0]) != r.meta["exclude_artist"]


def test_user_state(res, requests):
    r = requests[0]
    u = res.users.get(r)
    assert len(u.positions) == len({h.track_id for h in r.history}) and (u.days > 0).all()
    assert u.total == sum(h.count for h in r.history) and u.centroid is not None
    assert res.users.get(r) is u                                     # один раз на пользователя
    w = u.weights(30)
    assert w.shape == u.counts.shape and (w <= u.weights()).all()     # затухание только уменьшает вес


def test_index_does_not_depend_on_test_split():
    """Ответы на другом тестовом сплите (test_private): index — тот же, его тексты исключаются по index_excludes."""
    train = pd.DataFrame({"query_id": [f"q{i}" for i in range(300)], "user_id": 1, "query_type": "genre",
                          "query": [f"text {i % 150}" for i in range(300)], "split": "train"})
    public = train.head(40).assign(split="test_public")
    private = train.tail(40).assign(split="test_private")
    pcfg = {"train_split": "train", "test_split": "test_private", "index_excludes": "test_public", "seed": 1,
            "buckets": {"index": [0, 6], "ranker_train": [6, 9], "valid": [9, 10]}, "sizes": {}}
    a = make_part("index", train, public, {**pcfg, "test_split": "test_public"}, public)
    b = make_part("index", train, private, pcfg, public)
    assert a["query_id"].tolist() == b["query_id"].tolist()
    assert not set(a["normalized"]) & set(public["query"].map(normalize_query))
