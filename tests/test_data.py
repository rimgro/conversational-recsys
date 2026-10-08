import os

from recsys.config import deep_update
from recsys.data import load_data
from recsys.data.crs import aggregate_history, parse_demographics
from recsys.data.synthetic import make_synthetic


def test_aggregate_history():
    items = aggregate_history(["a", "b", "a", "c", "a"], [1, 2, 3, 4, 5])
    assert [(h.track_id, h.count, h.timestamp) for h in items] == [("a", 3.0, 5), ("c", 1.0, 4), ("b", 1.0, 2)]


def test_parse_demographics():
    assert parse_demographics(None) == {}
    assert parse_demographics({"age": 0.0, "country": "AQ", "gender": "m"}) == {"gender": "m"}
    assert parse_demographics('{"age": 25, "country": "DE", "gender": "f"}') == {"age": 25, "gender": "f",
                                                                                  "country": "DE"}


def test_crs_loader_reads_files(cfg, tmp_path):
    """Синтетика -> parquet в формате датасета -> load_data(source=crs): тот же путь, что у настоящих файлов."""
    meta, splits = make_synthetic(n_tracks=300, n_users=20, seed=1)
    meta.to_parquet(os.path.join(tmp_path, "tracks_meta.parquet"), index=False)
    for name, tables in splits.items():
        tables["users"].to_parquet(os.path.join(tmp_path, f"{name}.parquet"), index=False)
        tables["queries"].to_parquet(os.path.join(tmp_path, f"{name}_queries.parquet"), index=False)
        tables["qrels"].to_parquet(os.path.join(tmp_path, f"{name}_qrels.parquet"), index=False)
    c = deep_update(cfg, {"data": {"source": "crs", "cache_dir": str(tmp_path / "cache"), "max_queries_per_user": None,
                                   "crs": {"dir": str(tmp_path), "n_users": 10, "splits": ["train", "test_public"]}}})
    data = load_data(c, verbose=False)
    assert len(data.catalog) == 300 and data.catalog.embeddings.shape == (300, 128)
    assert {r.user_id for r in data.splits["train"]} <= {str(100000 + u) for u in range(20)}
    assert len({r.user_id for r in data.requests}) == 10
    queries = splits["test_public"]["queries"]
    users = {int(r.user_id) for r in data.requests}
    assert sorted(r.request_id for r in data.requests) == sorted(queries[queries.user_id.isin(users)].query_id)
    r = data.requests[0]
    assert len(r.target_ids) == 1 and r.dialog[0].role == "user" and r.meta["query_type"] and r.meta["artist"]
    sim = next(r for r in data.splits["train"] + data.requests if r.meta["query_type"] == "similar_to")
    assert sim.meta["exclude_ids"] and sim.meta["exclude_artist"] and sim.meta["is_new"]
    # второй раз каталог берётся из кэша вместе с эмбеддингами
    again = load_data(c, verbose=False)
    assert (again.catalog.embeddings == data.catalog.embeddings).all()
    assert again.catalog.df["tags"].tolist() == data.catalog.df["tags"].tolist()


def test_synthetic_targets_mostly_from_history(data):
    pos = [r for r in data.requests if r.meta["source"] == "positive"]
    is_new = [r.meta["is_new"] for r in pos]
    in_hist = [r.target_ids[0] in {h.track_id for h in r.history} for r in pos]
    assert 0.7 < 1 - sum(is_new) / len(is_new) and is_new == [not x for x in in_hist]
    for r in data.requests:  # discovery: цель — новый трек; similar_to — другого артиста, чем образец
        if r.meta["source"] == "discovery":
            assert r.meta["is_new"] and r.target_ids[0] not in {h.track_id for h in r.history}
            if r.meta["query_type"] == "similar_to":
                assert data.catalog.artist(r.target_ids[0]) != r.meta["exclude_artist"]
