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
    for name, df in splits.items():
        df.to_parquet(os.path.join(tmp_path, f"{name}.parquet"), index=False)
    c = deep_update(cfg, {"data": {"source": "crs", "cache_dir": str(tmp_path / "cache"),
                                   "crs": {"dir": str(tmp_path), "n_users": 10}}})
    data = load_data(c, verbose=False)
    assert len(data.catalog) == 300 and data.catalog.embeddings.shape == (300, 128)
    assert {r.user_id for r in data.splits["train"]} <= {str(100000 + u) for u in range(20)}
    assert len({r.user_id for r in data.requests}) == 10
    r = data.requests[0]
    assert len(r.target_ids) == 1 and r.dialog[0].role == "user" and r.meta["query_family"]
    # второй раз каталог берётся из кэша вместе с эмбеддингами
    again = load_data(c, verbose=False)
    assert (again.catalog.embeddings == data.catalog.embeddings).all()
    assert again.catalog.df["tags"].tolist() == data.catalog.df["tags"].tolist()


def test_synthetic_targets_mostly_from_history(data):
    is_new = [r.meta["is_new"] for r in data.requests]
    in_hist = [r.target_ids[0] in {h.track_id for h in r.history} for r in data.requests]
    assert 0.7 < 1 - sum(is_new) / len(is_new) and is_new == [not x for x in in_hist]


def test_crs_loader_reads_parts(cfg, tmp_path):
    """Полный датасет лежит частями: train-00000-of-00003.parquet, ... — результат тот же, что из одного файла."""
    meta, splits = make_synthetic(n_tracks=300, n_users=20, seed=1)
    whole, parts = tmp_path / "whole", tmp_path / "parts"
    whole.mkdir(), parts.mkdir()
    for name, df in {"tracks_meta": meta, **splits}.items():
        df.to_parquet(whole / f"{name}.parquet", index=False)
        for i, chunk in enumerate((df.iloc[:5], df.iloc[5:12], df.iloc[12:])):  # части идут подряд
            chunk.to_parquet(parts / f"{name}-{i:05d}-of-00003.parquet", index=False)
    load = {d: load_data(deep_update(cfg, {"data": {"source": "crs", "cache_dir": None, "crs": {"dir": str(d),
                                                                                              "n_users": None}}}),
                         verbose=False) for d in (whole, parts)}
    assert sorted(load[whole].catalog.track_ids) == sorted(load[parts].catalog.track_ids)
    key = lambda data: sorted(r.request_id for r in data.splits["train"])  # noqa: E731
    assert key(load[whole]) == key(load[parts]) and key(load[whole])
