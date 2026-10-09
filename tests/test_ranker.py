"""Ранкер на маленьком синтетическом датасете в формате Music4All-CRS (struct-демография, список позитивов)."""
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ranker import candidates, data, evaluate, features, priors, train
from ranker.ranker import LGBMRanker

N_TRACKS, N_USERS = 300, 120
GENRES = ["rock", "pop", "jazz", "metal", "hip hop", "folk"]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("crs")
    ids = [f"t{i:05d}" for i in range(N_TRACKS)]
    genre = [GENRES[i % len(GENRES)] for i in range(N_TRACKS)]
    pq.write_table(pa.table({
        "m4a_id": ids,
        "m4a_artist": [f"artist{i % 40}" for i in range(N_TRACKS)],
        "m4a_album": [f"album{i % 80}" for i in range(N_TRACKS)],
        "release_year": [1960 + i % 60 for i in range(N_TRACKS)],
        "lang": ["en" if i % 3 else "ru" for i in range(N_TRACKS)],
        "spotify_popularity": rng.uniform(0, 100, N_TRACKS),
        "m4a_genres_full": genre,
        "m4a_tags_full": [f"{g},{'happy' if i % 2 else 'sad'}" for i, g in enumerate(genre)],
        "lastfm_tag_weights": [json.dumps({g: 100}) for g in genre],
    }), root / "tracks_meta.parquet")

    demo_type = pa.struct([("age", pa.int64()), ("country", pa.string()), ("gender", pa.string())])
    pos_type = pa.struct([("artist", pa.string()), ("is_new", pa.bool_()), ("m4a_id", pa.string()),
                          ("n_listens", pa.int64()), ("query", pa.string()), ("query_family", pa.string()),
                          ("spotify_id", pa.string()), ("title", pa.string()), ("ts", pa.int64())])
    for split in ("train", "test_public"):
        rows = {"user_id": [], "history": [], "history_ts": [], "positives": [], "user_demographics": []}
        for u in range(N_USERS):
            fav = GENRES[u % len(GENRES)]
            fav_ids = [i for i in range(N_TRACKS) if genre[i] == fav]
            hist = list(rng.choice(fav_ids, 30)) + list(rng.choice(N_TRACKS, 10))
            pos = list(dict.fromkeys(rng.choice(hist, 6)))   # в основном повторные прослушивания
            rows["user_id"].append(u)
            rows["history"].append([ids[i] for i in hist])
            rows["history_ts"].append(sorted(rng.integers(1.5e9, 1.57e9, len(hist)).tolist()))
            rows["positives"].append([{"artist": "", "is_new": False, "m4a_id": ids[i], "n_listens": 1,
                                       "query": f"хочу {genre[i]}", "query_family": "genre", "spotify_id": "",
                                       "title": "", "ts": 1580000000} for i in pos])
            rows["user_demographics"].append(None if u % 3 == 0 else
                                             {"age": int(15 + u % 50), "country": "RU", "gender": "mf"[u % 2]})
        pq.write_table(pa.table({
            "user_id": pa.array(rows["user_id"], pa.int64()),
            "history": pa.array(rows["history"], pa.list_(pa.string())),
            "history_ts": pa.array(rows["history_ts"], pa.list_(pa.int64())),
            "positives": pa.array(rows["positives"], pa.list_(pos_type)),
            "user_demographics": pa.array(rows["user_demographics"], demo_type),
        }), root / f"{split}.parquet")
    return root


def genre_source(tracks):
    by_genre = {}
    for tid, tags in zip(tracks["track_id"], tracks["tags"]):
        by_genre.setdefault(tags[0], []).append(tid)

    class Src:
        name = "bm25_genres"

        def search(self, query, words, k):
            return [(t, 1.0) for w in words for t in by_genre.get(w, [])][:k]
    return Src()


def random_source(tracks, seed):
    ids = tracks["track_id"].to_numpy()

    def fn(query, k):
        r = np.random.default_rng(abs(hash((query, seed))) % 2**32)
        return [(t, 1.0) for t in r.choice(ids, k, replace=False)]
    return candidates.FunctionSource(f"rand{seed}", fn)


def test_rrf():
    out = candidates.rrf({"a": ["x", "y"], "b": ["y", "z"]}, k=60, top_n=2)
    assert [t for t, _, _ in out] == ["y", "x"]
    assert out[0][2] == 2 and abs(out[0][1] - (1 / 62 + 1 / 61)) < 1e-12


def test_age_bucket_and_hash_split():
    assert [data.age_bucket(a) for a in (None, 0, 9, 15, 27, 49, 50, 99, 150)] == \
        ["unknown", "unknown", "unknown", "10s", "20s", "40s", "50+", "50+", "unknown"]
    ids = np.arange(5000)
    a = data.hash_split(ids, 0.65, "priors")
    assert np.array_equal(a, data.hash_split(ids[::-1], 0.65, "priors")[::-1])
    assert 0.6 < a.mean() < 0.7


def test_end_to_end(dataset, tmp_path):
    tracks = data.load_tracks(dataset)
    codes = features.Codes(tracks)
    tr = data.load_split(dataset, "train", tracks)
    assert set(tr.users["age_bucket"]) <= set(data.AGE_BUCKETS) and "unknown" in set(tr.users["age_bucket"])

    users = tr.users["user_id"].to_numpy()
    in_a = data.hash_split(users, 0.65, "priors")
    part_a, part_b = data.subset(tr, users[in_a]), data.subset(tr, users[~in_a])
    pri = priors.compute_priors(part_a, tracks, alpha=5.0)
    pri.save(tmp_path / "pri")
    pri = priors.Priors.load(tmp_path / "pri")
    assert 0 < pri.p0 < 1

    sources = [genre_source(tracks), random_source(tracks, 1)]
    q = part_b.queries.copy()
    q["words"] = candidates.parse_words(q, lambda s: s.split()[1:])
    pools = candidates.build_pools(q, sources, tracks, top_k=30, top_n=50)
    feats = features.build_features(pools, q, features.UserStats(part_b, codes), codes, pri)
    assert list(feats.columns[:3]) == ["qid", "track_idx", "label"]
    assert set(features.FEATURES) <= set(feats.columns)
    assert feats.groupby("qid")["label"].sum().max() == 1
    hist_rows = feats[feats["in_history"] == 1]
    assert (hist_rows["log_history_count"] > 0).all()
    assert feats["query_tag_frac"].between(0, 1).all()

    b_users = q["user_id"].unique()
    valid_users = set(b_users[data.hash_split(b_users, 0.2, "valid")])
    vmask = feats["qid"].map(q.set_index("qid")["user_id"]).isin(valid_users).to_numpy()
    booster, info = train.train(feats[~vmask], feats[vmask], params={"min_data_in_leaf": 5},
                                num_boost_round=50, early_stopping=10, log_every=0)
    train.save(booster, tmp_path / "model", info)

    ranker = LGBMRanker(tmp_path / "model")
    report = evaluate.compare(feats, ranker.score(feats), q, k=20)
    assert {"all", "genre"} <= set(report.index)
    # история и жанр в признаках, случайный источник — шум: ранкер должен обойти RRF
    assert report.loc["all", "ranker_ndcg@20"] > report.loc["all", "rrf_ndcg@20"]
    top = ranker.rank(feats, top_k=20)
    assert top.groupby("qid").size().max() <= 20


def test_queries_missing_from_pool_count_as_zero():
    queries = pd.DataFrame({"qid": ["a", "b"], "target_idx": [1, 2], "query_family": ["genre", "mood"],
                            "is_new": [False, True]})
    scored = pd.DataFrame({"qid": ["a", "a"], "track_idx": [5, 1], "s": [2.0, 1.0]})
    pq_ = evaluate.per_query(scored, queries, "s", k=20)
    assert pq_["ndcg@20"].tolist() == [pytest.approx(1 / np.log2(3)), 0.0]
    assert pq_["in_pool"].tolist() == [True, False]
