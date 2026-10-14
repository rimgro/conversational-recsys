"""Онлайн-инференс (recsys/inference.py): строки датасета -> тот же ответ, что у пайплайна по части; HTTP; онлайн-
кодирование башен совпадает с кодированием частей (encode_examples); башня Павла — канонические группы треков."""
import json
import threading
import types
import urllib.request

import numpy as np
import pandas as pd
import pytest

from recsys.data.crs import build_requests, make_request
from recsys.inference import Recommender, make_server, resolve_paths
from recsys.pipeline import Pipeline
from recsys.schemas import Context


@pytest.fixture(scope="module")
def rec(cfg):
    return Recommender(cfg, verbose=False)


@pytest.fixture(scope="module")
def rows(res):
    """Строки test_public синтетики: запросы части test и их пользователи — как в файлах датасета."""
    queries = res.dataset.part("test").head(15)
    users = res.dataset.users("test_public", queries["user_id"].unique())
    return queries, users


def test_make_request_same_as_build_requests(res, rows):
    queries, users = rows
    cutoff = res.dataset.cutoff("test_public")
    built = build_requests(queries, users, cutoff=cutoff)
    by_user = {u: r for u, r in zip(users["user_id"], users.to_dict("records"))}
    for q, b in zip(queries.to_dict("records"), built):
        r = make_request(q, by_user[q["user_id"]], cutoff)
        assert r.history == b.history and r.user_messages == b.user_messages and r.user_attrs == b.user_attrs
        assert r.request_id == b.request_id and r.meta["cutoff"] == cutoff and "query_type" not in r.meta
        assert list(r.listens[0]) == list(by_user[q["user_id"]]["history"]) and b.listens is None


def test_recommend_equals_pipeline(rec, cfg, res, rows):
    """Строки датасета через Recommender — та же выдача и ответ, что Pipeline.run по запросам части."""
    queries, users = rows
    pipe = Pipeline(cfg, res)
    by_user = {u: r for u, r in zip(users["user_id"], users.to_dict("records"))}
    for q, req in zip(queries.to_dict("records"), build_requests(queries, users, cutoff=res.dataset.cutoff("test_public"))):
        out = rec.recommend(q, by_user[q["user_id"]])
        want = pipe.run(req)
        assert out["query_id"] == q["query_id"] and out["top20"] == want.track_ids[:20]
        assert out["response"] == want.text and out["timings"]["total"] > 0


def test_cutoff_and_no_user_cache(rec, rows):
    queries, users = rows
    user = users.iloc[0].to_dict()
    assert rec.cutoff({}, user) == rec.windows["test_public"]           # граница окна сплита, как при обучении
    live = {k: v for k, v in user.items() if k != "split"}
    assert rec.cutoff({}, live) is None and rec.cutoff({"cutoff": 123}, live) == 123
    req = rec.request({"query": "rock"}, user)
    assert req.user_id is None and req.meta["user_id"] == str(user["user_id"])
    out = rec.recommend("some calm jazz", live)                          # только текст, живой пользователь
    assert len(out["top20"]) == 20 and out["query_id"] is None
    assert rec.recommend("rock", {})["response"]                         # ни истории, ни профиля


def test_http(rec, rows):
    queries, users = rows
    server = make_server(rec, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert json.load(urllib.request.urlopen(url + "/health"))["status"] == "ok"
        q, u = queries.iloc[0].to_dict(), users.iloc[0].to_dict()
        body = {"query": {"query_id": q["query_id"], "query": q["query"]},
                "user": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in u.items()}}

        def post(data):
            r = urllib.request.Request(url + "/recommend", data=json.dumps(data).encode(), method="POST")
            return json.load(urllib.request.urlopen(r))

        one = post(body)
        assert one["query_id"] == q["query_id"] and len(one["top20"]) == 20 and one["response"]
        assert [x["top20"] for x in post([body, body])] == [one["top20"]] * 2
        with pytest.raises(urllib.error.HTTPError) as e:
            post({"user": {}})                                           # нет query
        assert e.value.code == 400
    finally:
        server.shutdown()
        server.server_close()


def test_resolve_paths():
    cfg = {"data": {"dir": "data", "cache_dir": "/abs/cache", "track_docs": None}, "ranker": {"model_dir": "ranker"},
           "semantic_ids": {"dir": "semantic_ids"},
           "candgen": {"sources": {"twotower_pavel": {"dir": "", "bundle": "towers/p/model"}, "bm25": {"enabled": True}}}}
    out = resolve_paths(cfg, "/srv/b")
    assert out["data"] == {"dir": "/srv/b/data", "cache_dir": "/abs/cache", "track_docs": None}
    assert out["ranker"]["model_dir"] == "/srv/b/ranker" and out["semantic_ids"]["dir"] == "/srv/b/semantic_ids"
    assert out["candgen"]["sources"]["twotower_pavel"] == {"dir": "", "bundle": "/srv/b/towers/p/model"}
    assert cfg["data"]["dir"] == "data"                                  # исходный конфиг не меняется


# ---------------------------------------------------------------- башни онлайн

DAY = 86400


def _tables(ids, n_users=4, seed=0):
    """Пользователи с сырой историей (повторы, время по возрастанию; у последнего — пустая) и по два запроса."""
    rng = np.random.default_rng(seed)
    end = 1000 * DAY
    history = [list(rng.choice(ids, 25)) for _ in range(n_users - 1)] + [[]]
    ts = [sorted(rng.choice(np.arange(end - 300 * DAY, end), len(h), replace=False).tolist()) for h in history]
    users = pd.DataFrame({"user_id": range(n_users), "history": history, "history_ts": ts})
    q = pd.DataFrame({"query_id": [f"q{i}" for i in range(2 * n_users)], "user_id": np.repeat(range(n_users), 2),
                      "query": [f"song {i}" for i in range(2 * n_users)], "query_type": "mood"})
    return users, q, end


def _ctx(user, query, cutoff):
    from recsys.data.users import UserState
    req = make_request({"query": query}, user, cutoff)
    return Context(req, None, None, user=UserState(np.zeros(0, np.int32), np.zeros(0), np.zeros(0), cutoff))


@pytest.mark.parametrize("arch", ["lazy", "v2"])
def test_online_encoder_equals_encode_examples(arch):
    """OnlineEncoder (одна реплика) даёт тот же вектор, что кодирование части (encode_examples): та же история
    (lazy — по сырым прослушиваниям до cutoff, v2 — последние уникальные), тот же текст."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from recsys.twotower.build import OnlineEncoder
    from recsys.twotower.data import SequenceBuilder, build_examples
    from recsys.twotower.lazy import LazyBuilder, lazy_history
    from recsys.twotower.train import encode_examples
    if arch == "lazy":
        import test_lazy as t
        model, tcfg = t._model(), t.TCFG
        with torch.no_grad():                    # гейты cross-attention на старте 0: декодер не смотрит на историю
            for block in model.user.xattn.values():
                block.gate.fill_(1.0)
    else:
        import test_twotower as t
        model, tcfg = t._model_v2()
    model.eval()
    ids = [f"t{i}" for i in range(t.N_ITEMS)]
    pos = {x: i for i, x in enumerate(ids)}
    users, q, end = _tables(ids)
    n = tcfg["input"]["history_len"]
    make_history = (lambda row: lazy_history(row["history"], row["history_ts"], pos, n, end)) if arch == "lazy" else None
    ex = build_examples({"users": users, "queries": q}, pos, t._spec(), n, tcfg["input"]["max_query_tokens"],
                        make_history=make_history)
    builder = LazyBuilder(t._spec(), "find") if arch == "lazy" else SequenceBuilder(t._spec(), "find")
    offline = encode_examples(model, ex, builder, torch.device("cpu"), batch_size=3)
    enc = OnlineEncoder(model, builder, types.SimpleNamespace(sid=types.SimpleNamespace(_pos=pos)),
                        {**tcfg, "arch": arch}, precision="fp32")
    by_user = users.set_index("user_id")
    for i, row in enumerate(q.itertuples()):
        u = by_user.loc[row.user_id].to_dict()
        online = enc.encode(_ctx(u, row.query, end))
        assert np.allclose(online, offline[i], atol=1e-5), (arch, i)
    u = by_user.loc[0].to_dict()                 # история действительно влияет на вектор
    other = {"history": u["history"][:3], "history_ts": u["history_ts"][:3]}
    assert not np.allclose(enc.encode(_ctx(u, "song 0", end)), enc.encode(_ctx(other, "song 0", end)), atol=1e-4)


def test_pavel_search_canonical_groups(res):
    """Записи одной песни (один spotify_id) — один трек: остаётся лучшая по скору; без запроса — пусто."""
    pytest.importorskip("torch")
    from recsys.candgen import REGISTRY
    from recsys.twotower.pavel import PavelTower
    tower = PavelTower.__new__(PavelTower)
    tower.ids = [str(x) for x in res.catalog.track_ids[:5]]
    tower.group = np.array([0, 1, 1, 2, 3])                     # 1 и 2 — одна песня
    tower.vectors = np.eye(5, dtype=np.float32)
    tower.encode = lambda query, history: np.array([0.1, 0.5, 0.9, 0.3, 0.0], dtype=np.float32)
    ids, scores = tower.search("x", [], k=3)
    assert ids == [tower.ids[2], tower.ids[3], tower.ids[0]] and np.allclose(scores, [0.9, 0.3, 0.1])
    assert tower.search("  ", [])[0] == []
    cg = REGISTRY["twotower_pavel"]({"top_k": 3, "dir": "", "online": True}, res)   # файлов списков нет
    cg._tower = tower
    got = cg.retrieve([_ctx({"history": []}, "x", None)])[0]
    assert got.ids.tolist() == [2, 3, 0]
