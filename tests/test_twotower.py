"""Двухбашенный кандген (recsys/twotower/, recsys/retrieval/twotower.py) на крошечной случайной Qwen3:
данные, лосс и маска дублей, этапы обучения с LoRA, сохранение и загрузка, источник по готовым векторам."""
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from recsys.twotower.data import (Examples, QueryDataset, SequenceBuilder, TokenizerSpec,  # noqa: E402
                                  build_examples, is_val_user, recent_unique)
from recsys.twotower.model import build_model, trainable_state  # noqa: E402
from recsys.twotower.train import encode_examples, load_weights, recall_at, train  # noqa: E402

N_ITEMS, VOCAB = 40, [8, 8, 8, 2]
TCFG = {
    "model": {"dim": 16, "precision": "fp32", "lora": {"r": 4, "alpha": 8, "dropout": 0.0}},
    "codes": {"init": "centroids"},
    "input": {"history_len": 5, "history_dropout": 0.5, "max_query_tokens": 12, "instruction": "find"},
    "loss": {"negatives": "full", "mask_same_song": True, "temperature": 0.1},
    "train": {"batch_size": 8, "num_workers": 0, "val_queries": 16, "eval_every": 0, "seed": 0,
              "stages": [{"name": "A", "epochs": 0.5, "lora": False, "lr": 1e-2},
                         {"name": "B", "epochs": 0.5, "lora": True, "lr": 1e-2, "lr_lora": 1e-3}]},
    "encode": {"batch_size": 16},
}


def _spec():
    return TokenizerSpec(lambda texts: [[10 + ord(c) % 200 for c in t] for t in texts], eos_id=1, pad_id=0)


def _base():
    cfg = transformers.Qwen3Config(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                   max_position_embeddings=512)
    torch.manual_seed(0)
    return transformers.AutoModel.from_config(cfg)


def _codes(seed=0):
    rng = np.random.default_rng(seed)
    codes = np.c_[rng.integers(0, 8, (N_ITEMS, 3)), np.zeros(N_ITEMS, int)]
    codes[1, :3] = codes[0, :3]
    codes[1, 3] = 1
    codebooks = rng.normal(size=(3, 8, 6)).astype(np.float32)
    groups = np.full(N_ITEMS, -1)
    groups[[2, 3]] = 0                                     # треки 2 и 3 — одна песня
    return codes, codebooks, groups


def _model(groups=None):
    codes, codebooks, g = _codes()
    return build_model(TCFG, _base(), codes, VOCAB, codebooks, song_groups=g if groups is None else groups)


def _examples(n_users=12, per_user=6, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"t{i}" for i in range(N_ITEMS)]
    users = pd.DataFrame({"user_id": range(n_users),
                          "history": [list(rng.choice(ids, 15)) for _ in range(n_users)]})
    q = pd.DataFrame({"query_id": [f"q{i}" for i in range(n_users * per_user)],
                      "user_id": np.repeat(range(n_users), per_user),
                      "query": [f"song {i % 7}" for i in range(n_users * per_user)],
                      "query_type": ["mood", "genre"] * (n_users * per_user // 2)})
    qrels = pd.DataFrame({"query_id": q["query_id"], "target_m4a_id": rng.choice(ids, len(q))})
    pos = {t: i for i, t in enumerate(ids)}
    return build_examples({"users": users, "queries": q, "qrels": qrels}, pos, _spec(), history_len=5,
                          max_query_tokens=12)


def test_recent_unique_and_val_split():
    pos = {"a": 0, "b": 1, "c": 2}
    assert recent_unique(["a", "b", "x", "a", "c", "c"], pos, 2).tolist() == [0, 2]   # свежие a, c; x без кода
    assert recent_unique([], pos, 3).tolist() == []
    assert is_val_user(123, 0.3) == is_val_user(123, 0.3) and not is_val_user(5, 0.0)


def test_sequence_left_padding_and_track_slots():
    b = SequenceBuilder(_spec(), "find", n_codes=4)
    batch = b.collate([(np.array([3, 5]), [11, 12], 3), (np.array([], dtype=np.int32), [11], 4)])
    tok, item, level, mask = batch["tok"], batch["item"], batch["level"], batch["mask"]
    assert tok.shape == item.shape == level.shape == mask.shape
    assert (tok[:, -1] == 1).all() and (mask[:, -1] == 1).all()          # EOS последним у всех: паддинг слева
    assert mask[1, 0] == 0 and mask[0].all()
    slots = item[0] >= 0
    assert item[0][slots].tolist() == [3, 3, 3, 3, 5, 5, 5, 5] and level[0][slots].tolist() == [0, 1, 2, 3] * 2
    assert (item[1] < 0).all() and batch["target"].tolist() == [3, 4]


def test_history_dropout_and_len():
    ex = _examples()
    full = QueryDataset(ex)[0][0]
    assert len(QueryDataset(ex, history_len=2)[0][0]) == 2 and (QueryDataset(ex, history_len=2)[0][0] == full[-2:]).all()
    torch.manual_seed(0)
    lens = {len(QueryDataset(ex, history_dropout=1.0)[0][0]) for _ in range(50)}
    assert min(lens) == 0 and max(lens) <= len(full)


def test_loss_masks_same_song_and_trains_only_new_params():
    model = _model()
    b = SequenceBuilder(_spec(), "find")
    batch = b.collate([(np.array([1, 2]), [11, 12], 2), (np.array([4]), [13], 2), (np.array([5]), [14], 7)])
    u = model.encode_users(batch)
    assert u.shape == (3, 16) and torch.allclose(u.norm(dim=1), torch.ones(3), atol=1e-5)
    for neg in ("full", "in_batch"):
        out = model.loss(batch, negatives=neg, logq=neg == "in_batch")
        assert torch.isfinite(out["loss"])
    out = model.loss(batch, negatives="full")
    out["loss"].backward()
    assert all(p.grad is None for p in model.user.base.parameters())             # Qwen заморожен
    assert model.user.sem.proj.weight.grad is not None and model.item.table.weight.grad is not None
    # трек 3 — другая запись цели 2: не негатив (-inf), сама цель и остальные треки — обычные логиты
    with torch.no_grad():
        logits, labels = model.logits(u, batch["target"], "full")
        assert torch.isinf(logits[0, 3]) and torch.isinf(logits[1, 3]) and torch.isfinite(logits[0, 2])
        assert torch.isfinite(logits[2]).all() and labels.tolist() == [2, 2, 7]
        # in_batch: у примеров 0 и 1 одна цель — друг другу не негативы
        logits, labels = model.logits(u, batch["target"], "in_batch")
        assert torch.isinf(logits[0, 1]) and torch.isinf(logits[1, 0]) and torch.isfinite(logits[0, 2])
        assert labels.tolist() == [0, 1, 2]
        unmasked, _ = model.logits(u, batch["target"], "full", mask_same_song=False)
        assert torch.isfinite(unmasked).all()


def test_train_stages_save_and_load(tmp_path):
    ex = _examples()
    tr, val = ex.subset(np.arange(0, 60)), ex.subset(np.arange(60, 72))
    b = SequenceBuilder(_spec(), "find")
    model = _model()
    best = train(model, tr, val, b, TCFG, str(tmp_path), torch.device("cpu"), log=lambda s: None)
    assert 0.0 <= best["recall@100"] <= 1.0 and (tmp_path / "model.pt").exists() and (tmp_path / "last.pt").exists()
    log = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert {r["stage"] for r in log if "val" in r} == {"A", "B"}
    assert any("lora_" in k for k in trainable_state(model))                  # этап B добавил LoRA
    assert not any(k.startswith("user.base.") and "lora_" not in k for k in trainable_state(model))

    fresh = _model()
    load_weights(fresh, str(tmp_path / "last.pt"), TCFG["model"]["lora"])
    u1 = encode_examples(model.eval(), val, b, torch.device("cpu"))
    u2 = encode_examples(fresh.eval(), val, b, torch.device("cpu"))
    assert np.allclose(u1, u2, atol=1e-5)
    r = recall_at(u1, fresh.encode_items().detach().numpy(), val.target)
    assert set(r) >= {"recall@10", "recall@100", "recall@200", "ranks"}


def test_retriever_reads_vectors(tmp_path, data):
    from recsys.data.load import load_data  # noqa: F401  (data — общая синтетика из conftest)
    from recsys.retrieval.twotower import TwoTowerRetriever
    from recsys.schemas import Context, DialogSummary, Message, Request, UserProfile
    cat = data.catalog
    ids = np.array([str(t) for t in cat.track_ids[:30]])
    rng = np.random.default_rng(0)
    v = rng.normal(size=(30, 8)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    np.save(tmp_path / "item_vectors.npy", v)
    np.save(tmp_path / "item_ids.npy", ids)
    np.savez(tmp_path / "query_vectors_test_public.npz", query_id=np.array(["q1"]), vectors=v[[4]].astype(np.float16))
    r = TwoTowerRetriever(cat, str(tmp_path), top_k=5)
    ctx = Context(Request(dialog=[Message("user", "x")], request_id="q1", shown_ids=[ids[7]]), DialogSummary(),
                  UserProfile(), banned_ids={ids[7]})
    got = [c.track_id for c in r.search(ctx)]
    assert got[0] == ids[4] and ids[7] not in got and len(got) == 5 and set(got) <= set(ids)
    with pytest.warns(UserWarning, match="нет вектора запроса"):
        assert r.search(Context(Request(request_id="нет"), DialogSummary(), UserProfile())) == []
    assert r.n_missing == 1


def test_device_attention_and_nan_guard():
    """sdpa на mps с паддингом слева даёт NaN — device_attention выбирает eager; NaN в векторах — ошибка, а не
    тихий recall (NaN-скор цели иначе считался бы найденным)."""
    from recsys.twotower.train import device_attention
    assert device_attention(torch.device("mps"), "sdpa") == "eager"
    assert device_attention(torch.device("cuda"), "sdpa") == "sdpa"
    m = _model()
    with torch.no_grad():
        m.user.proj.weight.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="NaN"):
        encode_examples(m, _examples(), SequenceBuilder(_spec(), "find"), torch.device("cpu"))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="нет mps")
def test_mps_eager_no_nan():
    codes, codebooks, g = _codes()
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                   attn_implementation="eager")
    m = build_model(TCFG, transformers.AutoModel.from_config(cfg), codes, VOCAB, codebooks, song_groups=g)
    dev = torch.device("mps")
    u = encode_examples(m.to(dev), _examples(), SequenceBuilder(_spec(), "find"), dev, batch_size=32)
    assert u.shape[1] == 16 and not np.isnan(u).any()


def test_resume_after_interruption(tmp_path):
    """Обрыв посреди этапа A: checkpoint.pt с последней валидации; resume продолжает A с того же шага
    (оптимизатор восстановлен), потом этап B с LoRA; законченный прогон resume не переобучает."""
    import copy
    ex = _examples()
    tr, val = ex.subset(np.arange(0, 64)), ex.subset(np.arange(64, 72))
    b = SequenceBuilder(_spec(), "find")
    cfg = copy.deepcopy(TCFG)
    cfg["train"].update(batch_size=4, eval_every=3, log_every=1,
                        stages=[{"name": "A", "epochs": 0.5, "lora": False, "lr": 1e-2},      # 8 шагов
                                {"name": "B", "epochs": 0.25, "lora": True, "lr": 1e-2, "lr_lora": 1e-3}])
    model = _model()
    real_loss, calls = model.loss, {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 5:
            raise RuntimeError("сессия оборвалась")
        return real_loss(*args, **kwargs)

    model.loss = flaky
    with pytest.raises(RuntimeError, match="оборвалась"):
        train(model, tr, val, b, cfg, str(tmp_path), torch.device("cpu"), log=lambda s: None)
    ck = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    assert (ck["stage_idx"], ck["step_in_stage"], ck["step_global"]) == (0, 3, 3) and not ck["lora"]

    msgs = []
    fresh = _model()
    train(fresh, tr, val, b, cfg, str(tmp_path), torch.device("cpu"), log=msgs.append, resume=True)
    assert any("продолжаю" in m for m in msgs) and any("этап A: шаги 4..8" in m for m in msgs)
    steps = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in steps if r.get("stage") == "B" and "loss" in r][-1] == 12   # 8 + 4 шага этапа B
    assert hasattr(fresh.user.base, "peft_config")
    ck = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    assert ck["done"] and ck["lora"]

    msgs = []
    again = _model()
    best = train(again, tr, val, b, cfg, str(tmp_path), torch.device("cpu"), log=msgs.append, resume=True)
    assert not any("[train] этап" in m for m in msgs) and best["recall@100"] >= 0       # ничего не обучал
