"""Двухбашенный кандген (recsys/twotower/) на крошечной случайной Qwen3: данные, лосс и маска дублей, этапы обучения
с LoRA, сохранение и загрузка. Кандген по готовым векторам — tests/test_retrieval.py::test_dense_reads_vectors."""
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from recsys.twotower.data import (QueryDataset, SequenceBuilder, TokenizerSpec,  # noqa: E402
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


def test_resume_from_best_without_checkpoint(tmp_path):
    """Прогон старой версии: есть только model.pt (веса, этап, шаг). resume продолжает с его шага, оптимизатор
    заново с разгоном lr; model.pt этапа B — с LoRA."""
    import copy
    ex = _examples()
    tr, val = ex.subset(np.arange(0, 64)), ex.subset(np.arange(64, 72))
    b = SequenceBuilder(_spec(), "find")
    cfg = copy.deepcopy(TCFG)
    cfg["train"].update(batch_size=4, eval_every=3, log_every=1,
                        stages=[{"name": "A", "epochs": 0.5, "lora": False, "lr": 1e-2},      # шаги 1..8
                                {"name": "B", "epochs": 0.25, "lora": True, "lr": 1e-2, "lr_lora": 1e-3}])  # 9..12
    train(_model(), tr, val, b, cfg, str(tmp_path), torch.device("cpu"), log=lambda s: None)
    (tmp_path / "checkpoint.pt").unlink()
    mk = torch.load(tmp_path / "model.pt", weights_only=False)
    mk["val"].update(stage="B", step=10)          # как будто лучшее было на 2-м шаге этапа B
    torch.save(mk, tmp_path / "model.pt")
    (tmp_path / "log.jsonl").unlink()

    msgs = []
    model = _model()
    train(model, tr, val, b, cfg, str(tmp_path), torch.device("cpu"), log=msgs.append, resume=True)
    assert any("продолжаю с весов" in m and "этап 2, шаг 2" in m for m in msgs)
    assert any("этап B: шаги 3..4" in m for m in msgs) and not any("этап A:" in m for m in msgs)
    lrs = [json.loads(line)["lr"] for line in (tmp_path / "log.jsonl").read_text().splitlines() if '"loss"' in line]
    assert lrs[0] < 1e-2 * 0.1                     # разгон lr после продолжения без оптимизатора
    assert hasattr(model.user.base, "peft_config") and torch.load(tmp_path / "checkpoint.pt", weights_only=False)["done"]


# ---------------------------------------------------------------- arch v2: коды + теги, общие веса, shortcut

def _tags():
    from recsys.twotower.tags import build_track_tags
    rng = np.random.default_rng(1)
    names = [f"tag{i}" for i in range(12)]
    df = pd.DataFrame({"track_id": [f"t{i}" for i in range(N_ITEMS)],
                       "tags": [list(rng.choice(names, 3, replace=False)) for _ in range(N_ITEMS)],
                       "tag_weights": [[100.0, 50.0, 10.0]] * N_ITEMS,
                       "genres": [["rock"] if i % 2 else [] for i in range(N_ITEMS)]})
    df.at[5, "tags"], df.at[5, "tag_weights"], df.at[5, "genres"] = ["unique_tag"], [100.0], []
    return build_track_tags(df, [f"t{i}" for i in range(N_ITEMS)], min_count=2, max_per_track=8)


def _model_v2(tags=True, shortcut=True):
    import copy
    from recsys.twotower.model import TwoTowerV2
    cfg = copy.deepcopy(TCFG)
    cfg.update(arch="v2", tags={"enabled": tags, "dim": 8}, shortcut={"enabled": shortcut})
    codes, codebooks, g = _codes()
    m = build_model(cfg, _base(), codes, VOCAB, codebooks, song_groups=g, tags=_tags() if tags else None)
    assert isinstance(m, TwoTowerV2)
    return m, cfg


def test_track_tags():
    t = _tags()
    assert t.vocab[0] == "<no_tags>" and "unique_tag" not in t.vocab          # у одного трека — не в словаре
    assert t.idx.shape == (N_ITEMS, 8) and np.allclose(t.w.sum(1), 1.0)       # веса трека нормированы
    assert t.idx[5, 0] == 0 and t.w[5, 0] == 1.0                              # тегов не осталось -> <no_tags>
    rock = t.vocab.index("rock")
    assert (t.idx[1] == rock).any() and not (t.idx[0] == rock).any()
    assert (t.idx[t.w == 0] == t.pad).all()


def test_v2_shared_tables_and_shortcut():
    m, _ = _model_v2()
    assert m.user.sem is m.item.sem                                           # общая таблица кодов
    b = SequenceBuilder(_spec(), "find")
    batch = b.collate([(np.array([1, 2, 3]), [11, 12], 2), (np.array([], dtype=np.int32), [13], 7)])
    out = m.loss(batch, negatives="full")
    assert torch.isfinite(out["loss"]) and 0 <= out["null"].item() <= 1
    assert m.user.last_null[1].item() == pytest.approx(1.0)                   # нет истории -> весь вес у пустого слота
    out["loss"].backward()
    assert m.user.sem.delta.weight.grad is not None and m.item.tags.emb.weight.grad is not None
    assert m.user.k.weight.grad is not None and m.user.alpha.grad is not None
    assert all(p.grad is None for p in m.user.base.parameters())

    # градиент в таблицу кодов приходит и со стороны трековой башни (общие веса): только выход, без Qwen
    m.zero_grad()
    m.encode_items().sum().backward()
    assert m.user.sem.delta.weight.grad is not None and m.user.sem.delta.weight.grad.abs().sum() > 0

    u = m.encode_users(batch)
    assert u.shape == (2, 16) and torch.allclose(u.norm(dim=1), torch.ones(2), atol=1e-5)
    logits, labels = m.logits(u, batch["target"], "in_batch")
    assert logits.shape == (2, 2) and labels.tolist() == [0, 1]


def test_v2_without_tags_and_shortcut():
    m, _ = _model_v2(tags=False, shortcut=False)
    b = SequenceBuilder(_spec(), "find")
    batch = b.collate([(np.array([1, 2]), [11], 3)])
    out = m.loss(batch)
    assert torch.isfinite(out["loss"]) and "null" not in out and m.item.tags is None


def test_v2_train_save_load(tmp_path):
    from recsys.twotower.train import evaluate
    ex = _examples()
    tr, val = ex.subset(np.arange(0, 60)), ex.subset(np.arange(60, 72))
    b = SequenceBuilder(_spec(), "find")
    m, cfg = _model_v2()
    train(m, tr, val, b, cfg, str(tmp_path), torch.device("cpu"), log=lambda s: None)
    log = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert any("null" in r for r in log if "loss" in r)
    fresh, _ = _model_v2()
    load_weights(fresh, str(tmp_path / "last.pt"), cfg["model"]["lora"])
    u1 = encode_examples(m.eval(), val, b, torch.device("cpu"))
    u2 = encode_examples(fresh.eval(), val, b, torch.device("cpu"))
    assert np.allclose(u1, u2, atol=1e-5)
    res = evaluate(fresh, val, b, torch.device("cpu"), "fp32", by_type=True)
    assert 0 <= res["null"] <= 1 and all("null" in v for v in res["by_type"].values())


def test_v2_shortcut_raises_history_target():
    """Короткий путь: трек из истории сдвигает u к своему вектору — косинус u с целью из истории растёт с α."""
    m, _ = _model_v2()
    m.eval()
    b = SequenceBuilder(_spec(), "find")
    batch = b.collate([(np.array([3, 9, 21]), [11], 9), (np.array([1, 30]), [12], 30)])
    with torch.no_grad():
        v = m.encode_items()[batch["target"]]
        m.user.alpha.fill_(0.0)
        s0 = (m.encode_users(batch) * v).sum(1)
        m.user.alpha.fill_(5.0)
        s1 = (m.encode_users(batch) * v).sum(1)
    assert (s1 > s0).all()
    is_hist = (batch["item"] >= 0) & (batch["level"] == 0)
    assert is_hist.sum(1).tolist() == [3, 2]                                   # ключи — по одному на трек истории


def test_encode_resumable_chunks(tmp_path, monkeypatch):
    """Векторы запросов кусками: второй запуск берёт куски с диска; нехватка памяти — пачка меньше, кусок заново."""
    import encode_twotower as enc
    model = _model().eval()
    ex = _examples()
    b = SequenceBuilder(_spec(), "go", n_codes=4)
    monkeypatch.setattr(enc, "CHUNK", 10)
    u1, _ = enc.encode_resumable(model, ex, b, torch.device("cpu"), "fp32", 8, None, tmp_path, "valid")
    files = sorted(p.name for p in tmp_path.iterdir())
    assert len(files) == (len(ex) + 9) // 10 and u1.shape[0] == len(ex)
    calls = []
    monkeypatch.setattr(enc, "encode_examples", lambda *a, **k: calls.append(1))
    u2, _ = enc.encode_resumable(model, ex, b, torch.device("cpu"), "fp32", 8, None, tmp_path, "valid")
    assert not calls and np.array_equal(u1, u2)                 # всё с диска
    (tmp_path / files[-1]).unlink()
    real = encode_examples
    fail = {"left": 1}

    def flaky(m, e, bl, d, p, bs, **k):
        if fail["left"]:
            fail["left"] -= 1
            raise torch.OutOfMemoryError("нет памяти")
        return real(m, e, bl, d, p, bs, **k)

    monkeypatch.setattr(enc, "encode_examples", flaky)
    u3, bs = enc.encode_resumable(model, ex, b, torch.device("cpu"), "fp32", 8, None, tmp_path, "valid")
    assert bs == 4 and np.allclose(u1.astype(np.float32), u3.astype(np.float32), atol=1e-3)
