"""Lazy-кандген (recsys/twotower/lazy.py) на крошечной случайной Qwen3: история, батч, encoder-decoder и prompt,
FLOPs, обучение по этапам с LoRA и загрузка."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from recsys.twotower.data import QueryDataset, TokenizerSpec, build_examples  # noqa: E402
from recsys.twotower.lazy import LazyBuilder, lazy_flops, lazy_history, qwen_body  # noqa: E402
from recsys.twotower.model import build_model, trainable_state  # noqa: E402
from recsys.twotower.tags import TrackTags  # noqa: E402
from recsys.twotower.train import encode_examples, load_weights, train  # noqa: E402

N_ITEMS, VOCAB, N_LAYERS = 40, [8, 8, 8, 2], 3
DAY = 86400
TCFG = {
    "arch": "lazy",
    "lazy": {"mode": "encdec", "encoder_layers": 0, "cross_every": 1},
    "model": {"dim": 16, "precision": "fp32", "lora": {"r": 4, "alpha": 8, "dropout": 0.0}},
    "codes": {"init": "centroids"},
    "tags": {"dim": 8},
    "input": {"history_len": 6, "history_dropout": 0.5, "max_query_tokens": 12, "instruction": "find"},
    "loss": {"negatives": "full", "mask_same_song": True, "temperature": 0.1},
    "train": {"batch_size": 8, "num_workers": 0, "val_queries": 16, "eval_every": 0, "seed": 0,
              "stages": [{"name": "A", "epochs": 0.5, "lora": False, "lr": 1e-2, "lr_copied": 1e-3},
                         {"name": "B", "epochs": 0.5, "lora": True, "lr": 1e-2, "lr_lora": 1e-3, "lr_copied": 1e-3}]},
    "encode": {"batch_size": 16},
}


def _spec():
    return TokenizerSpec(lambda texts: [[10 + ord(c) % 200 for c in t] for t in texts], eos_id=1, pad_id=0)


def _base():
    cfg = transformers.Qwen3Config(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=N_LAYERS,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                   max_position_embeddings=512, attn_implementation="eager")
    torch.manual_seed(0)
    return transformers.AutoModel.from_config(cfg)


def _model(E=0, mode="encdec", cross_every=1, text=True, tags=True):
    rng = np.random.default_rng(0)
    codes = np.c_[rng.integers(0, 8, (N_ITEMS, 3)), np.zeros(N_ITEMS, int)]
    codebooks = rng.normal(size=(3, 8, 6)).astype(np.float32)
    groups = np.full(N_ITEMS, -1)
    groups[[2, 3]] = 0
    tt = TrackTags(["<no_tags>", "rock", "pop"], rng.integers(0, 3, (N_ITEMS, 2)),
                   np.full((N_ITEMS, 2), 0.5, dtype=np.float32)) if tags else None
    item_text = rng.normal(size=(N_ITEMS, 10)).astype(np.float32) if text else None
    tcfg = {**TCFG, "lazy": {"mode": mode, "encoder_layers": E, "cross_every": cross_every}}
    return build_model(tcfg, _base(), codes, VOCAB, codebooks, song_groups=groups, tags=tt, item_text=item_text)


def _hist(items):
    return np.array([[i, 0, 0] for i in items], dtype=np.int64).reshape(-1, 3)


def test_lazy_history_decay_order_and_buckets():
    pos = {"a": 0, "b": 1, "c": 2, "d": 3}
    end = 1000 * DAY
    # a — 8 раз год назад, b — раз вчера, c — 3 раза неделю назад, x — без кода
    h = ["a"] * 8 + ["c"] * 3 + ["x", "b"]
    ts = [end - 365 * DAY] * 8 + [end - 7 * DAY] * 3 + [end - 2 * DAY, end - DAY]
    out = lazy_history(h, ts, pos, n=2, cutoff=end, decay_days=180)
    # скор: c 3·e^(-7/180) ≈ 2.9 > a 8·e^(-365/180) ≈ 1.05 > b ≈ 0.99 -> c, a; строки от старых к новым
    assert out[:, 0].tolist() == [0, 2]
    assert out[:, 1].tolist() == [3, 1]                    # корзины log2: 8 -> 3, 3 -> 1
    assert out[:, 2].tolist() == [8, 3]                    # давность: 366 дн -> 8, 8 дн -> 3
    assert lazy_history(h, ts, pos, n=2, cutoff=end, order="recent")[:, 0].tolist() == [2, 1]
    assert lazy_history([], [], pos, n=5).shape == (0, 3)


def test_collate_left_padding():
    b = LazyBuilder(_spec(), "find")
    batch = b.collate([(_hist([3, 5]), [11, 12], 3), (_hist([]), [11], 4)])
    assert batch["hist"].tolist() == [[3, 5], [-1, -1]] and batch["hmask"].tolist() == [[1, 1], [0, 0]]
    assert (batch["tok"][:, -1] == 1).all() and batch["mask"][1, 0] == 0 and batch["mask"][0].all()
    assert batch["target"].tolist() == [3, 4]


def _plain(model, batch, with_history):
    """Qwen из transformers: [история, текст] (with_history) или только текст -> u."""
    user, qm = model.user, qwen_body(model.user.base)
    x, m = qm.embed_tokens(batch["tok"]), batch["mask"]
    if with_history:
        x = torch.cat([user.history_tokens(batch, model.track_parts()[0]), x], dim=1)
        m = torch.cat([batch["hmask"], m], dim=1)
    h = qm(inputs_embeds=x, attention_mask=m, position_ids=(m.cumsum(1) - 1).clamp_min(0)).last_hidden_state[:, -1]
    return torch.nn.functional.normalize(user.proj(h), dim=-1)


def _batch():
    return LazyBuilder(_spec(), "find").collate([(_hist([3, 5, 7]), [11, 12], 3), (_hist([9]), [13, 14, 15], 4)])


def test_prompt_mode_equals_plain_decoder():
    """mode prompt — тот же результат, что Qwen из transformers на [история, текст]."""
    model, batch = _model(mode="prompt").eval(), _batch()
    with torch.no_grad():
        assert torch.allclose(model.encode_users(batch), _plain(model, batch, True), atol=1e-5)
    assert len(model.user.xattn) == 0 and len(model.user.encoder) == 0


def test_encdec_starts_as_plain_qwen_on_text():
    """Гейты cross-attention = 0: декодер — исходная Qwen по тексту, история не влияет (при любом энкодере)."""
    for E in (0, 2):
        model, batch = _model(E).eval(), _batch()
        assert len(model.user.encoder) == E and len(model.user.xattn) == N_LAYERS
        with torch.no_grad():
            assert torch.allclose(model.encode_users(batch), _plain(model, batch, False), atol=1e-5)


def test_encdec_reads_history():
    b = LazyBuilder(_spec(), "find")
    for E in (0, 1):
        model = _model(E).eval()
        with torch.no_grad():
            for blk in model.user.xattn.values():
                blk.gate.fill_(0.5)
            u1 = model.encode_users(b.collate([(_hist([3, 5]), [11, 12], 3)]))
            u2 = model.encode_users(b.collate([(_hist([6, 8]), [11, 12], 3)]))
            u0 = model.encode_users(b.collate([(_hist([]), [11, 12], 3)]))   # только пустой слот
        assert not torch.allclose(u1, u2) and torch.isfinite(u0).all()
    assert len(_model(cross_every=2).user.xattn) == 2                        # перед слоями 0 и 2


def test_flops_match_counter():
    """Линейные слои Qwen, энкодера и cross-attention — как у FlopCounterMode (aten.mm; sdpa с маской на CPU он
    не считает, проекции трека и выхода со смещением — aten.addmm, вне формулы)."""
    from torch.utils.flop_counter import FlopCounterMode
    b = LazyBuilder(_spec(), "find")
    L = 5
    batch = b.collate([(_hist(list(range(L))), [11, 12, 13], 3)])
    T = batch["tok"].shape[1]
    for mode, E, every in (("encdec", 0, 1), ("encdec", 2, 1), ("encdec", 1, 2), ("prompt", 0, 1)):
        model = _model(E, mode, every).eval()
        with torch.no_grad():
            reps = model.track_parts()[0]
            counter = FlopCounterMode(display=False)
            with counter:
                model.user(batch, reps)
        mm = sum(v for k, v in counter.get_flop_counts()["Global"].items() if str(k) == "aten.mm")
        f = lazy_flops(qwen_body(model.user.base).config, L, T, mode, E, every)
        assert mm == f["total"] - f["attention"], (mode, E, every)


def _examples(n_users=12, per_user=6, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"t{i}" for i in range(N_ITEMS)]
    users = pd.DataFrame({"user_id": range(n_users),
                          "history": [list(rng.choice(ids, 15)) for _ in range(n_users)],
                          "history_ts": [list(np.sort(rng.integers(0, 100 * DAY, 15))) for _ in range(n_users)]})
    q = pd.DataFrame({"query_id": [f"q{i}" for i in range(n_users * per_user)],
                      "user_id": np.repeat(range(n_users), per_user),
                      "query": [f"song {i % 7}" for i in range(n_users * per_user)],
                      "query_type": ["mood", "genre"] * (n_users * per_user // 2)})
    qrels = pd.DataFrame({"query_id": q["query_id"], "target_m4a_id": rng.choice(ids, len(q))})
    pos = {t: i for i, t in enumerate(ids)}
    return build_examples({"users": users, "queries": q, "qrels": qrels}, pos, _spec(), max_query_tokens=12,
                          make_history=lambda r: lazy_history(r["history"], r["history_ts"], pos, 6, 100 * DAY))


def test_train_stages_save_and_load(tmp_path):
    ex = _examples()
    assert ex.histories[0].shape == (6, 3) and len(QueryDataset(ex, history_len=2)[0][0]) == 2
    tr, val = ex.subset(np.arange(0, 60)), ex.subset(np.arange(60, 72))
    b = LazyBuilder(_spec(), "find")
    model = _model(E=1)
    out = model.loss(b.collate([QueryDataset(tr)[i] for i in range(4)]))
    out["loss"].backward()
    assert all(p.grad is None for p in model.user.base.parameters())             # Qwen заморожен
    assert all(blk.gate.grad is not None for blk in model.user.xattn.values())   # гейты учатся с первого шага
    copied = list(model.user.encoder.parameters()) + list(model.user.xattn.parameters())
    assert all(p.requires_grad for p in copied)                                  # копии слоёв Qwen учатся
    assert model.item.text_proj.weight.grad is not None
    model.zero_grad()
    tcfg = {**TCFG, "lazy": {"mode": "encdec", "encoder_layers": 1, "cross_every": 1}}
    best = train(model, tr, val, b, tcfg, str(tmp_path), torch.device("cpu"), log=lambda s: None)
    assert any(abs(blk.gate.item()) > 0 for blk in model.user.xattn.values())
    assert 0.0 <= best["recall@100"] <= 1.0 and (tmp_path / "model.pt").exists()
    assert any("lora_" in k for k in trainable_state(model))
    u = encode_examples(model, val, b, torch.device("cpu"))
    assert any(k.startswith("user.encoder.") for k in trainable_state(model))   # копии слоёв сохраняются
    fresh = _model(E=1)
    load_weights(fresh, str(tmp_path / "last.pt"))
    assert np.allclose(encode_examples(fresh, val, b, torch.device("cpu")), u, atol=1e-5)
