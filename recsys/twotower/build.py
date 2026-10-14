"""Сборка двухбашенного кандгена из конфига: коды, примеры сплита, модель; загрузка обученной модели.

Общее для scripts/train_twotower.py, scripts/encode_twotower.py и онлайн-кодирования запроса (OnlineEncoder).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from recsys.config import deep_update
from recsys.data.crs import split_path
from recsys.data.dataset import Dataset
from recsys.data.parts import PARTS, part_split
from recsys.semantic_ids import SemanticIDs, load_semantic_ids, random_codes, song_groups
from recsys.twotower.data import Examples, SequenceBuilder, TokenizerSpec, build_examples, recent_unique
from recsys.twotower.model import TwoTower, build_model, load_base
from recsys.twotower.train import device_attention, device_precision, load_weights, pick_device


@dataclass
class Codes:
    sid: SemanticIDs            # m4a_id <-> номер трека, коды RQ-VAE
    codes: np.ndarray           # коды, которые видит модель (RQ-VAE или случайные)
    vocab_sizes: list
    codebooks: Optional[np.ndarray]


def load_codes(cfg: Dict[str, Any]) -> Codes:
    """twotower.codes.source: rqvae — semantic_codes.npy; random — случайные коды той же формы (тогда и
    инициализация случайная: центроиды к чужим кодам не относятся)."""
    ccfg = cfg.get("twotower", {}).get("codes", {})
    sid = load_semantic_ids(cfg, checkpoint=True)
    if ccfg.get("source", "rqvae") == "random":
        codes = random_codes(sid.codes, seed=ccfg.get("seed", 0))
        return Codes(sid, codes, [int(c.max()) + 1 for c in codes.T], None)
    return Codes(sid, sid.codes, sid.vocab_sizes, sid.codebooks)


def tracks_meta_path(cfg: Dict[str, Any]) -> str:
    return split_path(cfg["data"]["dir"], "tracks_meta")


def load_examples(cfg: Dict[str, Any], part: str, spec: TokenizerSpec, sid: SemanticIDs,
                  n_users: Optional[int] = None) -> Examples:
    """Запросы части датасета (recsys/data/parts.py) или всего сплита (part — имя сплита, например train).
    Обучение по умолчанию — на всём train; поэтому в признаки ранкера twotower не идёт (candgen.sources.twotower.
    features: false). n_users — столько случайных пользователей (быстрая проверка)."""
    icfg = cfg.get("twotower", {}).get("input", {})
    ds = Dataset(cfg, verbose=False)
    if part in PARTS:
        split, queries = part_split(part, cfg["data"]["parts"]), ds.part(part)
    else:
        split, queries = part, ds.table(part, "queries")
    if n_users is not None:
        users = queries["user_id"].drop_duplicates()
        users = users.sample(n=min(n_users, len(users)), random_state=cfg["data"]["parts"].get("seed", 42))
        queries = queries[queries["user_id"].isin(set(users))]
    qrels = ds.table(split, "qrels")
    tables = {"users": ds.users(split, queries["user_id"].unique()), "queries": queries,
              "qrels": qrels[qrels["query_id"].isin(set(queries["query_id"]))]}
    make_history = None
    if cfg.get("twotower", {}).get("arch") == "lazy":   # по числу прослушиваний с затуханием, с корзинами
        from recsys.twotower.lazy import lazy_history
        cutoff, n = ds.cutoff(split), icfg.get("history_len", 256)
        order, decay = icfg.get("history_order", "decay"), icfg.get("decay_days", 180.0)

        def make_history(row):
            return lazy_history(row["history"], row.get("history_ts"), sid._pos, n, cutoff, decay, order)
    return build_examples(tables, sid._pos, spec, icfg.get("history_len", 64), icfg.get("max_query_tokens", 96),
                          make_history=make_history)


def make_builder(cfg: Dict[str, Any], spec: TokenizerSpec, n_codes: int):
    instruction = cfg.get("twotower", {}).get("input", {}).get("instruction", "")
    if cfg.get("twotower", {}).get("arch") == "lazy":
        from recsys.twotower.lazy import LazyBuilder
        return LazyBuilder(spec, instruction)
    return SequenceBuilder(spec, instruction, n_codes)


def load_item_text(cfg: Dict[str, Any], codes: Codes) -> Optional[np.ndarray]:
    """Векторы названий треков (arch: lazy, twotower.item.text; scripts/encode_item_text.py) в порядке строк кодов."""
    t = cfg.get("twotower", {}).get("item", {}).get("text", {})
    if not t.get("enabled", False):
        return None
    path = t.get("path", "artifacts/item_text/qwen3emb.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: нет векторов названий — python scripts/encode_item_text.py "
                                f"(или --set twotower.item.text.enabled=false)")
    with np.load(path) as z:
        pos = {t: i for i, t in enumerate(z["ids"].astype(str))}
        missing = [t for t in codes.sid.track_ids if t not in pos]
        if missing:
            raise ValueError(f"{path}: нет {len(missing)} треков (например, {missing[:3]}) — пересчитайте")
        return z["vectors"][[pos[t] for t in codes.sid.track_ids]].astype(np.float32)


def load_tags(cfg: Dict[str, Any], codes: Codes):
    """Теги треков (arch: v2) из каталога (кэш data.cache_dir), в порядке строк кодов."""
    from recsys.twotower.tags import build_track_tags
    t = cfg.get("twotower", {}).get("tags", {})
    return build_track_tags(Dataset(cfg, verbose=False).catalog.df, codes.sid.track_ids, min_count=t.get("min_count", 5),
                            max_per_track=t.get("max_per_track", 32), genre_weight=t.get("genre_weight", 0.5))


def new_model(cfg: Dict[str, Any], codes: Codes, base=None, log_q: Optional[np.ndarray] = None,
              groups: Optional[np.ndarray] = None, precision: Optional[str] = None, attn: Optional[str] = None):
    """-> (модель, TokenizerSpec). base=None — Qwen из twotower.model.name в точности precision
    (по умолчанию twotower.model.precision)."""
    tcfg = cfg.get("twotower", {})
    mcfg = tcfg.get("model", {})
    tok = None
    if base is None:
        base, tok = load_base(mcfg.get("name", "Qwen/Qwen3-Embedding-0.6B"), precision or mcfg.get("precision", "bf16"),
                              attn or mcfg.get("attn_implementation", "sdpa"))
    if groups is None and tcfg.get("loss", {}).get("mask_same_song", True):
        groups = song_groups(tracks_meta_path(cfg), codes.sid.track_ids)
    tags = None
    if tcfg.get("arch", "v1") in ("v2", "lazy") and tcfg.get("tags", {}).get("enabled", True):
        tags = load_tags(cfg, codes)
    model = build_model(tcfg, base, codes.codes, codes.vocab_sizes, codes.codebooks, song_groups=groups, log_q=log_q,
                        tags=tags, item_text=load_item_text(cfg, codes) if tcfg.get("arch") == "lazy" else None)
    return model, (TokenizerSpec.from_hf(tok) if tok is not None else None)


def load_trained(cfg: Dict[str, Any], checkpoint: str, device: Optional[torch.device] = None,
                 item_text: Optional[str] = None):
    """Обученная модель из model.pt: конфиг twotower берётся из чекпоинта.
    item_text — путь к векторам названий (arch: lazy) вместо записанного в чекпоинте (другая папка на сервере).
    -> (модель, SequenceBuilder, Codes, раздел twotower из чекпоинта)."""
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tcfg = ck["twotower"]
    if item_text:
        tcfg = deep_update(tcfg, {"item": {"text": {"path": item_text}}})
    cfg = {**cfg, "twotower": tcfg}
    codes = load_codes(cfg)
    device = device or pick_device()
    precision = device_precision(device, cfg["twotower"].get("model", {}).get("precision", "bf16"))
    attn = device_attention(device, cfg["twotower"].get("model", {}).get("attn_implementation", "sdpa"))
    model, spec = new_model(cfg, codes, groups=np.full(len(codes.codes), -1), precision=precision, attn=attn)  # маска — в лоссе
    load_weights(model, checkpoint)
    model.to(device).eval()
    return model, make_builder(cfg, spec, codes.codes.shape[1]), codes, cfg["twotower"]


class OnlineEncoder:
    """Вектор u для нового запроса (реплика диалога, онлайн-инференс): история + последняя реплика -> u.
    История — как при кодировании частей (load_examples): v1 / v2 — последние history_len уникальных треков
    (Request.history), lazy — lazy_history по сырым прослушиваниям (Request.listens) до cutoff пользователя.
    r всех треков (v2, lazy) от запроса не зависит: считается один раз при загрузке."""

    def __init__(self, model: TwoTower, builder: SequenceBuilder, codes: Codes, tcfg: Dict[str, Any],
                 precision: str = "bf16"):
        self.model, self.builder, self.codes, self.tcfg, self.precision = model, builder, codes, tcfg, precision
        self.device = next(model.parameters()).device
        icfg = tcfg.get("input", {})
        self.history_len, self.max_query_tokens = icfg.get("history_len", 64), icfg.get("max_query_tokens", 96)
        self.parts = None
        if hasattr(model, "track_parts"):   # как в encode_examples: без autocast (внутри — fp32)
            with torch.no_grad():
                self.parts = model.track_parts()

    @classmethod
    def load(cls, cfg: Dict[str, Any], checkpoint: str, device: Optional[torch.device] = None,
             item_text: Optional[str] = None) -> "OnlineEncoder":
        model, builder, codes, tcfg = load_trained(cfg, checkpoint, device, item_text)
        precision = device_precision(next(model.parameters()).device, tcfg.get("model", {}).get("precision", "bf16"))
        return cls(model, builder, codes, tcfg, precision)

    def history(self, request, cutoff: Optional[int]) -> np.ndarray:
        pos = self.codes.sid._pos
        if self.tcfg.get("arch") == "lazy":
            from recsys.twotower.lazy import lazy_history
            if request.listens is None:
                raise ValueError("arch lazy: нужны сырые прослушивания (Request.listens: history, history_ts)")
            icfg = self.tcfg.get("input", {})
            return lazy_history(request.listens[0], request.listens[1], pos, self.history_len, cutoff,
                                icfg.get("decay_days", 180.0), icfg.get("history_order", "decay"))
        chrono = [h.track_id for h in reversed(request.history)]   # Request.history — уникальные, свежие первыми
        return recent_unique(chrono, pos, self.history_len)

    @torch.no_grad()
    def encode(self, ctx) -> np.ndarray:
        """ctx — recsys.schemas.Context: запрос и cutoff пользователя (ctx.user)."""
        from recsys.twotower.train import autocast, move
        request = ctx.request
        hist = self.history(request, ctx.user.cutoff if ctx.user is not None else request.meta.get("cutoff"))
        text = request.user_messages[-1] if request.user_messages else ""
        query = self.builder.spec.tokenize([" " + text])[0][:self.max_query_tokens]
        batch = move(self.builder.collate([(hist, query, -1)]), self.device)
        with autocast(self.device, self.precision):
            u = self.model.encode_users(batch, self.parts) if self.parts is not None else self.model.encode_users(batch)
            return u.float().cpu().numpy()[0]
