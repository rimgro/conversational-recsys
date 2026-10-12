"""Сборка двухбашенного кандгена из конфига: коды, примеры сплита, модель; загрузка обученной модели.

Общее для scripts/train_twotower.py, scripts/encode_twotower.py и онлайн-кодирования запроса (OnlineEncoder).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from recsys.data.crs import read_split, split_path
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
    return split_path(cfg["data"]["crs"]["dir"], "tracks_meta")


def load_examples(cfg: Dict[str, Any], split: str, spec: TokenizerSpec, sid: SemanticIDs,
                  n_users: Optional[int] = None) -> Examples:
    icfg = cfg.get("twotower", {}).get("input", {})
    tables = read_split(cfg["data"]["crs"]["dir"], split, n_users=n_users, seed=cfg["data"]["crs"].get("seed", 42))
    return build_examples(tables, sid._pos, spec, icfg.get("history_len", 64), icfg.get("max_query_tokens", 96))


def make_builder(cfg: Dict[str, Any], spec: TokenizerSpec, n_codes: int) -> SequenceBuilder:
    return SequenceBuilder(spec, cfg.get("twotower", {}).get("input", {}).get("instruction", ""), n_codes)


def load_tags(cfg: Dict[str, Any], codes: Codes):
    """Теги треков (arch: v2) из каталога (кэш data.cache_dir), в порядке строк кодов."""
    from recsys.data.load import load_catalog
    from recsys.twotower.tags import build_track_tags
    t = cfg.get("twotower", {}).get("tags", {})
    return build_track_tags(load_catalog(cfg).df, codes.sid.track_ids, min_count=t.get("min_count", 5),
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
    if tcfg.get("arch", "v1") == "v2" and tcfg.get("tags", {}).get("enabled", True):
        tags = load_tags(cfg, codes)
    model = build_model(tcfg, base, codes.codes, codes.vocab_sizes, codes.codebooks, song_groups=groups, log_q=log_q,
                        tags=tags)
    return model, (TokenizerSpec.from_hf(tok) if tok is not None else None)


def load_trained(cfg: Dict[str, Any], checkpoint: str, device: Optional[torch.device] = None):
    """Обученная модель из model.pt: конфиг twotower берётся из чекпоинта.
    -> (модель, SequenceBuilder, Codes, раздел twotower из чекпоинта)."""
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = {**cfg, "twotower": ck["twotower"]}
    codes = load_codes(cfg)
    device = device or pick_device()
    precision = device_precision(device, cfg["twotower"].get("model", {}).get("precision", "bf16"))
    attn = device_attention(device, cfg["twotower"].get("model", {}).get("attn_implementation", "sdpa"))
    model, spec = new_model(cfg, codes, groups=np.full(len(codes.codes), -1), precision=precision, attn=attn)  # маска — в лоссе
    load_weights(model, checkpoint)
    model.to(device).eval()
    return model, make_builder(cfg, spec, codes.codes.shape[1]), codes, cfg["twotower"]


class OnlineEncoder:
    """Вектор u для нового запроса (демо, inference.ipynb): Request -> история + последняя реплика -> u."""

    def __init__(self, model: TwoTower, builder: SequenceBuilder, codes: Codes, history_len: int = 64,
                 max_query_tokens: int = 96, precision: str = "bf16"):
        self.model, self.builder, self.codes = model, builder, codes
        self.history_len, self.max_query_tokens, self.precision = history_len, max_query_tokens, precision

    @classmethod
    def load(cls, cfg: Dict[str, Any], checkpoint: str, device: Optional[torch.device] = None) -> "OnlineEncoder":
        model, builder, codes, tcfg = load_trained(cfg, checkpoint, device)
        icfg = tcfg.get("input", {})
        precision = device_precision(next(model.parameters()).device, tcfg.get("model", {}).get("precision", "bf16"))
        return cls(model, builder, codes, icfg.get("history_len", 64), icfg.get("max_query_tokens", 96), precision)

    @torch.no_grad()
    def encode(self, request) -> np.ndarray:
        from recsys.twotower.train import autocast, move
        chrono = [h.track_id for h in reversed(request.history)]   # Request.history — уникальные, свежие первыми
        hist = recent_unique(chrono, self.codes.sid._pos, self.history_len)
        text = request.user_messages[-1] if request.user_messages else ""
        query = self.builder.spec.tokenize([" " + text])[0][:self.max_query_tokens]
        device = next(self.model.parameters()).device
        batch = move(self.builder.collate([(hist, query, -1)]), device)
        with autocast(device, self.precision):
            return self.model.encode_users(batch).float().cpu().numpy()[0]
