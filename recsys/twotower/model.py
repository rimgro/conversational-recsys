"""Двухбашенная модель: пользовательская башня (Qwen + токены кодов) и трековая башня (по кодам), лосс.

Номер трека — строка codes (semantic_codes.npy). Код трека — n_codes чисел; в таблицах эмбеддингов они
лежат подряд: номер строки = offsets[позиция] + код (256 + 256 + 256 + 9 строк при кодах RQ-VAE).

  пользовательская башня  inputs_embeds: текст — родной эмбеддер Qwen (заморожен), позиции треков —
                          SemanticTokenEmbedding; последний токен (EOS) -> Linear(hidden -> dim) -> L2  = u
  трековая башня          v = L2(MLP(сумма строк таблицы по кодам трека))
  скор                    u · v / τ, τ обучается

Лосс (loss.negatives): full — softmax по всем трекам каталога; in_batch — по целям батча (+ logQ).
Другие записи той же песни (song_groups) не считаются негативами (loss.mask_same_song).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def fp32(device: torch.device):
    """Без autocast: головы башен и логиты — в fp32 (Qwen при этом в bf16 / fp16). При τ = 0.05 и softmax
    по 64k треков точности bf16 (~3 знака) не хватает."""
    return torch.autocast(device.type, enabled=False)


def code_layout(vocab_sizes: Sequence[int], codebooks: Optional[np.ndarray]):
    """-> offsets (n_codes,), центроиды подряд (сумма vocab_sizes, d) — нули у позиций без кодбука
    (номер при коллизии), и есть ли у строки центроид."""
    offsets = np.r_[0, np.cumsum(vocab_sizes)[:-1]].astype(np.int64)
    total = int(np.sum(vocab_sizes))
    d = codebooks.shape[2] if codebooks is not None else 0
    flat = np.zeros((total, d), dtype=np.float32)
    has = np.zeros(total, dtype=bool)
    if codebooks is not None:
        for k in range(len(codebooks)):
            flat[offsets[k]:offsets[k] + codebooks.shape[1]] = codebooks[k]
            has[offsets[k]:offsets[k] + codebooks.shape[1]] = True
    return offsets, flat, has


class SemanticTokenEmbedding(nn.Module):
    """Код на позиции level -> вектор размера hidden (вход Qwen).

    init=centroids: E = P · центроид + Δ + level_emb, P (d -> hidden) общая, Δ — поправка на строку (с нуля);
    P подобрана так, чтобы средняя норма E совпала с нормой обычных токенов Qwen (target_norm): иначе токены
    треков на старте заглушат текст или утонут в нём. Без центроидов (номер при коллизии, init=random) —
    только Δ (случайно, с той же нормой) + level_emb.
    """

    def __init__(self, vocab_sizes: Sequence[int], hidden: int, codebooks: Optional[np.ndarray] = None,
                 target_norm: float = 1.0, init: str = "centroids"):
        super().__init__()
        use_cb = init == "centroids" and codebooks is not None
        offsets, flat, has = code_layout(vocab_sizes, codebooks if use_cb else None)
        self.register_buffer("offsets", torch.from_numpy(offsets), persistent=False)
        self.register_buffer("centroids", torch.from_numpy(flat), persistent=False)
        self.register_buffer("has_centroid", torch.from_numpy(has), persistent=False)
        total = int(np.sum(vocab_sizes))
        std = target_norm / math.sqrt(hidden)
        self.delta = nn.Embedding(total, hidden)
        self.level = nn.Embedding(len(vocab_sizes), hidden)
        nn.init.normal_(self.level.weight, std=0.1 * std)
        with torch.no_grad():
            self.delta.weight.normal_(std=std)
            self.delta.weight[self.has_centroid] = 0.0
        self.proj = None
        if use_cb:
            self.proj = nn.Linear(flat.shape[1], hidden, bias=False)
            with torch.no_grad():
                norm = self.proj(self.centroids[self.has_centroid]).norm(dim=1).mean()
                self.proj.weight.mul_(target_norm / norm.clamp_min(1e-8))

    def forward(self, level: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        idx = self.offsets[level] + code
        out = self.delta(idx) + self.level(level)
        if self.proj is not None:
            out = out + self.proj(self.centroids[idx]) * self.has_centroid[idx].unsqueeze(-1)
        return out


class ItemTower(nn.Module):
    """v = L2(MLP(сумма строк таблицы по кодам)). init=centroids: таблица = центроиды RQ-VAE (номер при коллизии —
    нули), т.е. на старте сумма — квантованный вектор трека из RQ-VAE."""

    def __init__(self, vocab_sizes: Sequence[int], dim: int, codebooks: Optional[np.ndarray] = None,
                 init: str = "centroids", d_in: int = 32):
        super().__init__()
        use_cb = init == "centroids" and codebooks is not None
        offsets, flat, _ = code_layout(vocab_sizes, codebooks if use_cb else None)
        self.register_buffer("offsets", torch.from_numpy(offsets), persistent=False)
        d_in = flat.shape[1] if use_cb else d_in
        self.table = nn.Embedding(int(np.sum(vocab_sizes)), d_in)
        with torch.no_grad():
            if use_cb:
                self.table.weight.copy_(torch.from_numpy(flat))
            else:
                self.table.weight.normal_(std=1.0 / math.sqrt(d_in))
        self.mlp = nn.Sequential(nn.Linear(d_in, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """codes (n, n_codes) -> (n, dim), норма 1."""
        x = self.table(codes + self.offsets).sum(dim=1)
        return F.normalize(self.mlp(x), dim=-1)


class UserTower(nn.Module):
    """Qwen (base: transformers AutoModel, веса заморожены; LoRA добавляет add_lora) + токены кодов + проекция."""

    def __init__(self, base: nn.Module, sem: SemanticTokenEmbedding, dim: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.sem = sem
        self.proj = nn.Linear(base.config.hidden_size, dim)

    @property
    def dtype(self) -> torch.dtype:
        return self.base.get_input_embeddings().weight.dtype

    def forward(self, batch: Dict[str, torch.Tensor], codes: torch.Tensor) -> torch.Tensor:
        tok, item, level = batch["tok"], batch["item"], batch["level"]
        text = self.base.get_input_embeddings()(tok)
        is_track = item >= 0
        code = codes[item.clamp_min(0), level]
        x = torch.where(is_track.unsqueeze(-1), self.sem(level, code).to(text.dtype), text)
        h = self.base(inputs_embeds=x, attention_mask=batch["mask"]).last_hidden_state[:, -1]  # паддинг слева
        with fp32(h.device):
            return F.normalize(self.proj(h.float()), dim=-1)


class TwoTower(nn.Module):
    def __init__(self, user: UserTower, item: ItemTower, codes: np.ndarray, song_groups: Optional[np.ndarray] = None,
                 temperature: float = 0.05, learn_temperature: bool = True, log_q: Optional[np.ndarray] = None):
        super().__init__()
        self.user, self.item = user, item
        self.register_buffer("codes", torch.as_tensor(codes, dtype=torch.long), persistent=False)
        groups = song_groups if song_groups is not None else np.full(len(codes), -1)
        self.register_buffer("song_groups", torch.as_tensor(groups, dtype=torch.long), persistent=False)
        lq = log_q if log_q is not None else np.zeros(len(codes))
        self.register_buffer("log_q", torch.as_tensor(lq, dtype=torch.float32), persistent=False)
        self.log_tau = nn.Parameter(torch.tensor(math.log(temperature)), requires_grad=learn_temperature)

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp().clamp_min(0.01)

    def encode_users(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.user(batch, self.codes)

    def encode_items(self, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        with fp32(self.codes.device):
            return self.item(self.codes if idx is None else self.codes[idx])

    def logits(self, u: torch.Tensor, target: torch.Tensor, negatives: str = "full", logq: bool = False,
               mask_same_song: bool = True):
        """-> (логиты, метки). full: (B, n_tracks), метка — номер цели; in_batch: (B, B), столбец j — цель
        примера j, метка — i. Не негативы (-inf): другие записи той же песни; в in_batch ещё та же цель у другого
        примера."""
        with fp32(u.device):
            return self._logits(u.float(), target, negatives, logq, mask_same_song)

    def _logits(self, u, target, negatives, logq, mask_same_song):
        g = self.song_groups
        if negatives == "full":
            logits = u @ self.encode_items().T / self.tau
            labels = target
            if mask_same_song:
                gt = g[target]
                same = (g.unsqueeze(0) == gt.unsqueeze(1)) & (gt >= 0).unsqueeze(1)
                same[torch.arange(len(target), device=target.device), target] = False
                logits = logits.masked_fill(same, float("-inf"))
        elif negatives == "in_batch":
            logits = u @ self.encode_items(target).T / self.tau
            labels = torch.arange(len(target), device=target.device)
            if logq:
                logits = logits - self.log_q[target].unsqueeze(0)
            dup = target.unsqueeze(0) == target.unsqueeze(1)
            if mask_same_song:
                gt = g[target]
                dup = dup | ((gt.unsqueeze(0) == gt.unsqueeze(1)) & (gt >= 0).unsqueeze(0))
            dup.fill_diagonal_(False)
            logits = logits.masked_fill(dup, float("-inf"))
        else:
            raise ValueError(f"loss.negatives: full | in_batch, а не {negatives}")
        return logits, labels

    def loss(self, batch: Dict[str, torch.Tensor], negatives: str = "full", logq: bool = False,
             mask_same_song: bool = True) -> Dict[str, torch.Tensor]:
        logits, labels = self.logits(self.encode_users(batch), batch["target"], negatives, logq, mask_same_song)
        return {"loss": F.cross_entropy(logits.float(), labels), "tau": self.tau.detach()}


def input_embedding_norm(base: nn.Module, n: int = 20_000, seed: int = 0) -> float:
    """Средняя норма строк эмбеддера Qwen (по случайной выборке): масштаб для токенов треков."""
    w = base.get_input_embeddings().weight.detach()
    idx = torch.randperm(w.shape[0], generator=torch.Generator().manual_seed(seed))[:n]
    return float(w[idx.to(w.device)].float().norm(dim=1).mean())


def load_base(name: str, precision: str = "bf16", attn_implementation: Optional[str] = "sdpa"):
    """Qwen из Hugging Face (AutoModel без lm_head) и токенайзер (паддинг слева)."""
    from transformers import AutoModel, AutoTokenizer
    kwargs: Dict[str, Any] = {"torch_dtype": DTYPES[precision]}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    base = AutoModel.from_pretrained(name, **kwargs)
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    return base, tok


def build_model(tcfg: Dict[str, Any], base: nn.Module, codes: np.ndarray, vocab_sizes: Sequence[int],
                codebooks: Optional[np.ndarray] = None, song_groups: Optional[np.ndarray] = None,
                log_q: Optional[np.ndarray] = None) -> TwoTower:
    """tcfg — раздел twotower конфига; base — Qwen (load_base) или маленькая модель в тестах."""
    init = tcfg.get("codes", {}).get("init", "centroids")
    dim = tcfg.get("model", {}).get("dim", 256)
    lcfg = tcfg.get("loss", {})
    sem = SemanticTokenEmbedding(vocab_sizes, base.config.hidden_size, codebooks,
                                 target_norm=input_embedding_norm(base), init=init)
    return TwoTower(UserTower(base, sem, dim), ItemTower(vocab_sizes, dim, codebooks, init=init), codes,
                    song_groups=song_groups, temperature=lcfg.get("temperature", 0.05),
                    learn_temperature=lcfg.get("learn_temperature", True), log_q=log_q)


def add_lora(model: TwoTower, lora_cfg: Dict[str, Any]) -> None:
    """LoRA на attention Qwen (peft): обучаются только адаптеры, веса Qwen остаются замороженными."""
    from peft import LoraConfig, get_peft_model
    if hasattr(model.user.base, "peft_config"):
        return
    cfg = LoraConfig(r=lora_cfg.get("r", 16), lora_alpha=lora_cfg.get("alpha", 32),
                     lora_dropout=lora_cfg.get("dropout", 0.05),
                     target_modules=list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
                     task_type="FEATURE_EXTRACTION")
    model.user.base = get_peft_model(model.user.base, cfg)


def trainable_state(model: TwoTower) -> Dict[str, torch.Tensor]:
    """Всё, что обучается (таблицы, проекции, башня треков, τ, LoRA), без замороженных весов Qwen."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if not k.startswith("user.base.") or "lora_" in k}
