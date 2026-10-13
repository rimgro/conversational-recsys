"""Lazy-кандген (twotower.arch: lazy, docs/lazy.md): энкодер истории + декодер Qwen по запросу с cross-attention.

Один трек — один токен (не 4 токена кода, как в v1 / v2), поэтому история может быть длинной (256 треков).

  трек j:   r_j = MLP(concat(LN(MLP(concat E_k[код_k])), LN(теги), LN(Linear(текст «артист — название»))))
            v_j = L2(head(r_j))                                                        = кандидат
  история:  токен = LN(Linear(r_j) + E_pos[место по свежести] + E_count[прослушивания] + E_age[давность])

  lazy.mode: encdec (по умолчанию) — классический encoder-decoder:
    1. энкодер: encoder_layers слоёв (копии первых слоёв Qwen, обучаются целиком), только история,
       attention в обе стороны -> память M = [пустой слот, выход энкодера]  (encoder_layers = 0: M — сами токены)
    2. декодер: все слои Qwen, только текст «Instruct: … Query: <запрос> <EOS>»; перед слоем l (каждым
       cross_every-м) — cross-attention текста к M: x += tanh(g_l) · Attn(q: текст, k/v: M). Веса блока — копия
       self-attention слоя l, g_l = 0: на старте декодер — исходная Qwen.
  lazy.mode: prompt — decoder-only бейзлайн: история в промпте перед текстом, все слои Qwen.

  u = L2(Linear(h_EOS)), скор = u · v / τ, softmax по всему каталогу (как v2).

Текст трека (twotower.item.text) — векторы Qwen3-Embedding, посчитанные заранее (scripts/encode_item_text.py):
каталог целиком считается на каждом шаге (softmax по всем трекам), поэтому сама модель названия не кодирует.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from recsys.twotower.data import TokenizerSpec
from recsys.twotower.model import TagTable, TwoTower, code_layout, fp32, input_embedding_norm

N_BUCKETS = 16          # корзины log2: число прослушиваний 1, 2-3, 4-7, ...; давность в днях 0, 1, 2-3, ...
MODES = ("encdec", "prompt")


def log2_bucket(x: np.ndarray) -> np.ndarray:
    return np.minimum(np.floor(np.log2(np.maximum(x, 1.0))), N_BUCKETS - 1).astype(np.int64)


def lazy_history(history: Sequence[str], history_ts: Optional[Sequence[int]], pos: Dict[str, int], n: int,
                 cutoff: Optional[int] = None, decay_days: float = 180.0, order: str = "decay") -> np.ndarray:
    """История (прослушивания с повторами) -> (k <= n, 3) int64: номер трека, корзина числа прослушиваний,
    корзина давности последнего прослушивания (дни до cutoff). Отбор n треков: order=decay — по сумме
    exp(-давность / decay_days) по прослушиваниям (как кандген history), recent — последние. Строки — от старых
    к новым по последнему прослушиванию. Треки без кода пропускаются."""
    if order not in ("decay", "recent"):
        raise ValueError(f"twotower.input.history_order: decay | recent, а не {order}")
    if len(history) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    h = np.asarray(history).astype(str)
    ts = np.asarray(history_ts, dtype=np.int64) if history_ts is not None else np.arange(len(h), dtype=np.int64) * 86400
    end = int(cutoff) if cutoff is not None else int(ts.max()) + 1
    uniq, inv = np.unique(h, return_inverse=True)
    count = np.bincount(inv, minlength=len(uniq))
    score = np.bincount(inv, weights=np.exp(-(end - ts) / 86400.0 / decay_days), minlength=len(uniq))
    last = np.full(len(uniq), np.iinfo(np.int64).min)
    np.maximum.at(last, inv, ts)
    p = np.array([pos.get(t, -1) for t in uniq], dtype=np.int64)
    keep = np.flatnonzero(p >= 0)
    key = -score[keep] if order == "decay" else -last[keep].astype(np.float64)
    top = keep[np.argsort(key, kind="stable")[:n]]
    top = top[np.argsort(last[top], kind="stable")]
    age_days = np.maximum(end - last[top], 0) / 86400.0
    return np.stack([p[top], log2_bucket(count[top]), log2_bucket(age_days + 1.0)], axis=1)


class LazyBuilder:
    """Батч lazy: история (B, L) отдельно от текста (B, T); паддинг слева у обоих (пулинг — последний токен)."""

    def __init__(self, spec: TokenizerSpec, instruction: str):
        self.spec = spec
        self.prefix = spec.tokenize([f"Instruct: {instruction}\nQuery:"])[0]

    def collate(self, batch) -> Dict[str, torch.Tensor]:
        """batch: [(история (k, 3), токены запроса, цель)]."""
        B = len(batch)
        L = max(len(h) for h, _, _ in batch)
        texts = [self.prefix + list(q) + [self.spec.eos_id] for _, q, _ in batch]
        T = max(len(t) for t in texts)
        out = {"hist": torch.full((B, L), -1, dtype=torch.long), "hcount": torch.zeros((B, L), dtype=torch.long),
               "hage": torch.zeros((B, L), dtype=torch.long), "hmask": torch.zeros((B, L), dtype=torch.long),
               "tok": torch.full((B, T), self.spec.pad_id, dtype=torch.long),
               "mask": torch.zeros((B, T), dtype=torch.long)}
        for i, ((h, _, _), t) in enumerate(zip(batch, texts)):
            k = len(h)
            if k:
                h = torch.as_tensor(np.asarray(h), dtype=torch.long)
                out["hist"][i, L - k:], out["hcount"][i, L - k:], out["hage"][i, L - k:] = h[:, 0], h[:, 1], h[:, 2]
                out["hmask"][i, L - k:] = 1
            out["tok"][i, T - len(t):] = torch.tensor(t)
            out["mask"][i, T - len(t):] = 1
        out["target"] = torch.tensor([t for _, _, t in batch], dtype=torch.long)
        return out


class TrackEncoder(nn.Module):
    """Трек -> r (вход истории и кандидат) и v = L2(head(r)). Коды: таблица на каждую позицию кода (init=centroids —
    центроиды RQ-VAE), 4 строки склеиваются и идут в MLP; теги — TagTable; текст — готовые векторы (буфер)."""

    def __init__(self, vocab_sizes: Sequence[int], codebooks: Optional[np.ndarray], tags: Optional[TagTable],
                 text: Optional[np.ndarray], dim: int = 256, d_part: int = 256, d_code: int = 32,
                 init: str = "centroids"):
        super().__init__()
        use_cb = init == "centroids" and codebooks is not None
        offsets, flat, has = code_layout(vocab_sizes, codebooks if use_cb else None)
        self.register_buffer("offsets", torch.from_numpy(offsets), persistent=False)
        d_code = flat.shape[1] if use_cb else d_code
        self.code_table = nn.Embedding(int(np.sum(vocab_sizes)), d_code)
        with torch.no_grad():
            scale = float(np.linalg.norm(flat[has], axis=1).mean()) if use_cb else 1.0
            self.code_table.weight.normal_(std=scale / math.sqrt(d_code))
            if use_cb:
                self.code_table.weight[torch.from_numpy(has)] = torch.from_numpy(flat[has])
        self.code_mlp = nn.Sequential(nn.Linear(len(vocab_sizes) * d_code, d_part), nn.GELU(), nn.Linear(d_part, d_part))
        self.code_ln = nn.LayerNorm(d_part)
        d_in = d_part
        self.tags = tags
        if tags is not None:
            self.tag_ln = nn.LayerNorm(tags.emb.embedding_dim)
            d_in += tags.emb.embedding_dim
        self.text_proj = None
        if text is not None:
            self.register_buffer("text", torch.as_tensor(text, dtype=torch.float32), persistent=False)
            self.text_proj = nn.Linear(text.shape[1], d_part)
            self.text_ln = nn.LayerNorm(d_part)
            d_in += d_part
        self.mlp = nn.Sequential(nn.Linear(d_in, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.head = nn.Linear(dim, dim)

    def reps(self, codes: torch.Tensor) -> torch.Tensor:
        """codes (n_tracks, n_codes) — все треки каталога -> r (n_tracks, dim)."""
        parts = [self.code_ln(self.code_mlp(self.code_table(codes + self.offsets).flatten(1)))]
        if self.tags is not None:
            parts.append(self.tag_ln(self.tags()))
        if self.text_proj is not None:
            parts.append(self.text_ln(self.text_proj(self.text)))
        return self.mlp(torch.cat(parts, dim=-1))

    def out(self, r: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(r), dim=-1)


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x (B, heads, S, d); cos, sin (B, S, d) — как apply_rotary_pos_emb в transformers (rotate_half)."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin


def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, scale: float) -> torch.Tensor:
    """q (B, h, Tq, d), k / v (B, h_kv, Tk, d) — GQA: головы k / v повторяются до h. -> (B, Tq, h·d)."""
    rep = q.shape[1] // k.shape[1]
    if rep > 1:
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    return o.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


def layer_step(layer: nn.Module, x: torch.Tensor, cos, sin, mask: torch.Tensor) -> torch.Tensor:
    """Слой Qwen3 своими модулями (нормы, q/k/v/o, MLP; LoRA — если есть). mask (B, 1, S, S), True — можно смотреть."""
    att = layer.self_attn
    hd = att.head_dim
    h = layer.input_layernorm(x)
    B, S = h.shape[:2]
    q = _rope(att.q_norm(att.q_proj(h).view(B, S, -1, hd)).transpose(1, 2), cos, sin)
    k = _rope(att.k_norm(att.k_proj(h).view(B, S, -1, hd)).transpose(1, 2), cos, sin)
    v = att.v_proj(h).view(B, S, -1, hd).transpose(1, 2)
    x = x + att.o_proj(_attend(q, k, v, mask, att.scaling))
    return x + layer.mlp(layer.post_attention_layernorm(x))


class CrossAttention(nn.Module):
    """x + tanh(g) · Attn(q: текст, k / v: память истории). q/k/v/o и нормы голов — копия self-attention слоя Qwen,
    перед которым стоит блок; свои RMSNorm у текста и у памяти (старт — input_layernorm слоя). Без RoPE: порядок
    истории — в E_pos её токенов. g = 0: на старте блок ничего не меняет."""

    def __init__(self, layer: nn.Module):
        super().__init__()
        att = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            setattr(self, name, copy.deepcopy(getattr(att, name)).float())
        self.x_norm = copy.deepcopy(layer.input_layernorm).float()
        self.mem_norm = copy.deepcopy(layer.input_layernorm).float()
        self.head_dim, self.scaling = att.head_dim, att.scaling
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, mem: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x (B, T, H) текст, mem (B, M, H) память, mask (B, 1, 1, M)."""
        B, T, M, hd = x.shape[0], x.shape[1], mem.shape[1], self.head_dim
        m = self.mem_norm(mem)
        q = self.q_norm(self.q_proj(self.x_norm(x)).view(B, T, -1, hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(m).view(B, M, -1, hd)).transpose(1, 2)
        v = self.v_proj(m).view(B, M, -1, hd).transpose(1, 2)
        o = self.o_proj(_attend(q, k.to(q.dtype), v.to(q.dtype), mask, self.scaling))
        return x + (torch.tanh(self.gate) * o).to(x.dtype)


def qwen_body(base: nn.Module) -> nn.Module:
    """Qwen3Model (embed_tokens, layers, norm, rotary_emb) — и после get_peft_model (LoRA)."""
    return base.get_base_model() if hasattr(base, "get_base_model") else base


def _positions(valid: torch.Tensor) -> torch.Tensor:
    """Позиции без паддинга слева: 0, 1, ... по настоящим токенам."""
    return (valid.long().cumsum(1) - 1).clamp_min(0)


def _causal_mask(valid: torch.Tensor) -> torch.Tensor:
    """(B, S) -> (B, 1, S, S): causal + паддинг; диагональ всегда (у строк паддинга не пусто, иначе NaN)."""
    S = valid.shape[1]
    causal = torch.ones(S, S, dtype=torch.bool, device=valid.device).tril()
    eye = torch.eye(S, dtype=torch.bool, device=valid.device)
    return ((causal & valid.unsqueeze(1)) | eye).unsqueeze(1)


class LazyUserTower(nn.Module):
    """mode encdec: энкодер истории -> память; Qwen по тексту с cross-attention к памяти.
    mode prompt: история в промпте Qwen перед текстом (decoder-only)."""

    def __init__(self, base: nn.Module, rep_dim: int, dim: int, mode: str = "encdec", encoder_layers: int = 0,
                 cross_every: int = 1, history_len: int = 256, target_norm: float = 1.0):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"twotower.lazy.mode: {' | '.join(MODES)}, а не {mode}")
        self.base, self.mode = base, mode
        for p in self.base.parameters():
            p.requires_grad_(False)
        qm = qwen_body(base)
        hidden = base.config.hidden_size
        self.track_in = nn.Linear(rep_dim, hidden)
        self.pos_emb = nn.Embedding(history_len, hidden)      # 0 — самый свежий трек
        self.count_emb = nn.Embedding(N_BUCKETS, hidden)
        self.age_emb = nn.Embedding(N_BUCKETS, hidden)
        for e in (self.pos_emb, self.count_emb, self.age_emb):
            nn.init.normal_(e.weight, std=0.02)
        self.track_ln = nn.LayerNorm(hidden)            # норма токена истории = норме обычных токенов Qwen
        with torch.no_grad():
            self.track_ln.weight.fill_(target_norm / math.sqrt(hidden))
        self.proj = nn.Linear(hidden, dim)
        self.encoder = nn.ModuleList()
        self.xattn = nn.ModuleDict()
        if mode == "encdec":
            n = len(qm.layers)
            if not 0 <= encoder_layers <= n:
                raise ValueError(f"twotower.lazy.encoder_layers: 0..{n}, а не {encoder_layers}")
            # энкодер — копии первых слоёв Qwen (обучаются целиком), в fp32: обучаемые веса не в bf16
            self.encoder = nn.ModuleList(copy.deepcopy(qm.layers[i]).float() for i in range(encoder_layers))
            if encoder_layers:
                self.enc_norm = copy.deepcopy(qm.norm).float().requires_grad_(True)
            self.xattn = nn.ModuleDict({str(i): CrossAttention(qm.layers[i]) for i in range(0, n, max(1, cross_every))})
            self.null = nn.Parameter(torch.randn(hidden) * target_norm / math.sqrt(hidden))   # «ничего из истории»
            for p in list(self.encoder.parameters()) + list(self.xattn.parameters()):
                p.requires_grad_(True)                       # копии замороженной Qwen: иначе тоже заморожены

    @property
    def dtype(self) -> torch.dtype:
        return qwen_body(self.base).embed_tokens.weight.dtype

    def history_tokens(self, batch: Dict[str, torch.Tensor], reps: torch.Tensor) -> torch.Tensor:
        hm = batch["hmask"]
        rank = (hm.flip(1).cumsum(1).flip(1) - 1).clamp(0, self.pos_emb.num_embeddings - 1)   # 0 — самый свежий
        x = (self.track_in(reps[batch["hist"].clamp_min(0)]) + self.pos_emb(rank)
             + self.count_emb(batch["hcount"]) + self.age_emb(batch["hage"]))
        return self.track_ln(x)

    def _run(self, fn, *args):
        ckpt = self.training and getattr(qwen_body(self.base), "gradient_checkpointing", False)
        return checkpoint(fn, *args, use_reentrant=False) if ckpt else fn(*args)

    def memory(self, batch: Dict[str, torch.Tensor], reps: torch.Tensor):
        """encdec: история -> энкодер -> память (B, 1 + L, H) и её маска (B, 1, 1, 1 + L); первый слот — пустой."""
        qm = qwen_body(self.base)
        mem = self.history_tokens(batch, reps)
        hm = batch["hmask"].bool()
        if len(self.encoder) and hm.shape[1]:            # у всего батча пустая история — энкодеру нечего делать
            cos, sin = qm.rotary_emb(mem, _positions(hm))
            L = hm.shape[1]
            eye = torch.eye(L, dtype=torch.bool, device=mem.device)
            mask = (hm.unsqueeze(1) | eye).unsqueeze(1)              # в обе стороны, без паддинга
            for layer in self.encoder:
                mem = self._run(layer_step, layer, mem, cos, sin, mask)
            mem = self.enc_norm(mem)
        B = mem.shape[0]
        mem = torch.cat([self.null.to(mem.dtype).expand(B, 1, -1), mem], dim=1)
        mmask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=hm.device), hm], dim=1)[:, None, None, :]
        return mem, mmask

    def _decoder_step(self, i: int, layer: nn.Module, x, cos, sin, mask, mem, mmask):
        if str(i) in self.xattn:
            x = self.xattn[str(i)](x, mem, mmask)
        return layer_step(layer, x, cos, sin, mask)

    def forward(self, batch: Dict[str, torch.Tensor], reps: torch.Tensor) -> torch.Tensor:
        """reps (n_tracks, rep_dim) — r всех треков (LazyTwoTower.track_parts)."""
        qm = qwen_body(self.base)
        x = qm.embed_tokens(batch["tok"])
        if self.mode == "prompt":
            L = batch["hist"].shape[1]
            x = torch.cat([self.history_tokens(batch, reps).to(x.dtype), x], dim=1)
            valid = torch.cat([batch["hmask"], batch["mask"]], dim=1).bool()
            cos, sin = qm.rotary_emb(x, _positions(valid))
            mask = _causal_mask(valid)
            for layer in qm.layers:
                x = self._run(layer_step, layer, x, cos, sin, mask)
            x = x[:, L:]
        else:
            mem, mmask = self.memory(batch, reps)
            valid = batch["mask"].bool()
            cos, sin = qm.rotary_emb(x, _positions(valid))
            mask = _causal_mask(valid)
            for i, layer in enumerate(qm.layers):
                x = self._run(self._decoder_step, i, layer, x, cos, sin, mask, mem, mmask)
        h = qm.norm(x)[:, -1]
        with fp32(h.device):
            return F.normalize(self.proj(h.float()), dim=-1)


class LazyTwoTower(TwoTower):
    """TwoTower с TrackEncoder: r всех треков считается один раз на шаг — и для истории, и для кандидатов."""

    def track_parts(self):
        with fp32(self.codes.device):
            reps = self.item.reps(self.codes)
            return reps, self.item.out(reps)

    def encode_users(self, batch: Dict[str, torch.Tensor], parts=None) -> torch.Tensor:
        reps = (parts if parts is not None else self.track_parts())[0]
        return self.user(batch, reps)

    def encode_items(self, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        v = self.track_parts()[1]
        return v if idx is None else v[idx]

    def loss(self, batch: Dict[str, torch.Tensor], negatives: str = "full", logq: bool = False,
             mask_same_song: bool = True) -> Dict[str, torch.Tensor]:
        parts = self.track_parts()
        u = self.encode_users(batch, parts)
        logits, labels = self.logits(u, batch["target"], negatives, logq, mask_same_song, items=parts[1])
        out = {"loss": F.cross_entropy(logits.float(), labels), "tau": self.tau.detach()}
        if len(self.user.xattn):                         # средний |tanh(g)|: насколько декодер слушает историю
            out["gate"] = torch.stack([torch.tanh(b.gate).abs() for b in self.user.xattn.values()]).mean().detach()
        return out


def build_lazy(tcfg: Dict[str, Any], base: nn.Module, codes: np.ndarray, vocab_sizes: Sequence[int],
               codebooks: Optional[np.ndarray] = None, tags=None, item_text: Optional[np.ndarray] = None,
               **kw) -> LazyTwoTower:
    """tags — TrackTags (или None: без тегов), item_text — (n_tracks, d) векторы названий (или None)."""
    dim = tcfg.get("model", {}).get("dim", 256)
    lz = tcfg.get("lazy", {})
    table = None
    if tags is not None:
        table = TagTable(tags.idx, tags.w, len(tags.vocab), tcfg.get("tags", {}).get("dim", 256))
    item = TrackEncoder(vocab_sizes, codebooks, table, item_text, dim,
                        init=tcfg.get("codes", {}).get("init", "centroids"))
    user = LazyUserTower(base, dim, dim, lz.get("mode", "encdec"), lz.get("encoder_layers", 0),
                         lz.get("cross_every", 1), tcfg.get("input", {}).get("history_len", 256),
                         target_norm=input_embedding_norm(base))
    return LazyTwoTower(user, item, codes, **kw)


def lazy_flops(config, L: float, T: float, mode: str = "encdec", encoder_layers: int = 0,
               cross_every: int = 1) -> Dict[str, float]:
    """FLOPs на один запрос (прямой проход; умножение-сложение = 2) при истории L треков и тексте T токенов.
    config — config Qwen. Башня треков и проекции — копейки, не считаются.
      encoder    энкодер по истории (prompt: история через все слои Qwen) — зависит только от пользователя
      memory_kv  k / v памяти во всех блоках cross-attention — тоже только от пользователя
      query      декодер по тексту, его cross-attention к памяти (prompt: текст, смотрящий на историю и себя)
      total      всё вместе — один пример на обучении (backward ~ ещё 1-2 таких)
      attention  из total — q·k и веса·v (маскированное тоже считается: оно вычисляется), остальное — линейные слои
    Если хранить encoder + memory_kv на пользователя (cache_kv_bytes, fp16), онлайн — только query;
    prompt тоже так может (история перед текстом, causal), но кэш — k / v всех слоёв."""
    if mode not in MODES:
        raise ValueError(f"twotower.lazy.mode: {' | '.join(MODES)}, а не {mode}")
    H, I, n = config.hidden_size, config.intermediate_size, config.num_hidden_layers
    hd = getattr(config, "head_dim", None) or H // config.num_attention_heads
    nh, nkv = config.num_attention_heads, config.num_key_value_heads
    kv, qo, mlp = 2 * H * nkv * hd, 2 * H * nh * hd, 3 * H * I
    layer = kv + qo + mlp
    att = 4 * nh * hd                                   # на пару (строка запроса, ключ)
    if mode == "prompt":
        S = L + T
        encoder = n * (2 * L * layer + att * L * S)
        query = n * (2 * T * layer + att * T * S)
        return {"encoder": encoder, "memory_kv": 0.0, "query": query, "total": encoder + query,
                "attention": n * att * S * S, "cache_kv_bytes": 2.0 * n * L * kv / H}
    E, M = min(encoder_layers, n), L + 1
    nx = len(range(0, n, max(1, cross_every)))
    encoder = E * (2 * L * layer + att * L * L)
    memory_kv = nx * 2 * M * kv
    query = n * (2 * T * layer + att * T * T) + nx * (2 * T * qo + att * T * M)
    return {"encoder": float(encoder), "memory_kv": float(memory_kv), "query": float(query),
            "total": float(encoder + memory_kv + query), "attention": float(E * att * L * L + n * att * T * T + nx * att * T * M),
            "cache_kv_bytes": 2.0 * nx * M * kv / H}
