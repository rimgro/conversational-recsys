"""Двухбашенная модель: пользовательская башня (Qwen + токены кодов) и трековая башня (по кодам), лосс.

Номер трека — строка codes (semantic_codes.npy). Код трека — n_codes чисел; в таблицах эмбеддингов они
лежат подряд: номер строки = offsets[позиция] + код (256 + 256 + 256 + 9 строк при кодах RQ-VAE).

  пользовательская башня  inputs_embeds: текст — родной эмбеддер Qwen (заморожен), позиции треков —
                          SemanticTokenEmbedding; последний токен (EOS) -> Linear(hidden -> dim) -> L2  = u
  трековая башня          v = L2(MLP(сумма строк таблицы по кодам трека))
  скор                    u · v / τ, τ обучается

Лосс (loss.negatives): full — softmax по всем трекам каталога; in_batch — по целям батча (+ logQ).
Другие записи той же песни (song_groups) не считаются негативами (loss.mask_same_song).

Архитектура twotower.arch (у чекпоинтов без ключа — v1):
  v1  выше: таблицы кодов входа и трековой башни — разные, трек описывается только кодами.
  v2  трек = коды RQ-VAE + теги, таблицы общие для обеих башен:
        токен трека на входе Qwen = E_k[код_k] + Linear(теги трека)        (4 токена на трек, как в v1)
        r = MLP(concat(LN(среднее E_k[коды] -> 256), LN(теги -> 256)))     представление трека
        v = L2(Linear(r))                                                  трек-кандидат
      E_k — те же таблицы SemanticTokenEmbedding, теги — одна таблица TagTable (общие веса башен).
      shortcut: u = L2(Linear(h) + α · Σ_j a_j v_j): внимание запроса (ключи — r треков истории) к векторам
      v_j тех же треков-кандидатов и «пустой слот» (ничего из истории: запрос нового, пустая история). Трек из
      истории, подходящий под запрос, сдвигает u прямо к своему v, не через слои Qwen; читаются сами v_j, а не
      проекция с нуля — иначе на старте чтение истории — шум, и модель учится выключать его пустым слотом.
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
        row_level = np.repeat(np.arange(len(vocab_sizes)), vocab_sizes)
        self.register_buffer("row_level", torch.from_numpy(row_level), persistent=False)
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

    def table(self) -> torch.Tensor:
        """Все строки сразу (сумма vocab_sizes, hidden): то же, что forward для каждой (позиция, код)."""
        out = self.delta.weight + self.level(self.row_level)
        if self.proj is not None:
            out = out + self.proj(self.centroids) * self.has_centroid.unsqueeze(-1)
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
               mask_same_song: bool = True, items: Optional[torch.Tensor] = None):
        """-> (логиты, метки). full: (B, n_tracks), метка — номер цели; in_batch: (B, B), столбец j — цель
        примера j, метка — i. Не негативы (-inf): другие записи той же песни; в in_batch ещё та же цель у другого
        примера. items — готовые векторы всех треков (иначе считаются здесь)."""
        with fp32(u.device):
            return self._logits(u.float(), target, negatives, logq, mask_same_song, items)

    def _logits(self, u, target, negatives, logq, mask_same_song, items=None):
        g = self.song_groups
        if negatives == "full":
            logits = u @ (self.encode_items() if items is None else items).T / self.tau
            labels = target
            if mask_same_song:
                gt = g[target]
                same = (g.unsqueeze(0) == gt.unsqueeze(1)) & (gt >= 0).unsqueeze(1)
                same[torch.arange(len(target), device=target.device), target] = False
                logits = logits.masked_fill(same, float("-inf"))
        elif negatives == "in_batch":
            logits = u @ (self.encode_items(target) if items is None else items[target]).T / self.tau
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


# ---------------------------------------------------------------- arch: v2 (коды + теги, общие веса башен)

class TagTable(nn.Module):
    """Вектор тегов трека: взвешенная сумма эмбеддингов его тегов (tags.py: idx, w — буферы на все треки)."""

    def __init__(self, idx: np.ndarray, w: np.ndarray, n_tags: int, dim: int = 256):
        super().__init__()
        self.register_buffer("idx", torch.as_tensor(idx, dtype=torch.long), persistent=False)
        self.register_buffer("w", torch.as_tensor(w, dtype=torch.float32), persistent=False)
        self.emb = nn.Embedding(n_tags + 1, dim, padding_idx=n_tags)   # последняя строка — пустое место (pad)
        nn.init.normal_(self.emb.weight, std=1.0 / math.sqrt(dim))
        with torch.no_grad():
            self.emb.weight[n_tags] = 0.0

    def forward(self, items: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(n, dim) для треков items (по умолчанию — все)."""
        idx, w = (self.idx, self.w) if items is None else (self.idx[items], self.w[items])
        return F.embedding_bag(idx, self.emb.weight, per_sample_weights=w, mode="sum",
                               padding_idx=self.emb.padding_idx)


class SharedItemTower(nn.Module):
    """r = MLP(concat(LN(коды), LN(теги))), v = L2(Linear(r)). Коды — строки той же SemanticTokenEmbedding, что
    на входе Qwen (общая таблица), спроецированные в d_part; теги — та же TagTable, что добавляется к токенам."""

    def __init__(self, sem: SemanticTokenEmbedding, tags: Optional[TagTable], hidden: int, dim: int = 256,
                 d_part: int = 256):
        super().__init__()
        self.sem, self.tags = sem, tags
        self.code_proj = nn.Linear(hidden, d_part)
        self.code_ln = nn.LayerNorm(d_part)
        d_in = d_part
        if tags is not None:
            self.tag_ln = nn.LayerNorm(tags.emb.embedding_dim)
            d_in += tags.emb.embedding_dim
        self.mlp = nn.Sequential(nn.Linear(d_in, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.head = nn.Linear(dim, dim)

    def reps(self, codes: torch.Tensor, tag_vec: Optional[torch.Tensor]) -> torch.Tensor:
        """codes (n, n_codes) — коды треков; tag_vec (n, d_tag) -> r (n, dim)."""
        rows = codes + self.sem.offsets
        table = self.code_proj(self.sem.table())                              # (строк кодов, d_part)
        parts = [self.code_ln(F.embedding_bag(rows, table, mode="mean"))]
        if self.tags is not None:
            parts.append(self.tag_ln(tag_vec))
        return self.mlp(torch.cat(parts, dim=-1))

    def out(self, r: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(r), dim=-1)


class UserTowerV2(UserTower):
    """Токены треков = коды (общая таблица) + Linear(теги трека); u = L2(Linear(h) + shortcut по истории)."""

    def __init__(self, base: nn.Module, sem: SemanticTokenEmbedding, dim: int, tag_dim: Optional[int],
                 shortcut: bool = True):
        super().__init__(base, sem, dim)
        hidden = base.config.hidden_size
        self.tag_in = None
        if tag_dim:
            self.tag_in = nn.Linear(tag_dim, hidden)
            nn.init.zeros_(self.tag_in.weight)      # старт: токены как в v1, теги подмешиваются по мере обучения
            nn.init.zeros_(self.tag_in.bias)
        self.shortcut = shortcut
        if shortcut:
            self.q = nn.Linear(hidden, dim)
            self.k = nn.Linear(dim, dim)
            self.null_key = nn.Parameter(torch.zeros(dim))
            self.null_val = nn.Parameter(torch.zeros(dim))
            self.alpha = nn.Parameter(torch.tensor(1.0))      # вес прочитанного из истории относительно Linear(h)
        self.last_null: Optional[torch.Tensor] = None     # вес пустого слота у последнего батча (для разбора)

    def forward(self, batch: Dict[str, torch.Tensor], codes: torch.Tensor, reps: Optional[torch.Tensor] = None,
                tag_vec: Optional[torch.Tensor] = None, items: Optional[torch.Tensor] = None) -> torch.Tensor:
        """reps (n_tracks, dim) — r всех треков, tag_vec — их теги, items — их векторы-кандидаты v
        (TwoTowerV2.track_parts)."""
        tok, item, level = batch["tok"], batch["item"], batch["level"]
        text = self.base.get_input_embeddings()(tok)
        is_track = item >= 0
        it = item.clamp_min(0)
        x_track = self.sem.table()[self.sem.offsets[level] + codes[it, level]]
        if self.tag_in is not None:
            x_track = x_track + self.tag_in(tag_vec[it])
        x = torch.where(is_track.unsqueeze(-1), x_track.to(text.dtype), text)
        h = self.base(inputs_embeds=x, attention_mask=batch["mask"]).last_hidden_state[:, -1]
        with fp32(h.device):
            h = h.float()
            u = self.proj(h)
            if self.shortcut:
                u = u + self.alpha * self.history_read(h, reps[it], items[it], is_track & (level == 0))
            return F.normalize(u, dim=-1)

    def history_read(self, h: torch.Tensor, r: torch.Tensor, v: torch.Tensor, is_hist: torch.Tensor) -> torch.Tensor:
        """Внимание запроса к трекам истории (по одной позиции на трек; ключи — r, значения — v) + пустой слот
        -> (B, dim)."""
        d = r.shape[-1]
        q = self.q(h)                                                         # (B, dim)
        scores = torch.einsum("bd,btd->bt", q, self.k(r)) / math.sqrt(d)
        scores = scores.masked_fill(~is_hist, float("-inf"))
        null = (q @ self.null_key / math.sqrt(d)).unsqueeze(1)                # (B, 1)
        a = torch.softmax(torch.cat([scores, null], dim=1), dim=1)
        self.last_null = a[:, -1].detach()
        return torch.einsum("bt,btd->bd", a[:, :-1], v) + a[:, -1:] * self.null_val


class TwoTowerV2(TwoTower):
    """TwoTower с общими таблицами башен: r всех треков считается один раз на шаг и идёт и в выход (v),
    и в shortcut пользовательской башни."""

    def track_parts(self):
        """-> (r, теги или None, v) всех треков: один раз на шаг, общие для обеих башен."""
        with fp32(self.codes.device):
            tag_vec = self.item.tags() if self.item.tags is not None else None
            reps = self.item.reps(self.codes, tag_vec)
            return reps, tag_vec, self.item.out(reps)

    def encode_users(self, batch: Dict[str, torch.Tensor], parts=None) -> torch.Tensor:
        reps, tag_vec, items = parts if parts is not None else self.track_parts()
        return self.user(batch, self.codes, reps, tag_vec, items)

    def encode_items(self, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        v = self.track_parts()[2]
        return v if idx is None else v[idx]

    def loss(self, batch: Dict[str, torch.Tensor], negatives: str = "full", logq: bool = False,
             mask_same_song: bool = True) -> Dict[str, torch.Tensor]:
        parts = self.track_parts()
        u = self.encode_users(batch, parts)
        logits, labels = self.logits(u, batch["target"], negatives, logq, mask_same_song, items=parts[2])
        out = {"loss": F.cross_entropy(logits.float(), labels), "tau": self.tau.detach()}
        if self.user.last_null is not None:
            out["null"] = self.user.last_null.mean()
        return out


def input_embedding_norm(base: nn.Module, n: int = 20_000, seed: int = 0) -> float:
    """Средняя норма строк эмбеддера Qwen (по случайной выборке): масштаб для токенов треков."""
    w = base.get_input_embeddings().weight.detach()
    idx = torch.randperm(w.shape[0], generator=torch.Generator().manual_seed(seed))[:n]
    return float(w[idx.to(w.device)].float().norm(dim=1).mean())


def load_base(name: str, precision: str = "bf16", attn_implementation: Optional[str] = "sdpa"):
    """Qwen из Hugging Face (AutoModel без lm_head) и токенайзер (паддинг слева)."""
    from transformers import AutoModel, AutoTokenizer
    import transformers
    new_api = int(transformers.__version__.split(".")[0]) >= 5 or tuple(map(int, transformers.__version__.split(".")[:2])) >= (4, 56)
    kwargs: Dict[str, Any] = {"dtype" if new_api else "torch_dtype": DTYPES[precision]}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    base = AutoModel.from_pretrained(name, **kwargs)
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    return base, tok


def build_model(tcfg: Dict[str, Any], base: nn.Module, codes: np.ndarray, vocab_sizes: Sequence[int],
                codebooks: Optional[np.ndarray] = None, song_groups: Optional[np.ndarray] = None,
                log_q: Optional[np.ndarray] = None, tags=None, item_text: Optional[np.ndarray] = None) -> TwoTower:
    """tcfg — раздел twotower конфига; base — Qwen (load_base) или маленькая модель в тестах.
    tags — TrackTags (tags.py) для arch: v2 / lazy с тегами; item_text — векторы названий треков (arch: lazy)."""
    init = tcfg.get("codes", {}).get("init", "centroids")
    dim = tcfg.get("model", {}).get("dim", 256)
    lcfg = tcfg.get("loss", {})
    hidden = base.config.hidden_size
    sem = SemanticTokenEmbedding(vocab_sizes, hidden, codebooks, target_norm=input_embedding_norm(base), init=init)
    kw = dict(song_groups=song_groups, temperature=lcfg.get("temperature", 0.05),
              learn_temperature=lcfg.get("learn_temperature", True), log_q=log_q)
    arch = tcfg.get("arch", "v1")
    if arch == "lazy":
        from recsys.twotower.lazy import build_lazy
        return build_lazy(tcfg, base, codes, vocab_sizes, codebooks, tags=tags, item_text=item_text, **kw)
    if arch == "v1":
        return TwoTower(UserTower(base, sem, dim), ItemTower(vocab_sizes, dim, codebooks, init=init), codes, **kw)
    if arch != "v2":
        raise ValueError(f"twotower.arch: v1 | v2 | lazy, а не {arch}")
    tcfg_tags = tcfg.get("tags", {})
    table = None
    if tcfg_tags.get("enabled", True):
        if tags is None:
            raise ValueError("arch v2 с тегами: нужен tags (build_track_tags)")
        table = TagTable(tags.idx, tags.w, len(tags.vocab), tcfg_tags.get("dim", 256))
    user = UserTowerV2(base, sem, dim, table.emb.embedding_dim if table is not None else None,
                       shortcut=tcfg.get("shortcut", {}).get("enabled", True))
    item = SharedItemTower(sem, table, hidden, dim)
    return TwoTowerV2(user, item, codes, **kw)


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
