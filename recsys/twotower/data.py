"""Данные двухбашенного кандгена: запрос с историей -> последовательность для пользовательской башни.

Пример = (последние history_len уникальных треков пользователя, текст запроса, номер цели). Номер трека везде —
строка semantic_codes.npy (tracks_meta по m4a_id, recsys/semantic_ids.py).

Последовательность (паддинг слева: пулинг — по последнему токену):

    Instruct: <instruction>\\nQuery: History: [c1 c2 c3 c4] ... [c1 c2 c3 c4] Request: <запрос> <EOS>

Для каждой позиции в батче три числа: tok — id текстового токена (у позиций трека — pad), item — номер трека
(-1 у текста), level — позиция кода 0..3. Эмбеддинги собирает модель (model.py).
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

Tokenize = Callable[[List[str]], List[List[int]]]


class HFTokenize:
    """Тексты -> id без спецтокенов. Класс, а не lambda: DataLoader с num_workers на macOS передаёт collate в
    процессы через pickle."""

    def __init__(self, tok):
        self.tok = tok

    def __call__(self, texts: List[str]) -> List[List[int]]:
        return self.tok(list(texts), add_special_tokens=False)["input_ids"]


@dataclass
class TokenizerSpec:
    """То, что нужно от токенайзера: tokenize (тексты -> id без спецтокенов), eos (пулинг), pad."""
    tokenize: Tokenize
    eos_id: int
    pad_id: int

    @classmethod
    def from_hf(cls, tok) -> "TokenizerSpec":
        """Токенайзер transformers. EOS — <|endoftext|>, как у Qwen3-Embedding (иначе tok.eos_token_id)."""
        eos = tok.convert_tokens_to_ids("<|endoftext|>")
        if eos is None or eos == tok.unk_token_id:
            eos = tok.eos_token_id
        pad = tok.pad_token_id if tok.pad_token_id is not None else eos
        return cls(HFTokenize(tok), int(eos), int(pad))


def recent_unique(history: Sequence[str], pos: Dict[str, int], n: int) -> np.ndarray:
    """Последние n уникальных треков истории (хронологической), от старых к новым; треки без кода пропускаются."""
    out: List[int] = []
    seen = set()
    for t in reversed(history):
        if t in seen:
            continue
        seen.add(t)
        p = pos.get(t)
        if p is not None:
            out.append(p)
            if len(out) == n:
                break
    return np.array(out[::-1], dtype=np.int32)


def is_val_user(user_id, share: float) -> bool:
    """Пользователь в валидации (детерминированно по хешу user_id): целиком, со всеми запросами."""
    return zlib.crc32(str(user_id).encode()) % 10_000 < share * 10_000


@dataclass
class Examples:
    """Запросы сплита: история — у пользователя (общая), запрос и цель — у каждого запроса."""
    histories: List[np.ndarray]     # по пользователю: номера треков, от старых к новым
    user: np.ndarray                # (n,) индекс в histories
    query_tokens: List[List[int]]   # (n,) токены запроса
    target: np.ndarray              # (n,) номер цели, -1 — нет (сплит без qrels)
    query_id: np.ndarray            # (n,) str
    query_type: np.ndarray          # (n,) str
    user_id: np.ndarray             # (n,) user_id запроса

    def __len__(self) -> int:
        return len(self.target)

    def subset(self, idx: np.ndarray) -> "Examples":
        idx = np.asarray(idx)
        return Examples(self.histories, self.user[idx], [self.query_tokens[i] for i in idx], self.target[idx],
                        self.query_id[idx], self.query_type[idx], self.user_id[idx])


def build_examples(tables: Dict[str, pd.DataFrame], pos: Dict[str, int], spec: TokenizerSpec,
                   history_len: int = 64, max_query_tokens: int = 96,
                   make_history: Optional[Callable[[pd.Series], np.ndarray]] = None) -> Examples:
    """tables — users, queries, qrels части (build.load_examples); pos — m4a_id -> номер трека (SemanticIDs._pos).
    make_history — строка пользователя -> история модели (arch: lazy — lazy.lazy_history); по умолчанию
    последние history_len уникальных треков."""
    users = tables["users"]
    uidx = {u: i for i, u in enumerate(users["user_id"])}
    if make_history is not None:
        histories = [make_history(row) for _, row in users.iterrows()]
    else:
        histories = [recent_unique(list(h), pos, history_len) for h in users["history"]]
    q = tables["queries"]
    q = q[q["user_id"].isin(uidx)].reset_index(drop=True)
    qrels = tables.get("qrels")
    if qrels is not None and len(qrels):
        tmap = dict(zip(qrels["query_id"], qrels["target_m4a_id"].astype(str)))
        target = np.array([pos.get(tmap.get(qid, ""), -1) for qid in q["query_id"]], dtype=np.int64)
    else:
        target = np.full(len(q), -1, dtype=np.int64)
    texts = [" " + t for t in q["query"].astype(str)]   # после «Request:» — через пробел
    tokens = [t[:max_query_tokens] for t in spec.tokenize(texts)] if len(q) else []
    return Examples(histories, q["user_id"].map(uidx).to_numpy(np.int64), tokens, target,
                    q["query_id"].to_numpy().astype(str), q["query_type"].to_numpy().astype(str),
                    q["user_id"].to_numpy())


class SequenceBuilder:
    """Склеивает инструкцию, историю (по n_codes позиций на трек) и запрос; паддинг слева."""

    def __init__(self, spec: TokenizerSpec, instruction: str, n_codes: int = 4):
        self.spec, self.n_codes = spec, n_codes
        self.prefix, self.middle = spec.tokenize([f"Instruct: {instruction}\nQuery: History:", " Request:"])

    def build(self, history: np.ndarray, query: List[int]):
        """-> tok, item, level (списки одной длины)."""
        n_hist = len(history) * self.n_codes
        tok = self.prefix + [self.spec.pad_id] * n_hist + self.middle + list(query) + [self.spec.eos_id]
        item = [-1] * len(self.prefix) + np.repeat(history, self.n_codes).tolist() + [-1] * (len(self.middle) + len(query) + 1)
        level = [0] * len(self.prefix) + list(range(self.n_codes)) * len(history) + [0] * (len(self.middle) + len(query) + 1)
        return tok, item, level

    def collate(self, batch) -> Dict[str, torch.Tensor]:
        """batch: [(history, query_tokens, target)] -> тензоры (B, T), паддинг слева."""
        seqs = [self.build(h, q) for h, q, _ in batch]
        T = max(len(s[0]) for s in seqs)
        B = len(seqs)
        tok = torch.full((B, T), self.spec.pad_id, dtype=torch.long)
        item = torch.full((B, T), -1, dtype=torch.long)
        level = torch.zeros((B, T), dtype=torch.long)
        mask = torch.zeros((B, T), dtype=torch.long)
        for i, (t, it, lv) in enumerate(seqs):
            n = len(t)
            tok[i, T - n:] = torch.tensor(t)
            item[i, T - n:] = torch.tensor(it)
            level[i, T - n:] = torch.tensor(lv)
            mask[i, T - n:] = 1
        target = torch.tensor([t for _, _, t in batch], dtype=torch.long)
        return {"tok": tok, "item": item, "level": level, "mask": mask, "target": target}


class QueryDataset(torch.utils.data.Dataset):
    """history_dropout (только train): с этой вероятностью история обрезается до случайной длины 0..len —
    модель работает при любой длине истории, и ablation по ней считается без переобучения."""

    def __init__(self, ex: Examples, history_dropout: float = 0.0, history_len: Optional[int] = None):
        self.ex, self.history_dropout, self.history_len = ex, history_dropout, history_len

    def __len__(self) -> int:
        return len(self.ex)

    def __getitem__(self, i: int):
        h = self.ex.histories[self.ex.user[i]]
        if self.history_len is not None:
            h = h[len(h) - min(self.history_len, len(h)):]
        if self.history_dropout and torch.rand(()).item() < self.history_dropout:
            k = int(torch.randint(0, len(h) + 1, ()).item())
            h = h[len(h) - k:]
        return h, self.ex.query_tokens[i], int(self.ex.target[i])
