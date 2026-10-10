"""Semantic ID треков: коды RQ-VAE по MuQ-эмбеддингу и сам RQ-VAE (для двухбашенного кандгена).

Файлы — в папке semantic_ids.dir (artifacts/semantic_ids, в git не хранятся; описание: docs/semantic_ids.md):

  semantic_codes.npy    (n_tracks, 4) int64: три кода RQ-VAE (0..255) + номер среди треков с теми же тремя
                        кодами (0..8, почти всегда 0) — без него 7.8% треков неотличимы. Строка i = i-й трек
                        tracks_meta по возрастанию m4a_id.
  rqvae_checkpoint.pt   {'state_dict', 'config', 'input_dim'}: энкодер MuQ (128) -> 128 -> 32 (ReLU), три кодбука
                        256 x 32, декодер 32 -> 128. Код уровня k — ближайший (евклидово) центроид к остатку.

Векторы: центроиды кодбуков (codebooks[k][c], 32 числа) и их сумма по уровням — квантованный вектор трека
(приближение выхода энкодера z; остаток в среднем ~22% нормы z).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

CODES_FILE = "semantic_codes.npy"
CHECKPOINT_FILE = "rqvae_checkpoint.pt"


class RQVAE:
    """RQ-VAE из чекпоинта в numpy: encode (MuQ -> z), quantize (z -> коды), decode (z -> MuQ)."""

    def __init__(self, state: Dict[str, np.ndarray], config: Optional[Dict[str, Any]] = None):
        self.w = state
        self.config = config or {}
        self.codebooks = np.stack([state[f"codebooks.{k}"] for k in range(self.n_levels(state))])  # (L, K, d)

    @staticmethod
    def n_levels(state: Dict[str, np.ndarray]) -> int:
        return sum(1 for k in state if k.startswith("codebooks."))

    @classmethod
    def load(cls, path: str) -> "RQVAE":
        import torch  # только здесь: остальной модуль без torch
        ck = torch.load(path, map_location="cpu", weights_only=True)
        state = {k: v.numpy().astype(np.float32) for k, v in ck["state_dict"].items()}
        return cls(state, ck.get("config"))

    def _mlp(self, x: np.ndarray, prefix: str, layers: Sequence[int]) -> np.ndarray:
        for i, n in enumerate(layers):
            x = x @ self.w[f"{prefix}.{n}.weight"].T
            if f"{prefix}.{n}.bias" in self.w:
                x = x + self.w[f"{prefix}.{n}.bias"]
            if i < len(layers) - 1:
                x = np.maximum(x, 0.0)
        return x

    def encode(self, x: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):  # numpy 2.0 + Accelerate (macOS) даёт ложные warnings в matmul
            return self._mlp(np.asarray(x, dtype=np.float32), "encoder", (0, 2, 4))

    def decode(self, z: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            return self._mlp(np.asarray(z, dtype=np.float32), "decoder", (0, 2, 4))

    def quantize(self, z: np.ndarray) -> np.ndarray:
        """z (n, d) -> коды (n, L): на каждом уровне ближайший центроид к остатку."""
        r = np.array(z, dtype=np.float32)
        out = np.empty((len(r), len(self.codebooks)), dtype=np.int64)
        for k, cb in enumerate(self.codebooks):
            with np.errstate(all="ignore"):
                d = (r ** 2).sum(1, keepdims=True) - 2 * r @ cb.T + (cb ** 2).sum(1)
            out[:, k] = d.argmin(1)
            r -= cb[out[:, k]]
        return out


@dataclass
class SemanticIDs:
    """Коды треков: строка i — трек track_ids[i]; codebooks (L, K, d) — если загружен чекпоинт."""
    track_ids: np.ndarray
    codes: np.ndarray
    codebooks: Optional[np.ndarray] = None

    def __post_init__(self):
        self.track_ids = np.asarray(self.track_ids).astype(str)
        if len(self.track_ids) != len(self.codes):
            raise ValueError(f"кодов {len(self.codes)}, а треков {len(self.track_ids)}")
        self._pos = {t: i for i, t in enumerate(self.track_ids)}

    def __len__(self) -> int:
        return len(self.codes)

    @property
    def vocab_sizes(self) -> list:
        """Сколько значений у каждой позиции кода: размеры таблиц эмбеддингов (последняя — номер при коллизии)."""
        if self.codebooks is not None:
            sizes = [self.codebooks.shape[1]] * len(self.codebooks)
            return sizes + [int(c.max()) + 1 for c in self.codes.T[len(sizes):]]
        return [int(c.max()) + 1 for c in self.codes.T]

    def positions(self, track_ids: Iterable[str]) -> np.ndarray:
        """Строки кодов для track_ids; KeyError, если у трека нет кода."""
        return np.array([self._pos[str(t)] for t in track_ids], dtype=np.int64)

    def aligned(self, track_ids: Iterable[str]) -> np.ndarray:
        """Коды в порядке track_ids (например, каталога): (n, n_codes)."""
        return self.codes[self.positions(track_ids)]

    def centroids(self) -> np.ndarray:
        """(n, L, d): центроид каждого уровня для каждого трека."""
        if self.codebooks is None:
            raise ValueError("кодбуков нет: загрузите с чекпоинтом (load_semantic_ids(..., checkpoint=True))")
        L = len(self.codebooks)
        return np.stack([self.codebooks[k][self.codes[:, k]] for k in range(L)], axis=1)

    def quantized(self) -> np.ndarray:
        """(n, d): сумма центроидов по уровням — квантованный вектор трека в пространстве RQ-VAE."""
        return self.centroids().sum(axis=1)


def sorted_track_ids(meta_path: str) -> np.ndarray:
    """m4a_id из tracks_meta по возрастанию: порядок строк semantic_codes.npy."""
    import pyarrow.parquet as pq
    return np.sort(pq.read_table(meta_path, columns=["m4a_id"]).column("m4a_id").to_numpy().astype(str))


def semantic_paths(cfg: Dict[str, Any]) -> Dict[str, str]:
    s = cfg.get("semantic_ids", {})
    d = s.get("dir", "artifacts/semantic_ids")
    return {"codes": os.path.join(d, s.get("codes", CODES_FILE)),
            "checkpoint": os.path.join(d, s.get("checkpoint", CHECKPOINT_FILE))}


def load_semantic_ids(cfg: Dict[str, Any], checkpoint: bool = True) -> SemanticIDs:
    """Коды из semantic_ids.dir с m4a_id из tracks_meta (data.crs.dir); checkpoint — ещё и кодбуки RQ-VAE."""
    from recsys.data.crs import split_path
    paths = semantic_paths(cfg)
    if not os.path.exists(paths["codes"]):
        raise FileNotFoundError(f"нет {paths['codes']}: положите коды и чекпоинт RQ-VAE в {os.path.dirname(paths['codes'])} "
                                f"(docs/semantic_ids.md)")
    codes = np.load(paths["codes"])
    ids = sorted_track_ids(split_path(cfg["data"]["crs"]["dir"], "tracks_meta"))
    codebooks = RQVAE.load(paths["checkpoint"]).codebooks if checkpoint else None
    return SemanticIDs(ids, codes, codebooks)
