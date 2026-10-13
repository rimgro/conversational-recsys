"""Плотные кандгены: косинус вектора запроса и векторов треков, точный поиск по всему каталогу (64k — это быстро,
ANN не нужен). Векторы считаются заранее на GPU и лежат в папке candgen.sources.<имя>.dir:

  item_vectors.npy + item_ids.npy   векторы треков (n, dim) и их m4a_id
  query_vectors_*.npz               query_id -> вектор запроса (Request.request_id = query_id датасета); все файлы
                                    папки, имя не важно: query_vectors_train.npz (ranker_train, valid) и
                                    query_vectors_test_public.npz (test, submit) или по частям

  hnsw      EmbeddingGemma-2 по тексту запроса против векторов треков из индекса hnsw-v1 (dev/hnsw);
            файлы — python scripts/encode_hnsw.py. Модель предобучена, на наших запросах не учится — утечки нет.
  twotower  двухбашенный (recsys/twotower/, docs/twotower.md): история + запрос против трека;
            файлы — python scripts/encode_twotower.py. Учится только на части index (data/parts.py).

Запроса нет в файлах (новая реплика в диалоге) — online: true кодирует его моделью (hnsw — EmbeddingGemma
на CPU/GPU, twotower — Qwen + обученные веса, нужен torch); иначе для него пусто и предупреждение.
"""
from __future__ import annotations

import glob
import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from recsys.candgen.base import Candgen, Cands, register, top_positions
from recsys.schemas import Context


class _Dense(Candgen):
    def __init__(self, cfg, res):
        super().__init__(cfg, res)
        self.dir = cfg.get("dir") or ""
        if not os.path.exists(os.path.join(self.dir, "item_vectors.npy")):
            raise FileNotFoundError(f"candgen.sources.{self.name}.dir: нет {self.dir}/item_vectors.npy "
                                    f"({self.how_to_build})")
        catalog = res.catalog
        v = np.load(os.path.join(self.dir, "item_vectors.npy")).astype(np.float32)
        ids = np.load(os.path.join(self.dir, "item_ids.npy")).astype(str)
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
        self.vectors = np.zeros((len(catalog), v.shape[1]), dtype=np.float32)   # в порядке каталога
        self.known = np.zeros(len(catalog), dtype=bool)
        pos = np.array([-1 if p is None else p for p in map(catalog.pos, ids)])
        ok = pos >= 0
        self.vectors[pos[ok]] = v[ok]
        self.known[pos[ok]] = True
        # векторы запросов: одна матрица (float16, как в файлах) и номер строки по query_id — сотни тысяч запросов
        ids_, vecs = [], []
        for path in sorted(glob.glob(os.path.join(self.dir, cfg.get("query_files", "query_vectors_*.npz")))):
            with np.load(path, allow_pickle=False) as z:
                ids_.append(z["query_id"].astype(str))
                vecs.append(z["vectors"])
        self.query_matrix = np.concatenate(vecs) if vecs else np.zeros((0, v.shape[1]), dtype=np.float16)
        self.query_row: Dict[str, int] = {q: i for i, q in enumerate(np.concatenate(ids_))} if ids_ else {}
        self._encoder: Any = None
        self.n_missing = 0   # запросы без вектора (нет в файлах и online выключен)

    how_to_build = ""

    def encoder(self):
        """Модель для запросов, которых нет в файлах (online: true); грузится при первом таком запросе."""
        raise NotImplementedError

    def query_vector(self, ctx: Context) -> Optional[np.ndarray]:
        row = self.query_row.get(str(ctx.request.request_id)) if ctx.request.request_id else None
        u = self.query_matrix[row].astype(np.float32) if row is not None else None
        if u is None and self.cfg.get("online"):
            if self._encoder is None:
                self._encoder = self.encoder()
            u = np.asarray(self._encoder.encode(self.encoder_input(ctx)), dtype=np.float32)
        return u

    def encoder_input(self, ctx: Context):
        return ctx

    def check_coverage(self, contexts: List[Context], part: str) -> None:
        """Векторы есть для запросов части? Нет для больше max_missing (по умолчанию 1%) — ошибка с подсказкой,
        а не молча пустые списки: без векторов кандген ничего не даёт, а ранкер учится без него."""
        if self.cfg.get("online"):
            return
        ids = [str(c.request.request_id) for c in contexts]
        missing = sum(q not in self.query_row for q in ids)
        if missing > float(self.cfg.get("max_missing", 0.01)) * len(ids):
            raise ValueError(
                f"{self.name}: нет векторов для {missing} из {len(ids)} запросов части {part} в {self.dir}. "
                f"Посчитать: {self.how_to_build} (нужны сплиты train и test_public), "
                f"или выключить: --sources -{self.name}")

    def retrieve(self, batch: List[Context]) -> List[Cands]:
        out = []
        for ctx in batch:
            u = self.query_vector(ctx)
            if u is None:
                if self.n_missing == 0:
                    warnings.warn(f"{self.name}: нет вектора запроса {ctx.request.request_id} (нет в {self.dir} "
                                  f"и online: false), кандидатов не будет")
                self.n_missing += 1
                out.append(Cands.empty())
                continue
            u = u / max(float(np.linalg.norm(u)), 1e-8)
            with np.errstate(all="ignore"):   # numpy 2.0 + Accelerate (macOS) даёт ложные warnings в matmul
                scores = self.vectors @ u
            scores[~self.known] = -np.inf
            out.append(top_positions(scores, self.top_k, positive=False))
        return out

    def cache_key(self) -> Dict[str, Any]:
        files = sorted(glob.glob(os.path.join(self.dir, "*.np[yz]")))
        return {**super().cache_key(), "files": {os.path.basename(f): int(os.path.getmtime(f)) for f in files}}


@register("hnsw", "dense")
class HNSW(_Dense):
    """EmbeddingGemma-2: текст последней реплики против векторов треков (метаданные + текст песни)."""
    how_to_build = "python scripts/encode_hnsw.py --parts ranker_train,valid,test,submit"

    def encoder(self):
        from recsys.candgen.query_embedder import GemmaQueryEmbedder
        return GemmaQueryEmbedder(**self.cfg.get("embedder", {}))

    def encoder_input(self, ctx: Context) -> str:
        messages = ctx.request.user_messages
        return messages[-1] if messages else ""


@register("twotower", "dense")
class TwoTower(_Dense):
    """Двухбашенный: история (semantic ID) и запрос против вектора трека."""
    how_to_build = "python scripts/encode_twotower.py --run-dir <папка> --splits train,test_public"

    def encoder(self):
        from recsys.twotower.build import OnlineEncoder
        return OnlineEncoder.load(self.res.cfg, os.path.join(self.dir, self.cfg.get("checkpoint", "model.pt")))

    def encoder_input(self, ctx: Context):
        return ctx.request
