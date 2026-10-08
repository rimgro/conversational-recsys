"""Вектор запроса для HNSW, построенный у нас (на GPU DataSphere это быстрее, чем на CPU сервера).

Рецепт тот же, что в сервисе hnsw-v1 (recsys/retrieval/gemma_lancedb.py в теге hnsw-v1), иначе векторы
запроса и индекса будут несовместимы: текст 'task: search result | query: <запрос>' с схлопнутыми пробелами ->
google/embeddinggemma-2 через sentence-transformers -> 768 чисел, NaN -> 0, L2-нормировка.

Нужны sentence-transformers и доступ к модели на Hugging Face (модель закрытая: принять лицензию, HF_TOKEN).
Модель грузится при первом запросе и одна на процесс.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MODEL_NAME = "google/embeddinggemma-2"
DIM = 768
QUERY_TEMPLATE = "task: search result | query: {query}"

_MODELS: Dict[Tuple[str, Optional[str], Optional[str], int], Any] = {}


def query_text(query: str) -> str:
    return QUERY_TEMPLATE.format(query=re.sub(r"\s+", " ", str(query or "")).strip())


class GemmaQueryEmbedder:
    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None, dtype: Optional[str] = None,
                 max_seq_length: int = 2048):
        self.key = (model_name, device, dtype, max_seq_length)

    def _model(self):
        if self.key not in _MODELS:
            from sentence_transformers import SentenceTransformer
            model_name, device, dtype, max_seq_length = self.key
            kwargs: Dict[str, Any] = {}
            if dtype:
                import torch
                kwargs["model_kwargs"] = {"torch_dtype": {"float16": torch.float16, "bfloat16": torch.bfloat16,
                                                          "float32": torch.float32}.get(dtype, dtype)}
            model = SentenceTransformer(model_name, device=device, trust_remote_code=True, **kwargs)
            model.max_seq_length = max_seq_length
            _MODELS[self.key] = model
        return _MODELS[self.key]

    def encode(self, query: str) -> List[float]:
        vec = self._model().encode([query_text(query)], normalize_embeddings=False, convert_to_numpy=True)[0]
        vec = np.nan_to_num(np.asarray(vec, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        norm = float(np.linalg.norm(vec))
        return (vec / norm if norm > 0 else vec).tolist()
