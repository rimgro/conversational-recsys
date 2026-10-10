"""Вектор запроса для HNSW, построенный у нас (на GPU DataSphere это быстрее, чем на CPU сервера).

Рецепт тот же, что в сервисе hnsw-v1 (recsys/retrieval/gemma_lancedb.py в теге hnsw-v1), иначе векторы
запроса и индекса будут несовместимы: текст 'task: search result | query: <запрос>' с схлопнутыми пробелами ->
google/embeddinggemma-2 через sentence-transformers -> 768 чисел, NaN -> 0, L2-нормировка.

Нужны sentence-transformers и доступ к модели на Hugging Face (модель закрытая: принять лицензию, HF_TOKEN).
Подготовка один раз на машину — python scripts/make_embed.py. Модель грузится при первом запросе и одна на процесс.
Устройство: GPU (cuda, float16), если он есть, иначе CPU (float32). В DataSphere кэш моделей — на диске проекта.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MODEL_NAME = "google/embeddinggemma-2"
DIM = 768
QUERY_TEMPLATE = "task: search result | query: {query}"

DATASPHERE_PROJECT = "/home/jupyter/project"
_MODELS: Dict[Tuple[str, Optional[str], Optional[str], int], Any] = {}


def setup_cache() -> str:
    """Где лежат веса Hugging Face. В DataSphere — на диске проекта (переживает перезапуск ВМ), если HF_HOME
    не задан явно; локально — стандартный ~/.cache/huggingface. Вызывать до импорта transformers."""
    if "HF_HOME" not in os.environ and os.path.isdir(DATASPHERE_PROJECT):
        os.environ["HF_HOME"] = os.path.join(DATASPHERE_PROJECT, "hf_cache")
    return os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def pick_device(device: Optional[str], dtype: Optional[str], cuda_available: bool) -> Tuple[str, Optional[str]]:
    """device / dtype из конфига или автоматически: cuda + float16, если есть GPU, иначе cpu + float32."""
    device = device or ("cuda" if cuda_available else "cpu")
    if dtype is None and device.startswith("cuda"):
        dtype = "float16"
    return device, dtype


def query_text(query: str) -> str:
    return QUERY_TEMPLATE.format(query=re.sub(r"\s+", " ", str(query or "")).strip())


class GemmaQueryEmbedder:
    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None, dtype: Optional[str] = None,
                 max_seq_length: int = 2048):
        self.model_name, self.max_seq_length = model_name, max_seq_length
        self.requested = (device, dtype)
        self.key: Optional[Tuple[str, Optional[str], Optional[str], int]] = None  # известен после загрузки

    @property
    def device(self) -> Optional[str]:
        return self.key[1] if self.key else None

    def _model(self):
        if self.key is None:
            setup_cache()
            import torch
            device, dtype = pick_device(*self.requested, cuda_available=torch.cuda.is_available())
            self.key = (self.model_name, device, dtype, self.max_seq_length)
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
