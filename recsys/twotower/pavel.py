"""Башня Павла онлайн (кандген twotower_pavel, online: true): Qwen3-Embedding + LoRA по истории и запросу -> вектор
256, точный поиск по векторам треков его экспорта.

Код модели — из ветки codex/qwen-vps-api (recsys/two_tower_qwen/model.py, features.py, bundle.py:
QwenRetriever.encode_query; recsys/serving/index.py: канонические группы треков), только то, что нужно для
инференса: обучения, HNSW и HTTP здесь нет. Имена параметров те же — веса экспорта грузятся как есть.

Папка экспорта (bundle Павла: полный exports/<bundle_id>/ или left/ и right/ split_serving_bundle.py в одной папке):
  config.json              {"training": TrainConfig}: qwen_model, qwen_revision, hash_buckets, embedding_dim, ...
  tokenizer/               токенайзер Qwen
  query_trainable.pt       обученные веса левой башни (LoRA, проекция событий, маркеры, выход)
  history_features.npz     признаки треков для событий истории: indices, weights, audio, numbers
  catalog.json             [{m4a_id, spotify_id, artist}] — строки векторов и признаков
  track_vectors.npy        (n, 256) векторы треков (правая башня), L2-нормированы

Готовые top-200 (artifacts/twotower_pavel/*_top200.parquet) посчитаны этой же моделью (HNSW, ef 1024); точный
поиск даёт почти те же списки. Отличие: для similar_to Павел убирал образец и его артиста по qrels — онлайн
этого нет (query_type и ответы модели не известны).
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
FILES = ("config.json", "query_trainable.pt", "history_features.npz", "catalog.json", "track_vectors.npy",
         "tokenizer")


def load_backbone(training: Dict, dtype: torch.dtype, cache_dir: Optional[str] = None):
    """Замороженная Qwen (закреплённая ревизия) + LoRA с теми же параметрами, что при обучении."""
    import transformers
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModel
    new_api = tuple(map(int, transformers.__version__.split(".")[:2])) >= (4, 56)   # torch_dtype -> dtype
    kwargs = {"dtype" if new_api else "torch_dtype": dtype, "revision": training.get("qwen_revision"),
              "cache_dir": cache_dir, "trust_remote_code": False, "attn_implementation": "sdpa"}
    try:
        backbone = AutoModel.from_pretrained(training["qwen_model"], local_files_only=True, **kwargs)
    except OSError:   # весов ещё нет в кэше Hugging Face — скачать
        backbone = AutoModel.from_pretrained(training["qwen_model"], **kwargs)
    backbone.requires_grad_(False)
    backbone.config.use_cache = False
    return get_peft_model(backbone, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=training.get("lora_rank", 16),
        lora_alpha=training.get("lora_alpha", 32), lora_dropout=training.get("lora_dropout", 0.05),
        target_modules=list(LORA_TARGETS), bias="none"))


class EventProjector(nn.Module):
    def __init__(self, buckets: int, hidden_size: int):
        super().__init__()
        self.hash_projection = nn.EmbeddingBag(buckets, 256, mode="sum")
        self.hash_norm = nn.LayerNorm(256)
        self.audio = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 128), nn.GELU())
        self.numbers = nn.Sequential(nn.LayerNorm(14), nn.Linear(14, 64), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(448, hidden_size), nn.GELU(),
                                    nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size))
        self.log_scale = nn.Parameter(torch.tensor(math.log(.02)))

    def forward(self, hash_indices, hash_weights, audio, numbers):
        shape = hash_indices.shape[:-1]
        width = hash_indices.shape[-1]
        offsets = torch.arange(0, hash_indices.numel(), width, device=hash_indices.device)
        categories = self.hash_projection(hash_indices.reshape(-1), offsets,
                                          per_sample_weights=hash_weights.reshape(-1).float())
        categories = self.hash_norm(categories)
        combined = torch.cat((categories, self.audio(audio.reshape(-1, 128)),
                              self.numbers(numbers.reshape(-1, 14))), dim=-1)
        return (self.fusion(combined) * self.log_scale.exp()).reshape(*shape, -1)


class QwenHistoryTower(nn.Module):
    """[HISTORY] события истории [QUERY] текст запроса [RET] -> выход на [RET] -> L2(Linear) = вектор 256."""

    def __init__(self, backbone, hash_buckets: int = 8192, embedding_dim: int = 256):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = int(backbone.config.hidden_size)
        self.events = EventProjector(hash_buckets, self.hidden_size)
        self.marker_embeddings = nn.Parameter(torch.empty(3, self.hidden_size))
        self.output = nn.Sequential(nn.LayerNorm(self.hidden_size), nn.Linear(self.hidden_size, embedding_dim))

    def assemble(self, event_tokens, history_mask, query_ids, query_mask):
        text = self.backbone.get_input_embeddings()(query_ids)
        markers = self.marker_embeddings.to(text.dtype)
        sequences = []
        for i in range(len(text)):
            sequences.append(torch.cat((markers[0:1], event_tokens[i][history_mask[i]].to(text.dtype),
                                        markers[1:2], text[i][query_mask[i]], markers[2:3]), dim=0))
        inputs = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        lengths = torch.tensor([len(s) for s in sequences], device=inputs.device)
        attention = torch.arange(inputs.shape[1], device=inputs.device)[None, :] < lengths[:, None]
        positions = (attention.long().cumsum(-1) - 1).clamp_min(0).masked_fill(~attention, 0)
        return inputs, attention.long(), positions, lengths - 1

    def forward(self, hash_indices, hash_weights, audio, numbers, history_mask, query_ids, query_mask):
        events = self.events(hash_indices, hash_weights, audio, numbers)
        inputs, attention, positions, final_positions = self.assemble(
            events, history_mask.bool(), query_ids, query_mask.bool())
        result = self.backbone(inputs_embeds=inputs, attention_mask=attention,
                               position_ids=positions, use_cache=False, return_dict=True)
        final = result.last_hidden_state[torch.arange(len(inputs), device=inputs.device), final_positions]
        return F.normalize(self.output(final.float()).float(), dim=-1)

    def load_trainable_state(self, state: Dict[str, torch.Tensor]) -> None:
        parameters = {name: p for name, p in self.named_parameters() if p.requires_grad}
        if set(state) != set(parameters):
            raise ValueError("twotower_pavel: обучаемые веса экспорта не совпали с моделью "
                             f"(нет {sorted(set(parameters) - set(state))[:3]}, лишние {sorted(set(state) - set(parameters))[:3]})")
        with torch.no_grad():
            for name, param in parameters.items():
                param.copy_(state[name].to(param.device, param.dtype))


def recent_history(history: Sequence[str], id_to_index: Dict[str, int], max_events: int = 256) -> List[int]:
    """Последние max_events прослушиваний (с повторами, по порядку), затем без неизвестных треков."""
    return [id_to_index[str(t)] for t in list(history)[-max_events:] if str(t) in id_to_index]


def _text(value) -> str:
    if value is None or isinstance(value, float) and not math.isfinite(value):
        return ""
    value = " ".join(str(value).split())
    return "" if value.casefold() in {"", "nan", "none", "null", "unknown", "n/a"} else value


class PavelTower:
    """Экспорт Павла: запрос + история -> top-k m4a_id (без повторов канонических групп: записи с одним
    spotify_id — один трек, остаётся первая по скору)."""

    def __init__(self, bundle_dir: str, device: Optional[str] = None, cache_dir: Optional[str] = None):
        missing = [f for f in FILES if not os.path.exists(os.path.join(bundle_dir, f))]
        if missing:
            raise FileNotFoundError(f"twotower_pavel: в {bundle_dir} нет {missing} (экспорт башни Павла, "
                                    f"recsys/twotower/pavel.py)")
        from transformers import AutoTokenizer
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = ((torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
                      if self.device.type == "cuda" else torch.float32)
        with open(os.path.join(bundle_dir, "config.json"), encoding="utf8") as f:
            self.training = json.load(f)["training"]
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(bundle_dir, "tokenizer"), trust_remote_code=False)
        self.tokenizer.padding_side = "right"
        backbone = load_backbone(self.training, self.dtype, cache_dir)
        self.tower = QwenHistoryTower(backbone, self.training.get("hash_buckets", 8192),
                                      self.training.get("embedding_dim", 256)).to(self.device)
        state = torch.load(os.path.join(bundle_dir, "query_trainable.pt"), map_location="cpu", weights_only=True)
        self.tower.load_trainable_state(state)
        self.tower.eval().requires_grad_(False)
        with open(os.path.join(bundle_dir, "catalog.json"), encoding="utf8") as f:
            catalog = json.load(f)
        self.ids = [str(r["m4a_id"]) for r in catalog]
        self.lookup = {t: i for i, t in enumerate(self.ids)}
        keys = ["spotify:" + _text(r.get("spotify_id")) if _text(r.get("spotify_id")) else "m4a:" + str(r["m4a_id"])
                for r in catalog]
        self.group = np.unique(keys, return_inverse=True)[1].ravel()
        with np.load(os.path.join(bundle_dir, "history_features.npz"), allow_pickle=False) as z:
            self.features = {k: torch.tensor(z[k], device=self.device) for k in ("indices", "weights", "audio", "numbers")}
        self.vectors = np.load(os.path.join(bundle_dir, "track_vectors.npy")).astype(np.float32)
        if len(self.vectors) != len(self.ids) or any(len(v) != len(self.ids) for v in self.features.values()):
            raise ValueError(f"twotower_pavel: в {bundle_dir} число треков в catalog.json, векторах и признаках разное")

    @torch.inference_mode()
    def encode(self, query: str, history: Sequence[str]) -> np.ndarray:
        """Вектор запроса (256, L2): последние history_events прослушиваний + текст (QwenRetriever.encode_query)."""
        selected = recent_history(history, self.lookup, self.training.get("history_events", 256))
        h = torch.tensor([selected or [0]], device=self.device)
        mask = torch.tensor([[True] * len(selected) or [False]], device=self.device)
        text = self.tokenizer([query], padding=True, truncation=True, max_length=self.training.get("query_tokens", 128),
                              add_special_tokens=False, return_tensors="pt")
        inputs = {"hash_indices": self.features["indices"][h], "hash_weights": self.features["weights"][h],
                  "audio": self.features["audio"][h], "numbers": self.features["numbers"][h], "history_mask": mask,
                  "query_ids": text["input_ids"].to(self.device), "query_mask": text["attention_mask"].to(self.device)}
        if self.device.type == "cuda":
            with torch.autocast("cuda", dtype=self.dtype):
                vector = self.tower(**inputs)
        else:
            vector = self.tower(**inputs)
        return vector[0].float().cpu().numpy()

    def search(self, query: str, history: Sequence[str], k: int = 200) -> Tuple[List[str], np.ndarray]:
        """top-k (m4a_id, косинус) точным поиском; пустой запрос — пусто (так у Павла: запрос обязателен)."""
        if not query.strip():
            return [], np.zeros(0, dtype=np.float32)
        scores = self.vectors @ self.encode(query, history)
        order = np.argsort(-scores, kind="stable")
        _, first = np.unique(self.group[order], return_index=True)   # первая (лучшая) запись каждой группы
        order = order[np.sort(first)][:k]
        return [self.ids[i] for i in order], scores[order].astype(np.float32)
