"""Continuous event tokens + trainable markers + frozen Qwen body with LoRA."""
import math
import time

import torch
from torch import nn
from torch.nn import functional as F

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MARKERS = ("[HISTORY]", "[QUERY]", "[RET]")


def attach_lora(backbone, config):
    from peft import LoraConfig, TaskType, get_peft_model
    backbone.requires_grad_(False)
    backbone.config.use_cache = False
    return get_peft_model(backbone, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=config.lora_rank,
        lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
        target_modules=list(LORA_TARGETS), bias="none"))


def _cached_pretrained(factory, model_id, *, attempts=4, **kwargs):
    """Use complete local files first; retry only transient transport failures."""
    from requests.exceptions import ConnectionError, Timeout
    try:
        return factory(model_id, local_files_only=True, **kwargs)
    except OSError:
        # A first run can have a tokenizer in cache but no model weights yet.
        pass
    for attempt in range(attempts):
        try:
            return factory(model_id, **kwargs)
        except (ConnectionError, Timeout):
            if attempt + 1 == attempts:
                raise
            delay = 2 ** (attempt + 1)
            print(f"Hugging Face connection interrupted; retry {attempt + 2}/{attempts} in {delay}s",
                  flush=True)
            time.sleep(delay)


def load_backbone(config, dtype=torch.float32, cache_dir=None, tokenizer=None):
    from transformers import AutoModel, AutoTokenizer
    if tokenizer is None:
        tokenizer = _cached_pretrained(AutoTokenizer.from_pretrained, config.qwen_model,
            revision=config.qwen_revision, cache_dir=cache_dir, trust_remote_code=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # AutoModel omits lm_head; no logits, generation or vocabulary-sized output.
    backbone = _cached_pretrained(AutoModel.from_pretrained, config.qwen_model, revision=config.qwen_revision,
        torch_dtype=dtype, cache_dir=cache_dir, trust_remote_code=False,
        attn_implementation="sdpa")
    backbone = attach_lora(backbone, config)
    if config.gradient_checkpointing:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return backbone, tokenizer


class EventProjector(nn.Module):
    def __init__(self, buckets, hidden_size):
        super().__init__()
        # Weighted sum of table rows equals a linear projection of a sparse hash.
        self.hash_projection = nn.EmbeddingBag(buckets, 256, mode="sum")
        nn.init.normal_(self.hash_projection.weight, std=.02)
        self.hash_norm = nn.LayerNorm(256)
        self.audio = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 128), nn.GELU())
        # Six values + six missingness bits + audio/category presence.
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
    def __init__(self, backbone, hash_buckets=8192, embedding_dim=256):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = int(backbone.config.hidden_size)
        self.events = EventProjector(hash_buckets, self.hidden_size)
        self.marker_embeddings = nn.Parameter(torch.empty(3, self.hidden_size))
        nn.init.normal_(self.marker_embeddings, std=.02)
        self.output = nn.Sequential(nn.LayerNorm(self.hidden_size),
                                    nn.Linear(self.hidden_size, embedding_dim))

    @torch.no_grad()
    def initialize_markers(self, tokenizer):
        embedding = self.backbone.get_input_embeddings()
        for i, text in enumerate(("Listening history", "Music search query")):
            ids = tokenizer.encode(text, add_special_tokens=False)
            ids = torch.tensor(ids, device=embedding.weight.device, dtype=torch.long)
            self.marker_embeddings[i].copy_(embedding(ids).float().mean(0))
        # Retrieval pooling starts from the pretrained final-token embedding.
        self.marker_embeddings[2].copy_(embedding.weight[tokenizer.eos_token_id].float())
        scale = embedding.weight[:min(2048, len(embedding.weight))].float().square().mean().sqrt()
        self.events.log_scale.copy_(scale.clamp_min(1e-4).log())

    def assemble(self, event_tokens, history_mask, query_ids, query_mask):
        """No cached gradients; RET is never truncated and padding is never pooled."""
        text = self.backbone.get_input_embeddings()(query_ids)
        markers = self.marker_embeddings.to(text.dtype)
        sequences = []
        for i in range(len(text)):
            sequences.append(torch.cat((markers[0:1], event_tokens[i][history_mask[i]].to(text.dtype),
                markers[1:2], text[i][query_mask[i]], markers[2:3]), dim=0))
        inputs = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        lengths = torch.tensor([len(s) for s in sequences], device=inputs.device)
        attention = torch.arange(inputs.shape[1], device=inputs.device)[None, :] < lengths[:, None]
        positions = attention.long().cumsum(-1) - 1
        positions = positions.clamp_min(0).masked_fill(~attention, 0)
        return inputs, attention.long(), positions, lengths - 1

    def forward(self, hash_indices, hash_weights, audio, numbers, history_mask, query_ids, query_mask):
        events = self.events(hash_indices, hash_weights, audio, numbers)
        inputs, attention, positions, final_positions = self.assemble(
            events, history_mask.bool(), query_ids, query_mask.bool())
        # Frozen weights still require autograd through their inputs to train events.
        result = self.backbone(inputs_embeds=inputs, attention_mask=attention,
                               position_ids=positions, use_cache=False, return_dict=True)
        final = result.last_hidden_state[torch.arange(len(inputs), device=inputs.device), final_positions]
        return F.normalize(self.output(final.float()).float(), dim=-1)

    def trainable_state(self):
        return {name: value.detach().cpu().clone() for name, value in self.named_parameters()
                if value.requires_grad}

    def load_trainable_state(self, state):
        parameters = {name: p for name, p in self.named_parameters() if p.requires_grad}
        if set(state) != set(parameters):
            raise ValueError("Trainable parameter schema differs from checkpoint")
        with torch.no_grad():
            for name, param in parameters.items():
                param.copy_(state[name].to(param.device, param.dtype))


class QwenTwoTower(nn.Module):
    def __init__(self, query_tower, item_tower):
        super().__init__()
        self.query_tower, self.item_tower = query_tower, item_tower

    def item(self, features):
        return F.normalize(self.item_tower(features).float(), dim=-1)


def sampled_softmax_loss(query, positive, negative, positive_groups, negative_groups,
                         catalog_size, temperature=.08):
    """Uniform IID draws WITH replacement, deterministic positive (expected count 1).

    Duplicate negative draws keep their multiplicity. Same-record accidental hits
    are masked, including m4a aliases with the same spotify_id. No logQ at serving.
    """
    if len(negative) == 0 or catalog_size < 2 or temperature <= 0:
        raise ValueError("Invalid sampled softmax settings")
    positive_logits = (query.float() * positive.float()).sum(-1, keepdim=True) / temperature
    negative_logits = query.float() @ negative.float().T / temperature
    negative_logits -= math.log(len(negative) / catalog_size)
    accidental = positive_groups[:, None].eq(negative_groups[None, :])
    negative_logits = negative_logits.masked_fill(accidental, -torch.inf)
    logits = torch.cat((positive_logits, negative_logits), dim=1)
    return F.cross_entropy(logits, torch.zeros(len(query), dtype=torch.long, device=query.device))
