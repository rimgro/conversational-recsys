"""Pretrained loading must survive a dropped connection without extra tokenizer calls."""
from types import SimpleNamespace

import pytest
import torch
from requests.exceptions import ConnectionError
from transformers import AutoModel, AutoTokenizer, Qwen3Config, Qwen3Model

from recsys.two_tower_qwen.config import TrainConfig
from recsys.two_tower_qwen.model import _cached_pretrained, load_backbone


def test_complete_cache_never_uses_network():
    calls = []
    cached = object()
    def factory(model_id, **kwargs):
        calls.append(kwargs)
        assert kwargs["local_files_only"] is True
        return cached
    assert _cached_pretrained(factory, "example", revision="pinned") is cached
    assert len(calls) == 1 and calls[0]["revision"] == "pinned"


def test_incomplete_cache_retries_connection_and_keeps_revision(monkeypatch):
    calls, waits = [], []
    monkeypatch.setattr("recsys.two_tower_qwen.model.time.sleep", waits.append)
    def factory(model_id, **kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise OSError("Weights not cached")
        if len(calls) == 2:
            raise ConnectionError("RemoteDisconnected")
        return "loaded"
    assert _cached_pretrained(factory, "example", revision="pinned") == "loaded"
    assert len(calls) == 3 and waits == [2]
    assert all(call["revision"] == "pinned" for call in calls)


def test_retry_is_bounded_and_model_errors_are_not_retried(monkeypatch):
    calls = []
    monkeypatch.setattr("recsys.two_tower_qwen.model.time.sleep", lambda _: None)
    def disconnected(model_id, **kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise OSError("Not cached")
        raise ConnectionError("RemoteDisconnected")
    with pytest.raises(ConnectionError):
        _cached_pretrained(disconnected, "example", attempts=3)
    assert len(calls) == 4
    def invalid(model_id, **kwargs):
        raise ValueError("Invalid model architecture")
    with pytest.raises(ValueError):
        _cached_pretrained(invalid, "example")


def test_existing_tokenizer_is_reused(monkeypatch):
    body = Qwen3Model(Qwen3Config(hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, vocab_size=64))
    tokenizer = SimpleNamespace(pad_token_id=0, padding_side="left")
    def unwanted_tokenizer(*args, **kwargs):
        pytest.fail("An existing tokenizer must not be reloaded")
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", unwanted_tokenizer)
    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *args, **kwargs: body)
    backbone, returned = load_backbone(TrainConfig(lora_rank=2), torch.float32,
                                      tokenizer=tokenizer)
    assert returned is tokenizer and returned.padding_side == "right"
    assert backbone.get_input_embeddings().weight.requires_grad is False
