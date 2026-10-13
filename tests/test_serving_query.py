"""Left-only loader and CPU API filtering against the frozen model/scorer."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3Model

from recsys.serving.artifacts import split_bundle
from recsys.serving.index import CandidateIndex
from recsys.serving import query
from recsys.two_tower_qwen.bundle import QwenRetriever
from recsys.two_tower_qwen.config import TrainConfig
from recsys.two_tower_qwen.model import QwenHistoryTower, attach_lora
from recsys.two_tower_qwen.metrics import CatalogScorer
from serving_fixture import make_export, write_manifest


def test_left_only_loader_uses_same_weights_history_and_encoding_recipe(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    config = TrainConfig(hash_buckets=32, lora_rank=2, lora_alpha=4, lora_dropout=.1)
    def backbone():
        torch.manual_seed(11)
        cfg = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=64)
        cfg._attn_implementation = "sdpa"
        return attach_lora(Qwen3Model(cfg), config)
    initial = QwenHistoryTower(backbone(), 32, 256)
    source = make_export(tmp_path / "source", n=12)
    (source / "config.json").write_text(json.dumps({"training": config.to_dict()}))
    torch.save(initial.trainable_state(), source / "query_trainable.pt")
    write_manifest(source)
    split_bundle(source, tmp_path / "release")
    class Tokenizer:
        padding_side = "right"
        def __call__(self, *args, **kwargs):
            return {'input_ids': torch.tensor([[4,5,0]]*len(args[0])), 'attention_mask': torch.tensor([[1,1,0]]*len(args[0]))}
    monkeypatch.setattr(query, "load_backbone", lambda *a, **kw: (backbone(), Tokenizer()))
    monkeypatch.setattr(query.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    encoder = query.QwenQueryEncoder(tmp_path / "release/left")
    assert not hasattr(encoder, 'index') and not hasattr(encoder, 'item_tower')
    for name, value in initial.trainable_state().items():
        assert torch.equal(value, dict(encoder.tower.named_parameters())[name])
    assert encoder.encode_query.__func__ is QwenRetriever.encode_query
    # The frozen retriever encoder operating on the same released state is identical.
    reference = QwenRetriever.__new__(QwenRetriever)
    reference.__dict__.update(encoder.__dict__)
    history = ['unknown','0','1','0'] * 100
    vector = encoder.encode_query('query', history)
    np.testing.assert_allclose(vector, reference.encode_query('query', history), atol=1e-7)
    assert vector.shape == (256,) and np.isfinite(vector).all()
    assert np.linalg.norm(vector) == pytest.approx(1, abs=1e-6)
    assert all(not p.requires_grad for p in encoder.tower.parameters())
    # Full-split vector stage uses fresh query text and the same history recipe.
    import importlib.util
    script=__import__('pathlib').Path(__file__).resolve().parents[1]/'tools/encode_split_queries.py'
    spec=importlib.util.spec_from_file_location('batch_encode',script)
    batching=importlib.util.module_from_spec(spec);spec.loader.exec_module(batching)
    import pandas as pd
    profiles={'histories':np.array([[0,1,0],[0,0,0]]),'masks':np.array([[1,1,1],[0,0,0]],bool)}
    rows=pd.DataFrame({'query':['query','query'],'profile_index':[0,1]})
    batched=batching.encode_batch(encoder,rows,profiles)
    np.testing.assert_allclose(batched[0],encoder.encode_query('query',['0','1','0']),atol=2e-6)
    np.testing.assert_allclose(batched[1],encoder.encode_query('query',[]),atol=2e-6)


@pytest.mark.parametrize('discovery', [False, True])
@pytest.mark.parametrize('similar', [False, True])
def test_cpu_server_filtering_matches_original_canonical_scorer(tmp_path, discovery, similar):
    source = make_export(tmp_path / 'source', n=40)
    split_bundle(source, tmp_path / 'split')
    index = CandidateIndex(tmp_path / 'split/right')
    vector = np.array([1.] + [0.] * 255, np.float32)
    history = ['0','1','8','unknown']
    row = {'query_type':'similar_to' if similar else 'exact','exclude_m4a_id':'4',
        'exclude_artist':'ARTIST0'}
    original = CatalogScorer(index.catalog, [row['query_type']])
    ranking = np.argsort(-(index.vectors @ vector), kind='stable')
    expected = original.clean_ranking([index.ids[i] for i in ranking], row,
        original.history_keys(history), discovery, k=20)
    current = index.search(vector, k=20, mode='exact', history=history, discovery=discovery, **row)
    assert current['ids'] == expected
