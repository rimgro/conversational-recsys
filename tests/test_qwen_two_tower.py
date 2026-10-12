"""CPU tests use a tiny real Qwen3 body, not pretrained recommendation claims."""
from pathlib import Path
import math

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
from transformers import Qwen3Config, Qwen3Model

from recsys.track_tower import TrackPreprocessor, TrackTransformerTower, pack_features
from recsys.two_tower_qwen.config import TrainConfig
from recsys.two_tower_qwen.data import PreparedData, atomic_json, balanced_validation, ExperimentBudget
from recsys.two_tower_qwen.features import HistoryHasher, categorical_fields, padded_histories, recent_history
from recsys.two_tower_qwen.metrics import CatalogScorer
from recsys.two_tower_qwen.model import QwenHistoryTower, QwenTwoTower, attach_lora, sampled_softmax_loss
from recsys.two_tower_qwen.training import Trainer, ScaleRetry


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def tiny_tower(checkpoint=False):
    torch.manual_seed(11)
    cfg = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=64, max_position_embeddings=512)
    cfg._attn_implementation = "sdpa"
    body = attach_lora(Qwen3Model(cfg), TrainConfig(lora_rank=2, lora_alpha=4, lora_dropout=.1))
    if checkpoint:
        body.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return QwenHistoryTower(body, 32, 8)


def example_inputs():
    return {"hash_indices": torch.tensor([[[1,2],[3,4],[0,0]],[[3,4],[0,0],[0,0]]]),
        "hash_weights": torch.ones(2,3,2), "audio": torch.randn(2,3,128),
        "numbers": torch.randn(2,3,14), "history_mask": torch.tensor([[1,1,0],[1,0,0]],dtype=torch.bool),
        "query_ids": torch.tensor([[4,5,0,0],[7,8,9,10]]),
        "query_mask": torch.tensor([[1,1,0,0],[1,1,1,1]],dtype=torch.bool)}


def test_hash_mean_namespaces_id_independence_and_canonical_codes():
    hasher = HistoryHasher(8192, 42)
    row = {"m4a_id":"id1", "title":"title1", "artist":" A  B ", "key":7.,
           "m4a_tags_full":"calm, rock, calm", "m4a_genres_full":"rock"}
    fields = categorical_fields(row)
    assert fields["tag"] == ["calm", "rock"] and fields["key"] == ["7"]
    assert hasher.row(row) == hasher.row({**row,"m4a_id":"id2","title":"title2","key":7})
    assert hasher.bucket_sign("genre=rock") != hasher.bucket_sign("tag=rock")
    sparse = dict(hasher.row({"m4a_tags_full":"calm, rock"}))
    for word in ("calm", "rock"):
        bucket, sign = hasher.bucket_sign("tag=" + word)
        assert sparse[bucket] == sign / 2
    assert HistoryHasher(8192,42).row(row) == hasher.row(row)


def test_sparse_projection_matches_dense_hash_linear():
    tower = tiny_tower()
    idx = torch.tensor([[1,2,1],[3,0,0]])
    weights = torch.tensor([[.5,-1.,.25],[1.,0.,0.]])
    offsets = torch.tensor([0,3])
    sparse = tower.events.hash_projection(idx.flatten(), offsets, per_sample_weights=weights.flatten())
    dense = torch.zeros(2,32).scatter_add_(1,idx,weights)
    assert torch.allclose(sparse, dense @ tower.events.hash_projection.weight, atol=1e-6)


def test_history_truncation_order_repeats_and_unknown():
    assert recent_history(["a","b","a","x","b"],{"a":0,"b":1},4) == [1,0,1]
    indices, mask = padded_histories([[],[0,1,0]])
    assert indices.shape == (2,3) and not mask[0].any() and mask[1].all()


def test_markers_ret_and_padding_invariance():
    tower = tiny_tower().eval()
    inp = example_inputs()
    events = tower.events(inp["hash_indices"],inp["hash_weights"],inp["audio"],inp["numbers"])
    seq, mask, positions, final = tower.assemble(events,inp["history_mask"],inp["query_ids"],inp["query_mask"])
    assert final.tolist() == [6,7] and mask.sum(1).tolist() == [7,8]
    assert torch.allclose(seq[0,6],tower.marker_embeddings[2])
    with torch.no_grad():
        batched = tower(**inp)
        single = tower(**{k:v[:1] for k,v in inp.items()})
    assert torch.allclose(batched[:1], single, atol=2e-6)
    assert torch.allclose(batched.norm(dim=1),torch.ones(2),atol=1e-6)
    assert positions[0,7] == 0


def test_gradient_policy_with_checkpointing_and_empty_history():
    tower = tiny_tower(checkpoint=True).train()
    inp = example_inputs()
    inp["history_mask"][1] = False
    output = tower(**inp)
    (output[:,0] - output[:,1]).sum().backward()
    assert tower.events.hash_projection.weight.grad.abs().sum() > 0
    assert tower.marker_embeddings.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n,p in tower.named_parameters() if "lora_" in n)
    assert all(p.grad is None and not p.requires_grad for n,p in tower.backbone.named_parameters() if "lora_" not in n)
    assert torch.isfinite(output).all()


def test_uniform_logq_and_alias_accidental_hits_keep_duplicate_draws():
    query,positive = torch.tensor([[1.,0.]]),torch.tensor([[1.,0.]])
    negative = torch.tensor([[0.,1.],[0.,1.],[1.,0.]])
    actual = sampled_softmax_loss(query,positive,negative,torch.tensor([5]),torch.tensor([1,1,5]),4,1.)
    logits = torch.tensor([[1.,-math.log(3/4),-math.log(3/4),-torch.inf]])
    expected = torch.nn.functional.cross_entropy(logits,torch.tensor([0]))
    assert torch.allclose(actual,expected)


def test_spotify_aliases_unknown_ids_similar_to_and_discovery_full_past():
    records = [{"m4a_id":"a","spotify_id":"same","artist":"old"},
               {"m4a_id":"alias","spotify_id":"same","artist":"old"},
               {"m4a_id":"b","spotify_id":"b","artist":"old"},
               {"m4a_id":"c","spotify_id":"c","artist":"new"}]
    scorer = CatalogScorer(records,["exact","similar_to"])
    row = {"query_id":"q","query_type":"exact","target_m4a_id":"c"}
    ranking = ["unknown","alias","a","b","c","c"]
    seen = scorer.history_keys(["a","b"])
    result = scorer.score(ranking,row,seen)
    assert result["top20"] == ["alias","b","c"]
    assert result["ranking"] == .5 and result["discovery"] == 1.
    result = scorer.score(ranking,{**row,"query_type":"similar_to","exclude_m4a_id":"a","exclude_artist":"old"},seen)
    assert result["top20"] == ["c"] and result["ranking"] == 1.
    assert not scorer.score(ranking,{**row,"target_m4a_id":"alias"},seen)["is_new_canonical"]


def test_macro_equal_type_weight_missing_discovery_is_not_zero():
    scorer = CatalogScorer([{"m4a_id":"a"},{"m4a_id":"b"}],["one","two"])
    rows = [scorer.score(["a","b"],{"query_id":str(i),"query_type":"one","target_m4a_id":"a"},set()) for i in range(4)]
    rows.append(scorer.score(["a"],{"query_id":"x","query_type":"two","target_m4a_id":"b"}, {"m4a:b"}))
    summary,_,_ = scorer.summarize(rows)
    assert summary["macro_ranking_ndcg20"] == .5
    assert summary["macro_discovery_ndcg20"] is None
    assert summary["macro_diversity"] is None


def test_validation_is_fixed_balanced_and_uses_only_holdout():
    frame = pd.DataFrame({"query_type":["a"]*8+["b"]*8,"is_new":[False,True]*8})
    rows = np.arange(2,16)
    first = balanced_validation(frame,rows,4,42)
    assert np.array_equal(first,balanced_validation(frame,rows,4,42))
    assert set(first) <= set(rows) and len(first) == 8
    assert frame.iloc[first].groupby("query_type").is_new.mean().eq(.5).all()


def toy_data(path):
    path.mkdir(exist_ok=True)
    rng = np.random.default_rng(2)
    records = [{"m4a_id":str(i),"artist":"artist"+str(i),"energy":i/8} for i in range(8)]
    prep = TrackPreprocessor().fit(records)
    cat,nums,_ = prep.transform(records)
    audio = rng.normal(size=(8,128)).astype(np.float32)
    packed = pack_features(audio,rng.normal(size=(8,6)),rng.normal(size=(8,6)),cat,nums,np.ones((8,5)))
    indices,weights = HistoryHasher(32).transform(records)
    pairs = pd.DataFrame({"query_id":[str(i) for i in range(8)],"query_type":["a","b"]*4,
        "target_index":np.arange(8),"target_m4a_id":[str(i) for i in range(8)],
        "user_index":[0]*6+[1]*2,"user_id":[1]*6+[9]*2})
    atomic_json(path/"item_args.json",{"text_dim":6,"num_categories":len(prep.vocabulary),"embedding_dim":8,"dropout":.1})
    h,mask = padded_histories([[0,1,0],[2,3]])
    return PreparedData(records,pairs,np.arange(6),np.arange(6,8),np.arange(6,8),h,mask,{9:frozenset()},
        np.array([[4+i,5,6] for i in range(8)]),np.ones((8,3),bool),packed,indices,weights,audio,
        np.column_stack((nums,np.ones((8,2)))).astype(np.float32),prep,CatalogScorer(records,["a","b"]),"toy",path)


def make_toy_trainer(path,data,resume=True):
    query = tiny_tower()
    model = QwenTwoTower(query,TrackTransformerTower(text_dim=6,num_categories=len(data.preprocessor.vocabulary),embedding_dim=8,dropout=.1))
    cfg = TrainConfig(hash_buckets=32,embedding_dim=8,negatives=8,micro_batch=2,accumulation=2,
        require_cuda=False,min_free_vram_gib=0,max_hours=1,export_reserve_seconds=0,resume=resume)
    return Trainer(model,data,cfg,path,torch.device("cpu"),torch.float32)


def test_mid_epoch_resume_replays_exact_optimizer_and_rng(tmp_path):
    data = toy_data(tmp_path/"cache")
    trainer = make_toy_trainer(tmp_path/"artifacts",data)
    trainer.optimizer_step(np.array([0,1]))
    trainer.step,trainer.cursor = 1,2
    trainer.save()
    uninterrupted_loss = trainer.optimizer_step(np.array([2,3]))
    expected = trainer._state()
    resumed = make_toy_trainer(tmp_path/"artifacts",data)
    assert resumed.step == 1 and resumed.cursor == 2
    replay_loss = resumed.optimizer_step(np.array([2,3]))
    assert abs(uninterrupted_loss-replay_loss) < 1e-7
    for part in expected:
        for name,value in expected[part].items():
            assert torch.equal(value,resumed._state()[part][name]), name


def test_benchmark_restores_weights_and_does_not_train(tmp_path):
    data = toy_data(tmp_path/"cache")
    trainer = make_toy_trainer(tmp_path/"artifacts",data)
    before = trainer._state()
    result = trainer.benchmark(steps=2)
    assert result["examples_per_second"] > 0 and trainer.step == 0
    assert not trainer.optimizer.state
    for part in before:
        for name,value in before[part].items():
            assert torch.equal(value,trainer._state()[part][name])


def test_overflow_retry_restores_both_dropout_and_negative_rng(tmp_path,monkeypatch):
    data = toy_data(tmp_path/"cache")
    trainer = make_toy_trainer(tmp_path/"artifacts",data)
    attempts = []
    def attempt(rows):
        attempts.append((torch.rand(3),torch.randint(100,(3,),generator=trainer.sampler)))
        if len(attempts) == 1:
            raise ScaleRetry()
        return 1.
    monkeypatch.setattr(trainer,"_optimizer_step_once",attempt)
    assert trainer.optimizer_step(np.array([0,1])) == 1.
    assert all(torch.equal(a,b) for a,b in zip(attempts[0],attempts[1]))


def test_budget_includes_preparation_and_persists_active_time(tmp_path,monkeypatch):
    from recsys.two_tower_qwen import data as module
    now = [100.]
    monkeypatch.setattr(module.time,"monotonic",lambda:now[0])
    dataset = tmp_path/"dataset"
    dataset.mkdir()
    (dataset/"data.parquet").write_bytes(b"fixture")
    cfg = TrainConfig(max_hours=.02,export_reserve_seconds=0)
    first = ExperimentBudget(tmp_path/"artifacts",dataset,cfg,started_at=100.)
    now[0] = 130.
    first.check()
    now[0] = 150.
    resumed = ExperimentBudget(tmp_path/"artifacts",dataset,cfg,started_at=150.)
    now[0] = 170.
    assert resumed.elapsed() == 50.
    now[0] = 200.
    with pytest.raises(TimeoutError):
        resumed.check()


def test_hardware_probe_restores_weights_optimizer_and_rng(tmp_path):
    from recsys.two_tower_qwen.benchmarking import BenchmarkSpec, run_benchmark, results_table
    trainer = make_toy_trainer(tmp_path/"artifacts", toy_data(tmp_path/"cache"), resume=False)
    before = trainer._state()
    rng, sampler = torch.get_rng_state(), trainer.sampler.get_state()
    spec = BenchmarkSpec(effective_batch=4, micro_batches=(4,2), warmup_steps=1,
                         measured_steps=2, repeats=2, max_minutes=2, validation_queries=0)
    result = run_benchmark(trainer,spec,tmp_path/"report.json","CPU check",234.)
    assert result["status"] == "completed" and all(m["status"] == "ok" for m in result["modes"])
    assert all(len(m["trials"]) == 2 for m in result["modes"])
    assert trainer.step == 0 and trainer.micro_batch == 2 and not trainer.optimizer.state
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(sampler,trainer.sampler.get_state())
    assert results_table([result])["rub/100k"].isna().all()
    for part in before:
        for name,value in before[part].items():
            assert torch.equal(value,trainer._state()[part][name]),name


def test_hardware_probe_oom_does_not_change_workload(tmp_path,monkeypatch):
    from recsys.two_tower_qwen.benchmarking import BenchmarkSpec, run_benchmark
    trainer = make_toy_trainer(tmp_path/"artifacts",toy_data(tmp_path/"cache"),resume=False)
    real = trainer.optimizer_step
    def limited(rows):
        if trainer.micro_batch == 4:
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        return real(rows)
    monkeypatch.setattr(trainer,"optimizer_step",limited)
    spec = BenchmarkSpec(effective_batch=4,micro_batches=(4,2),warmup_steps=1,
                         measured_steps=2,repeats=1,max_minutes=2,validation_queries=0)
    result = run_benchmark(trainer,spec,tmp_path/"report.json","CPU check",234.)
    assert [m["status"] for m in result["modes"]] == ["oom","ok"]
    assert result["modes"][0]["micro_batch"] == 4
    assert all(m["effective_batch"] == 4 for m in result["modes"])
    assert not trainer.optimizer.state


def test_hardware_probe_time_limit_saves_partial_result(tmp_path):
    from recsys.two_tower_qwen.benchmarking import BenchmarkSpec, run_benchmark
    import json
    trainer = make_toy_trainer(tmp_path/"artifacts",toy_data(tmp_path/"cache"),resume=False)
    spec = BenchmarkSpec(effective_batch=4,micro_batches=(4,),warmup_steps=1,
                         measured_steps=2,repeats=1,max_minutes=1e-9,validation_queries=0)
    path = tmp_path/"report.json"
    result = run_benchmark(trainer,spec,path,"CPU check",234.)
    assert result["status"] == "time_limit"
    assert json.loads(path.read_text())["modes"][0]["status"] == "time_limit"
    assert not trainer.optimizer.state


def test_hardware_economics_threshold_and_compatibility():
    from recsys.two_tower_qwen.benchmarking import compare_pair
    def report(label,price,speed):
        return {"comparison_key":"same", "label":label,"rub_per_hour":price,
            "environment":{"device_type":"cuda"},"modes":[{"status":"ok","kind":"fixed",
            "micro_batch":4,"queries_per_second":speed,"repeat_speed_min":speed,"repeat_speed_max":speed}]}
    l4,a100 = report("L4",234.,10.),report("A100",542.88,25.)
    result = compare_pair(l4,a100)
    assert result["second_speedup"] == 2.5
    assert result["break_even_speedup"] == pytest.approx(2.32)
    assert result["second_cost_ratio"] == pytest.approx(.928)
    assert result["cheaper_training_work"] == "A100"
    a100["modes"][0]["repeat_speed_min"] = 20.
    assert compare_pair(l4,a100)["cheaper_training_work"].startswith("uncertain")
    a100["comparison_key"] = "other"
    with pytest.raises(ValueError,match="differ"):
        compare_pair(l4,a100)
    a100["comparison_key"] = "same"
    a100["environment"]["device_type"] = "cpu"
    with pytest.raises(ValueError,match="CPU"):
        compare_pair(l4,a100)


def test_hardware_workload_is_fixed_across_micro_batches(tmp_path):
    from recsys.two_tower_qwen.benchmarking import BenchmarkSpec, benchmark_workload
    trainer = make_toy_trainer(tmp_path/"artifacts",toy_data(tmp_path/"cache"),resume=False)
    spec = BenchmarkSpec(effective_batch=4,micro_batches=(4,),warmup_steps=1,measured_steps=2,repeats=1)
    rows,identity,_ = benchmark_workload(trainer,spec)
    trainer.micro_batch = 1
    replay,replay_identity,_ = benchmark_workload(trainer,spec)
    assert np.array_equal(rows,replay) and identity == replay_identity
    assert set(rows) <= set(trainer.data.fit_rows)
    trainer.config.negatives += 1
    assert identity != benchmark_workload(trainer,spec)[1]


def test_full_fit_can_extend_epochs_without_losing_optimizer(tmp_path):
    data = toy_data(tmp_path / "cache")
    trainer = make_toy_trainer(tmp_path / "artifacts", data)
    trainer.config.epochs = 1
    trainer.config.validate_every_steps = 1
    trainer.fit()
    assert trainer.epoch == 1 and trainer.step == 2
    assert (trainer.run_dir / "best.pt").exists()
    resumed = make_toy_trainer(tmp_path / "artifacts", data)
    assert resumed.epoch == 1 and resumed.step == 2 and resumed.optimizer.state
    resumed.config.epochs = 2
    resumed.config.validate_every_steps = 1
    resumed.fit()
    assert resumed.epoch == 2 and resumed.step == 4
    assert len(list(resumed.run_dir.glob("step-*.pt"))) == 3
    assert (resumed.run_dir / "best.pt").exists() and (resumed.run_dir / "last.pt").exists()


def test_early_stopping_has_first_epoch_grace_and_counts_checks(tmp_path, monkeypatch):
    trainer = make_toy_trainer(tmp_path / "artifacts", toy_data(tmp_path / "cache"))
    trainer.config.epochs = 3
    trainer.config.patience = 2
    trainer.config.validate_every_steps = 1
    def constant_metric(rows):
        return {"macro_ranking_ndcg20": 0.0}, pd.DataFrame({"score": [0.0]}), pd.DataFrame({"type": ["a"]})
    monkeypatch.setattr(trainer, "evaluate", constant_metric)
    trainer.fit()
    assert trainer.epoch == 1 and trainer.step == 3  # never stop inside epoch zero
    assert trainer.history[0]["cursor"] == 4 and trainer.stale == 2
    assert __import__("json").loads((trainer.run_dir / "status.json").read_text())["reason"] == "early_stopping"


def test_checkpoint_callback_sees_complete_checkpoint_and_resumed_micro_keeps_batch(tmp_path):
    trainer = make_toy_trainer(tmp_path / "artifacts", toy_data(tmp_path / "cache"))
    committed = []
    def after_save(t):
        payload = torch.load(t.run_dir / "last.pt", map_location="cpu", weights_only=True)
        committed.append(payload["step"])
        assert not (t.run_dir / "last.pt.tmp").exists()
    trainer.checkpoint_callback = after_save
    trainer.micro_batch = 1
    trainer.save()
    resumed = make_toy_trainer(tmp_path / "artifacts", trainer.data)
    assert committed == [0] and resumed.micro_batch == 1 and resumed.effective_batch == 4
    resumed.config.epochs = 1
    resumed.fit()
    assert resumed.step == 2  # 4 + 2 examples, not 2 + 2 + 2


def test_probe_oom_restores_updates_and_preserves_effective_batch(tmp_path, monkeypatch):
    trainer = make_toy_trainer(tmp_path / "artifacts", toy_data(tmp_path / "cache"))
    before = trainer._state()
    real = trainer.optimizer_step
    attempts = []
    def oom_after_update(rows):
        attempts.append((trainer.micro_batch, len(rows)))
        if len(attempts) == 2:
            raise torch.cuda.OutOfMemoryError("probe failed after an earlier successful update")
        return real(rows)
    monkeypatch.setattr(trainer, "optimizer_step", oom_after_update)
    result = trainer.benchmark(steps=2)
    assert result["effective_batch"] == 4 and result["micro_batch"] == 1
    assert all(n == 4 for _, n in attempts)
    assert not trainer.optimizer.state
    for part, values in before.items():
        for name, value in values.items():
            assert torch.equal(value, trainer._state()[part][name])


def test_best_bundle_reload_matches_query_and_hnsw(tmp_path, monkeypatch):
    import json
    from recsys.two_tower_qwen import bundle
    trainer = make_toy_trainer(tmp_path / "artifacts", toy_data(tmp_path / "cache"))
    trainer.config.epochs = 1
    trainer.fit()
    class ToyTokenizer:
        padding_side = "right"
        def save_pretrained(self, path):
            path.mkdir()
            (path / "tokenizer_config.json").write_text('{"fixture":true}')
        def __call__(self, texts, **kwargs):
            return {"input_ids": torch.tensor([[4,5,6]]), "attention_mask": torch.ones(1,3,dtype=torch.long)}
    tokenizer = ToyTokenizer()
    exported = bundle.export_bundle(trainer, tokenizer)
    monkeypatch.setattr(bundle, "load_backbone", lambda *args, **kwargs: (tiny_tower().backbone, tokenizer))
    monkeypatch.setattr(bundle.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    retriever = bundle.QwenRetriever(exported, device="cpu")
    trainer.model.eval()
    with torch.no_grad():
        expected = trainer.model.query_tower(**trainer.inputs(np.array([0])))[0].numpy()
    actual = retriever.encode_query("English query", ["0", "1", "0"])
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    assert len(retriever.search("English query", ["0", "1", "0"], k=3)) == 3
    manifest = json.loads((exported / "manifest.json").read_text())
    assert manifest["dimension"] == 8 and manifest["tracks"] == 8
    (exported / "catalog.json").write_text('[]')
    with pytest.raises(ValueError, match="integrity"):
        bundle.QwenRetriever(exported, device="cpu")
