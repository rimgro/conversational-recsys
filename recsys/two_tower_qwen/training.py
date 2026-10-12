"""Joint training, bounded validation, trainable-only checkpoints and exact resume."""
from pathlib import Path
import contextlib
import json
import time

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from recsys.track_tower import TrackTransformerTower
from .data import atomic_json, digest, file_hash
from .model import QwenHistoryTower, QwenTwoTower, load_backbone, sampled_softmax_loss


def select_device(config):
    if config.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. CPU is only for an explicitly configured smoke test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"GPU: {torch.cuda.get_device_name()}; free {free / 1024**3:.2f} / {total / 1024**3:.2f} GiB")
        if free / 1024**3 < config.min_free_vram_gib:
            raise RuntimeError("Not enough free VRAM; use a dedicated training GPU")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print("Compute dtype:", dtype)
    return device, dtype


def make_model(config, data, device, dtype, cache_dir, backbone=None, tokenizer=None):
    torch.manual_seed(config.seed)
    if backbone is None:
        backbone, tokenizer = load_backbone(config, dtype, cache_dir, tokenizer=tokenizer)
    tower = QwenHistoryTower(backbone, config.hash_buckets, config.embedding_dim)
    tower.initialize_markers(tokenizer)
    item_args = json.loads((data.cache_dir / "item_args.json").read_text())
    model = QwenTwoTower(tower, TrackTransformerTower(**item_args)).to(device)
    bad = [name for name, p in backbone.named_parameters() if p.requires_grad and "lora_" not in name]
    if bad or backbone.get_input_embeddings().weight.requires_grad:
        raise RuntimeError(f"Original Qwen weights unexpectedly trainable: {bad}")
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model, tokenizer


class BudgetExpired(TimeoutError):
    pass


class ScaleRetry(FloatingPointError):
    pass


class Trainer:
    def __init__(self, model, data, config, artifact_dir, device, dtype, started_at=None, budget=None,
                 checkpoint_callback=None):
        config.validate()
        self.model, self.data, self.config = model, data, config
        self.device, self.dtype = device, dtype
        self.started_at = time.monotonic() if started_at is None else started_at
        self.budget = budget
        self.checkpoint_callback = checkpoint_callback
        package = Path(__file__).parent
        code_hash = {p.name: file_hash(p) for p in sorted(package.glob("*.py"))}
        code_hash["track_tower.py"] = file_hash(package.parent / "track_tower.py")
        self.signature = digest({"data": data.fingerprint, "config": config.identity(), "code": code_hash})
        self.run_dir = Path(artifact_dir) / "runs" / self.signature[:20]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(self.run_dir / "config.json", config.to_dict())
        atomic_json(self.run_dir / "identity.json", {"signature": self.signature, "data": data.fingerprint,
                                                  "config": config.identity(), "code_sha256": code_hash})
        groups = {"lora": [], "new": []}
        for name, p in model.named_parameters():
            if p.requires_grad:
                groups["lora" if "lora_" in name else "new"].append(p)
        self.optimizer = torch.optim.AdamW([
            {"params": groups["lora"], "lr": config.lora_lr, "name": "lora"},
            {"params": groups["new"], "lr": config.new_weights_lr, "name": "new"}], weight_decay=config.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and dtype == torch.float16))
        self.sampler = torch.Generator(device=device).manual_seed(config.seed + 10000)
        self.item_features = torch.tensor(np.array(data.item_features), device=device)
        self.hash_indices = torch.tensor(data.hash_indices, device=device)
        self.hash_weights = torch.tensor(data.hash_weights, device=device)
        self.audio = torch.tensor(data.audio, device=device)
        self.numbers = torch.tensor(data.history_numbers, device=device)
        self.record_groups = torch.tensor(data.scorer.group_ids, device=device)
        self.step = self.epoch = self.cursor = 0
        self.best_score, self.stale = -1., 0
        self.elapsed_previous = 0.
        self.history = []
        self.micro_batch = config.micro_batch
        self.effective_batch = config.micro_batch * config.accumulation
        self.running_loss = self.running_examples = 0.
        self.last_validation_step = -1
        if config.resume and (self.run_dir / "last.pt").is_file():
            self.restore(self.run_dir / "last.pt")
        elif not config.resume and (self.run_dir / "last.pt").exists():
            raise ValueError("Existing run found. Resume it or use a different ARTIFACT_DIR")

    def elapsed(self):
        if self.budget is not None:
            return self.budget.elapsed()
        return self.elapsed_previous + time.monotonic() - self.started_at

    def check_budget(self, reserve=False):
        if self.budget is not None:
            self.budget.record()
        limit = self.config.max_hours * 3600 - (self.config.export_reserve_seconds if reserve else 0)
        if self.elapsed() >= limit:
            raise BudgetExpired("Experiment wall-time budget reached")

    def autocast(self):
        return torch.autocast("cuda", dtype=self.dtype) if self.device.type == "cuda" else contextlib.nullcontext()

    def inputs(self, rows):
        selected = self.data.pairs.iloc[rows]
        user_indices = selected.user_index.to_numpy()
        mask = self.data.history_mask[user_indices]
        width = max(1, int(mask.sum(1).max()))
        h = torch.tensor(self.data.histories[user_indices, :width], device=self.device)
        return {"hash_indices": self.hash_indices[h], "hash_weights": self.hash_weights[h],
                "audio": self.audio[h], "numbers": self.numbers[h],
                "history_mask": torch.tensor(mask[:, :width], device=self.device),
                "query_ids": torch.tensor(np.asarray(self.data.query_ids[rows]), dtype=torch.long, device=self.device),
                "query_mask": torch.tensor(np.asarray(self.data.query_mask[rows]), device=self.device)}

    def optimizer_step(self, rows):
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if self.device.type == "cuda" else []
        sampler_rng = self.sampler.get_state()
        for attempt in range(8):
            try:
                return self._optimizer_step_once(rows)
            except ScaleRetry:
                # GradScaler lowered its scale and skipped the update. Replay the
                # same random negatives/dropout instead of silently losing examples.
                torch.set_rng_state(cpu_rng)
                if cuda_rng:
                    torch.cuda.set_rng_state_all(cuda_rng)
                self.sampler.set_state(sampler_rng)
                self.optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Repeated FP16 overflows after reducing the loss scale")

    def _optimizer_step_once(self, rows):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.
        for start in range(0, len(rows), self.micro_batch):
            selected = rows[start:start + self.micro_batch]
            positive_ids = torch.tensor(self.data.pairs.iloc[selected].target_index.to_numpy(), device=self.device)
            negative_ids = torch.randint(len(self.data.catalog), (self.config.negatives,),
                                          device=self.device, generator=self.sampler)
            # Encode each distinct track once; repeated IID draws retain multiplicity in logits.
            ids, inverse = torch.unique(torch.cat((positive_ids, negative_ids)), return_inverse=True)
            with self.autocast():
                query = self.model.query_tower(**self.inputs(selected))
                candidates = self.model.item(self.item_features[ids])
            # Keep similarity/logQ/softmax in float32 outside autocast.
            positive = candidates[inverse[:len(selected)]]
            negative = candidates[inverse[len(selected):]]
            loss = sampled_softmax_loss(query, positive, negative, self.record_groups[positive_ids],
                self.record_groups[negative_ids], len(self.data.catalog), self.config.temperature)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite sampled softmax loss")
            self.scaler.scale(loss * len(selected) / len(rows)).backward()
            total_loss += float(loss.detach()) * len(selected)
        self.scaler.unscale_(self.optimizer)
        norm = nn.utils.clip_grad_norm_((p for p in self.model.parameters() if p.requires_grad),
            self.config.max_grad_norm, error_if_nonfinite=not self.scaler.is_enabled())
        old_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scaler.is_enabled() and self.scaler.get_scale() < old_scale:
            raise ScaleRetry("FP16 overflow; retrying the same examples at a lower loss scale")
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient norm")
        return total_loss / len(rows)

    def _state(self):
        return {"query": self.model.query_tower.trainable_state(),
                "item": {k: v.detach().cpu().clone() for k, v in self.model.item_tower.state_dict().items()}}

    def _load_state(self, state):
        self.model.query_tower.load_trainable_state(state["query"])
        self.model.item_tower.load_state_dict(state["item"], strict=True)

    def save(self, name="last.pt"):
        if self.budget is not None:
            self.budget.record()
        payload = {"signature": self.signature, "weights": self._state(),
            "optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict(),
            "step": self.step, "epoch": self.epoch, "cursor": self.cursor,
            "micro_batch": self.micro_batch, "best_score": self.best_score, "stale": self.stale,
            "running_loss": self.running_loss, "running_examples": self.running_examples,
            "last_validation_step": self.last_validation_step, "history": self.history,
            "elapsed_seconds": self.elapsed(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
            "sampler_rng": self.sampler.get_state()}
        temporary = self.run_dir / (name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(self.run_dir / name)
        if name == "last.pt" and self.checkpoint_callback is not None:
            self.checkpoint_callback(self)

    def restore(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["signature"] != self.signature:
            raise ValueError("Checkpoint identity mismatch")
        self._load_state(state["weights"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        for attr in ("step", "epoch", "cursor", "micro_batch", "best_score", "stale",
                     "running_loss", "running_examples", "last_validation_step", "history"):
            setattr(self, attr, state[attr])
        self.elapsed_previous = state["elapsed_seconds"]
        torch.set_rng_state(state["torch_rng"])
        if self.device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        self.sampler.set_state(state["sampler_rng"])
        print(f"Resume: epoch {self.epoch}, cursor {self.cursor}, step {self.step}")

    def benchmark(self, steps=5):
        path = self.run_dir / "benchmark.json"
        if path.is_file():
            return json.loads(path.read_text())
        self.check_budget(reserve=True)
        self.save("pre_benchmark.pt")
        offset_before = self.elapsed_previous
        try:
            while True:
                # An OOM may reduce the micro-batch, never the effective batch.
                rows = self.data.fit_rows[:self.effective_batch]
                try:
                    if self.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    durations = []
                    for _ in range(steps):
                        self.check_budget(reserve=True)
                        if self.device.type == "cuda":
                            torch.cuda.synchronize()
                        started = time.perf_counter()
                        self.optimizer_step(rows)
                        if self.device.type == "cuda":
                            torch.cuda.synchronize()
                        durations.append(time.perf_counter() - started)
                    break
                except torch.cuda.OutOfMemoryError:
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.micro_batch == 1:
                        raise RuntimeError("OOM even with micro_batch=1; reduce negatives or use a larger GPU")
                    chosen_micro = max(1, self.micro_batch // 2)
                    # Earlier probe steps may already have updated the model.
                    self.restore(self.run_dir / "pre_benchmark.pt")
                    self.elapsed_previous = offset_before
                    self.micro_batch = chosen_micro
                    torch.cuda.empty_cache()
                    print("Retry benchmark with micro_batch:", self.micro_batch)
            chosen_micro = self.micro_batch
            speed = len(rows) / float(np.mean(durations[1:] or durations))
            result = {"micro_batch": chosen_micro, "effective_batch": len(rows), "examples_per_second": speed,
                "estimated_training_epoch_hours": len(self.data.fit_rows) / speed / 3600,
                "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3 if self.device.type == "cuda" else None,
                "includes_validation": False, "dtype": str(self.dtype), "steps": steps}
        finally:
            # Restore weights, optimizer AND random states; benchmark is not training.
            chosen_micro = self.micro_batch
            self.restore(self.run_dir / "pre_benchmark.pt")
            self.elapsed_previous = offset_before  # current invocation already includes benchmark time
            self.micro_batch = chosen_micro
            self.optimizer.zero_grad(set_to_none=True)
        atomic_json(path, result)
        print("Benchmark:", result)
        return result

    @torch.inference_mode()
    def item_vectors(self):
        self.model.eval()
        parts = []
        for start in range(0, len(self.data.catalog), self.config.item_evaluation_batch):
            self.check_budget()
            with self.autocast():
                parts.append(self.model.item(self.item_features[start:start + self.config.item_evaluation_batch]).float())
        return torch.cat(parts)

    @torch.inference_mode()
    def evaluate(self, rows):
        vectors = self.item_vectors()
        result = []
        for start in tqdm(range(0, len(rows), self.config.evaluation_batch), desc="Full-catalog validation"):
            self.check_budget()
            selected = rows[start:start + self.config.evaluation_batch]
            with self.autocast():
                query = self.model.query_tower(**self.inputs(selected)).float()
            scores = query @ vectors.T
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Invalid retrieval scores")
            rankings = torch.argsort(scores, dim=1, descending=True, stable=True).cpu().numpy()
            for row_no, ranking in zip(selected, rankings):
                row = self.data.pairs.iloc[row_no].to_dict()
                ordered_ids = [self.data.scorer.ids[i] for i in ranking]
                result.append(self.data.scorer.score(ordered_ids, row, self.data.listened[int(row["user_id"])]))
        return self.data.scorer.summarize(result)

    def validate(self):
        summary, rows, by_type = self.evaluate(self.data.validation_rows)
        score = summary["macro_ranking_ndcg20"]
        if score is None:
            raise ValueError("Macro-Ranking requires coverage of every query type")
        record = {"step": self.step, "epoch": self.epoch, "cursor": self.cursor,
            "train_loss": self.running_loss / max(1, self.running_examples),
            "elapsed_hours": self.elapsed() / 3600, **summary}
        self.history.append(record)
        self.last_validation_step = self.step
        improved = score > self.best_score
        self.best_score = max(self.best_score, score)
        progress = self.epoch + self.cursor / max(1, len(self.data.fit_rows))
        self.stale = (0 if improved or progress < self.config.early_stopping_min_epochs
                      else self.stale + 1)
        rows.to_parquet(self.run_dir / f"validation-step-{self.step}.parquet", index=False)
        by_type.to_csv(self.run_dir / f"types-step-{self.step}.csv")
        if improved:
            self.save("best.pt")
            atomic_json(self.run_dir / "best_metrics.json", summary)
        self.save(f"step-{self.step}.pt")
        checkpoints = sorted(self.run_dir.glob("step-*.pt"),
                             key=lambda p: int(p.stem.split("-")[1]), reverse=True)
        for old in checkpoints[self.config.keep_validation_checkpoints:]:
            old.unlink()
        self.save()
        atomic_json(self.run_dir / "history.json", self.history)
        print(record)
        return summary

    def should_stop_early(self):
        # The first complete epoch is a grace period for randomly initialized
        # history projections and the item tower. Patience begins after it.
        progress = self.epoch + self.cursor / max(1, len(self.data.fit_rows))
        return (progress >= self.config.early_stopping_min_epochs
                and self.stale >= self.config.patience)

    def fit(self):
        reason = "epochs_completed"
        self.save()  # transaction boundary before the first update, including a fresh run
        try:
            while self.epoch < self.config.epochs:
                if self.should_stop_early():
                    reason = "early_stopping"
                    break
                order = np.random.default_rng(self.config.seed + self.epoch).permutation(self.data.fit_rows)
                batch = self.effective_batch
                while self.cursor < len(order):
                    self.check_budget(reserve=True)
                    selected = order[self.cursor:self.cursor + batch]
                    loss = self.optimizer_step(selected)
                    self.cursor += len(selected)
                    self.step += 1
                    self.running_loss += loss * len(selected)
                    self.running_examples += len(selected)
                    if self.step % 20 == 0 or self.step == 1:
                        print(f"Step {self.step}: loss={loss:.4f}, elapsed={self.elapsed()/3600:.2f} h", flush=True)
                    if self.step % self.config.checkpoint_every_steps == 0:
                        self.save()
                    if self.step % self.config.validate_every_steps == 0:
                        self.validate()
                        if self.should_stop_early():
                            break
                if self.last_validation_step != self.step:
                    self.validate()
                if self.cursor < len(order):
                    reason = "early_stopping"
                    break
                self.epoch += 1
                self.cursor = 0
                self.running_loss = self.running_examples = 0.
                self.save()
        except BudgetExpired:
            reason = "time_budget"
            self.save()
            if self.step and self.last_validation_step != self.step:
                try:
                    self.validate()  # use the reserved time on a complete fixed validation set
                except BudgetExpired:
                    print("Validation interrupted by budget; partial results are not used for checkpoint selection")
        except KeyboardInterrupt:
            reason = "interrupted"
            # An interrupt can arrive halfway through an optimizer update. Roll back
            # to the last COMPLETE checkpoint rather than publish partial state.
            spent = self.elapsed()
            self.restore(self.run_dir / "last.pt")
            self.elapsed_previous, self.started_at = spent, time.monotonic()
            self.optimizer.zero_grad(set_to_none=True)
            print("Interrupted: restored the last complete checkpoint; some recent steps may need replay")
        self.save()
        atomic_json(self.run_dir / "status.json", {"reason": reason, "step": self.step,
            "elapsed_hours": self.elapsed()/3600, "best_macro_ranking": self.best_score if self.best_score >= 0 else None})
        return self.history

    def load_best(self):
        path = self.run_dir / "best.pt"
        if not path.is_file():
            raise RuntimeError("No completely evaluated checkpoint yet; resume training before export")
        state = torch.load(path, map_location="cpu", weights_only=True)
        self._load_state(state["weights"])
        self.model.eval()
        return state["step"]
