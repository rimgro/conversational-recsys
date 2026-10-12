"""Matched-workload training benchmarks; never infer GPU quality from CPU checks."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from .data import atomic_json, digest, file_hash


@dataclass
class BenchmarkSpec:
    effective_batch: int = 32
    micro_batches: tuple = (4, 8, 16, 32)
    warmup_steps: int = 3
    measured_steps: int = 8
    repeats: int = 3
    max_minutes: float = 30.
    validation_queries: int = 128

    def validate(self):
        if self.effective_batch < 1 or self.warmup_steps < 1 or self.measured_steps < 2 or self.repeats < 1:
            raise ValueError("Need positive batch/repeats, warmup and at least two measured steps")
        if not self.micro_batches or self.micro_batches[0] != 4:
            raise ValueError("The first mode must be the fixed micro-batch=4 baseline")
        if any(m < 1 or self.effective_batch % m for m in self.micro_batches):
            raise ValueError("Micro-batches must divide the effective batch")
        if len(set(self.micro_batches)) != len(self.micro_batches) or self.max_minutes <= 0 or self.validation_queries < 0:
            raise ValueError("Invalid benchmark limits")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def environment(device, dtype):
    versions = {}
    for package in ("torch", "transformers", "peft", "numpy", "pandas"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {"device_type": device.type, "dtype": str(dtype), "versions": versions,
              "python": platform.python_version(), "cuda_runtime": torch.version.cuda,
              "cpu_count_os": os.cpu_count(), "torch_cpu_threads": torch.get_num_threads(),
              "execution_flags": {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                  "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
              "gpu_name": None, "total_vram_gib": None, "nvidia_smi": None}
    try:
        import psutil
        result["system_ram_gib"] = psutil.virtual_memory().total / 1024**3
    except ImportError:
        result["system_ram_gib"] = None
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update(gpu_name=props.name, total_vram_gib=props.total_memory / 1024**3,
                      compute_capability=[props.major, props.minor])
        try:
            result["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return result


def benchmark_workload(trainer, spec):
    spec.validate()
    if len(trainer.data.fit_rows) < spec.effective_batch:
        raise ValueError("Training split is smaller than the effective batch")
    # Draw from the complete training split, independently of hardware and mode.
    count = (spec.warmup_steps + spec.measured_steps) * spec.effective_batch
    rng = np.random.default_rng(trainer.config.seed + 70000)
    rows = np.concatenate([rng.permutation(trainer.data.fit_rows)
                           for _ in range(int(np.ceil(count / len(trainer.data.fit_rows))))])[:count]
    cfg = trainer.config.identity()
    for key in ("micro_batch", "accumulation", "require_cuda", "min_free_vram_gib"):
        cfg.pop(key, None)
    package = Path(__file__).parent
    code = {name: file_hash(package / name) for name in ("model.py", "features.py", "training.py", "benchmarking.py")}
    code["track_tower.py"] = file_hash(package.parent / "track_tower.py")
    users = trainer.data.pairs.iloc[rows].user_index.to_numpy()
    lengths = trainer.data.history_mask[users].sum(1)
    queries = np.asarray(trainer.data.query_mask[rows]).sum(1)
    identity = {"data": trainer.data.fingerprint, "config": cfg, "code": code,
                "row_sha256": digest(rows.tolist()), "effective_batch": spec.effective_batch,
                "warmup_steps": spec.warmup_steps, "measured_steps": spec.measured_steps,
                "repeats": spec.repeats}
    description = {"catalog_tracks": len(trainer.data.catalog), "fit_queries": len(trainer.data.fit_rows),
        "distinct_probe_queries": int(len(np.unique(rows))), "measured_queries_per_repeat": spec.measured_steps * spec.effective_batch,
        "history_events_mean": float(lengths.mean()), "history_events_max": int(lengths.max()),
        "query_tokens_mean": float(queries.mean()), "query_tokens_max": int(queries.max())}
    return rows, identity, description


def run_benchmark(trainer, spec, output_path, label, rub_per_hour=None, preparation=None,
                  *, price_per_hour=None, currency="RUB"):
    """Rollback each trial and always restore the model, optimizer and RNG on exit.

    Timing includes the existing optimizer_step input gathering, random negatives,
    forward/backward, accumulation, clipping and AdamW. No training validation or
    checkpoint I/O is included in the steady-state step time.
    """
    if price_per_hour is not None and rub_per_hour is not None:
        raise ValueError("Supply one price, not both")
    if price_per_hour is None:
        if currency != "RUB":
            raise ValueError("rub_per_hour is denominated in RUB")
        price_per_hour = rub_per_hour
    if price_per_hour is None or not np.isfinite(price_per_hour) or price_per_hour <= 0:
        raise ValueError("Price must be positive")
    if currency not in ("RUB", "USD"):
        raise ValueError("Unsupported currency")
    rows, identity, workload = benchmark_workload(trainer, spec)
    env = environment(trainer.device, trainer.dtype)
    output_path = Path(output_path)
    result = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "label": label,
        "price_per_hour": float(price_per_hour), "currency": currency, "price_source": "user_supplied_editable",
        "environment": env, "workload": workload, "identity": identity, "spec": asdict(spec),
        "comparison_key": digest({"identity": identity, "dtype": env["dtype"], "versions": env["versions"],
                                  "execution_flags": env["execution_flags"],
                                  "cuda_runtime": env["cuda_runtime"], "python": env["python"]}),
        "preparation": preparation or {}, "modes": [], "overhead": {}, "status": "running"}
    if currency == "RUB":
        result["rub_per_hour"] = float(price_per_hour)
    # This path belongs to a new, isolated benchmark Trainer, never the training run.
    origin = "hardware_probe_origin.pt"
    trainer.save(origin)
    original_micro, offset = trainer.micro_batch, trainer.elapsed_previous
    started = time.monotonic()
    deadline = started + spec.max_minutes * 60

    def deadline_check():
        if time.monotonic() >= deadline:
            raise TimeoutError("Measured benchmark time limit reached")

    def restore():
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.restore(trainer.run_dir / origin)
        trainer.elapsed_previous = offset
        gc.collect()
        if trainer.device.type == "cuda":
            torch.cuda.empty_cache()

    atomic_json(output_path, result)
    try:
        for micro in spec.micro_batches:
            mode = {"micro_batch": micro, "accumulation": spec.effective_batch // micro,
                "effective_batch": spec.effective_batch, "kind": "fixed" if micro == 4 else "tuned",
                "status": "running", "trials": []}
            result["modes"].append(mode)
            try:
                for repeat in range(spec.repeats):
                    deadline_check()
                    restore()
                    trainer.micro_batch = micro
                    for step in range(spec.warmup_steps):
                        deadline_check()
                        trainer.optimizer_step(rows[step*spec.effective_batch:(step+1)*spec.effective_batch])
                    synchronize(trainer.device)
                    if trainer.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(trainer.device)
                    durations, losses = [], []
                    for step in range(spec.warmup_steps, spec.warmup_steps + spec.measured_steps):
                        deadline_check()
                        synchronize(trainer.device)
                        before = time.perf_counter()
                        loss = trainer.optimizer_step(rows[step*spec.effective_batch:(step+1)*spec.effective_batch])
                        synchronize(trainer.device)
                        durations.append(time.perf_counter() - before)
                        losses.append(float(loss))
                    trial = {"repeat": repeat, "step_seconds": durations, "losses": losses,
                        "queries_per_second": spec.measured_steps * spec.effective_batch / sum(durations),
                        "peak_allocated_gib": torch.cuda.max_memory_allocated(trainer.device) / 1024**3 if trainer.device.type == "cuda" else None,
                        "peak_reserved_gib": torch.cuda.max_memory_reserved(trainer.device) / 1024**3 if trainer.device.type == "cuda" else None}
                    mode["trials"].append(trial)
                    atomic_json(output_path, result)
                timing = np.concatenate([t["step_seconds"] for t in mode["trials"]])
                speed = len(timing) * spec.effective_batch / float(timing.sum())
                mode.update(status="ok", queries_per_second=speed, mean_step_seconds=float(timing.mean()),
                    step_p10_seconds=float(np.percentile(timing, 10)), step_p90_seconds=float(np.percentile(timing, 90)),
                    repeat_speed_min=min(t["queries_per_second"] for t in mode["trials"]),
                    repeat_speed_max=max(t["queries_per_second"] for t in mode["trials"]),
                    training_epoch_hours=workload["fit_queries"] / speed / 3600,
                    cost_per_100k_queries=price_per_hour * 100000 / speed / 3600 if trainer.device.type == "cuda" else None)
                if currency == "RUB":
                    mode["rub_per_100k_queries"] = mode["cost_per_100k_queries"]
            except torch.cuda.OutOfMemoryError:
                mode.update(status="oom", reason="Micro-batch does not fit; not silently reduced")
                print(f"micro={micro}: OOM; moving to the next mode")
                trainer.optimizer.zero_grad(set_to_none=True)
                gc.collect()
                if trainer.device.type == "cuda":
                    torch.cuda.empty_cache()
            except TimeoutError:
                mode.update(status="time_limit")
                result["status"] = "time_limit"
                break
            finally:
                atomic_json(output_path, result)
            if mode["status"] == "ok":
                print(f"micro={micro}, accumulation={mode['accumulation']}: {mode['queries_per_second']:.2f} queries/s")
        if result["status"] != "time_limit":
            restore()
            deadline_check()
            # Measure startup/checkpoint and a bounded quality-evaluation operation separately.
            synchronize(trainer.device)
            before = time.perf_counter()
            trainer.save("hardware_probe_checkpoint.pt")
            synchronize(trainer.device)
            result["overhead"]["checkpoint_seconds"] = time.perf_counter() - before
            if spec.validation_queries:
                validation = trainer.data.validation_rows[:spec.validation_queries]
                before = time.perf_counter()
                metrics, _, _ = trainer.evaluate(validation)
                synchronize(trainer.device)
                result["overhead"].update(validation_queries=len(validation),
                    validation_seconds=time.perf_counter() - before,
                    validation_catalog_tracks=workload["catalog_tracks"], validation_metrics=metrics)
            result["status"] = "completed"
    except TimeoutError:
        result["status"] = "time_limit"
    except KeyboardInterrupt:
        result["status"] = "interrupted"
        raise
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            restore()
            trainer.micro_batch = original_micro
            trainer.model.eval()
        finally:
            result["benchmark_wall_seconds"] = time.monotonic() - started
            atomic_json(output_path, result)
    return result


def results_table(reports):
    rows = []
    for report in reports:
        for mode in report["modes"]:
            trials = mode["trials"]
            row = {"run": report["label"], "GPU": report["environment"]["gpu_name"] or "CPU check",
                "dtype": report["environment"]["dtype"], "micro": mode["micro_batch"],
                "accumulation": mode["accumulation"], "status": mode["status"], "kind": mode["kind"],
                "queries/s": mode.get("queries_per_second"), "step, s": mode.get("mean_step_seconds"),
                "epoch train, h": mode.get("training_epoch_hours"), "rub/100k": mode.get("rub_per_100k_queries"),
                "peak allocated, GiB": max((t["peak_allocated_gib"] or 0 for t in trials), default=0) or None,
                "peak reserved, GiB": max((t["peak_reserved_gib"] or 0 for t in trials), default=0) or None}
            rows.append(row)
    return pd.DataFrame(rows)


def compare_pair(first, second, mode="fixed"):
    if first.get("currency", "RUB") != second.get("currency", "RUB"):
        raise ValueError("Report currencies differ")
    if first["comparison_key"] != second["comparison_key"]:
        raise ValueError("Workload, code, precision or library versions differ; rerun matching settings")
    if first["environment"]["device_type"] != "cuda" or second["environment"]["device_type"] != "cuda":
        raise ValueError("CPU smoke checks are not GPU economic evidence")
    def chosen(report):
        valid = [m for m in report["modes"] if m["status"] == "ok" and (mode == "best" or m["kind"] == "fixed")]
        if not valid:
            raise ValueError(f"No completed {mode} mode for {report['label']}")
        return max(valid, key=lambda m: m["queries_per_second"])
    a, b = chosen(first), chosen(second)
    speedup = b["queries_per_second"] / a["queries_per_second"]
    price_ratio = second.get("price_per_hour", second.get("rub_per_hour")) / first.get("price_per_hour", first.get("rub_per_hour"))
    # Trial extrema are observed variation, not a confidence interval.
    low = b["repeat_speed_min"] / a["repeat_speed_max"]
    high = b["repeat_speed_max"] / a["repeat_speed_min"]
    verdict = (second["label"] if low > price_ratio else first["label"] if high < price_ratio
               else "uncertain: repeat measurements")
    return {"mode": mode, "first": first["label"], "second": second["label"],
        "second_speedup": speedup, "break_even_speedup": price_ratio,
        "second_cost_ratio": price_ratio / speedup, "observed_speedup_range": [low, high],
        "cheaper_training_work": verdict, "selected_micro_batches": [a["micro_batch"], b["micro_batch"]],
        "scope": "steady-state optimizer work; no quality equivalence or full-job billing guarantee"}
