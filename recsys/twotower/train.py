"""Обучение двухбашенного кандгена по этапам (twotower.train.stages), валидация, сохранение и загрузка.

Этап = доля эпохи train + lr; lora: true добавляет LoRA на attention Qwen (до этого Qwen заморожен целиком:
сначала учатся токены кодов, проекция и башня треков, потом attention — иначе случайные новые параметры
испортят предобученную модель шумными градиентами).

Папка прогона (run_dir):
  model.pt          лучшее по recall@100 на валидации: обучаемые веса (без весов Qwen) + конфиг
  last.pt           после последнего шага
  checkpoint.pt     для продолжения после обрыва (resume): веса, оптимизатор, этап и шаг; пишется на каждой
                    валидации (train.eval_every) и в конце этапа
  log.jsonl         loss, τ, lr по шагам и метрики валидации
  config.yaml       полный конфиг
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from recsys.twotower.data import Examples, QueryDataset, SequenceBuilder, TokenizerSpec, is_val_user
from recsys.twotower.model import DTYPES, TwoTower, add_lora, trainable_state

KS = (10, 100, 200)


def pick_device(device: Optional[str] = None) -> torch.device:
    """--device, иначе cuda (GPU DataSphere), иначе mps (GPU Mac), иначе cpu. Код один и тот же: меняется
    только устройство и точность (bf16 / fp16 — только на cuda, на mps и cpu — fp32)."""
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_precision(device: torch.device, precision: str) -> str:
    """bf16 / fp16 — только на cuda (на mps и cpu — fp32); bf16 на GPU старше Ampere (T4 в Colab, V100) нет
    в железе — там fp16."""
    if device.type != "cuda":
        return "fp32"
    if precision == "bf16" and torch.cuda.get_device_capability(device)[0] < 8:
        return "fp16"
    return precision


def device_attention(device: torch.device, attn: Optional[str]) -> Optional[str]:
    """sdpa на mps с паддингом слева даёт NaN (полностью замаскированные строки паддинга; на cuda transformers
    это обходит сам) — на mps eager."""
    return "eager" if device.type == "mps" else attn


def autocast(device: torch.device, precision: str):
    """bf16 / fp16 на GPU; на CPU и в fp32 — без autocast."""
    if device.type == "cuda" and precision in ("bf16", "fp16"):
        return torch.autocast("cuda", dtype=DTYPES[precision])
    return torch.autocast("cpu", enabled=False)


def split_train_val(ex: Examples, val_share: float):
    """Валидация — пользователи целиком (по хешу user_id): у одного пользователя десятки запросов с той же
    историей, деление по запросам завысило бы метрики."""
    val = np.array([is_val_user(u, val_share) for u in ex.user_id], dtype=bool)
    return ex.subset(np.flatnonzero(~val)), ex.subset(np.flatnonzero(val))


def target_log_q(ex: Examples, n_items: int) -> np.ndarray:
    """log частоты трека как цели в train (+1 сглаживание): поправка logQ для in-batch."""
    cnt = np.bincount(ex.target[ex.target >= 0], minlength=n_items).astype(np.float64) + 1.0
    return np.log(cnt / cnt.sum()).astype(np.float32)


def move(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def encode_examples(model: TwoTower, ex: Examples, builder: SequenceBuilder, device: torch.device,
                    precision: str = "fp32", batch_size: int = 256, history_len: Optional[int] = None,
                    log: Optional[Callable[[str], None]] = None, nulls: Optional[List[np.ndarray]] = None) -> np.ndarray:
    """Векторы u запросов (n, dim), float32. history_len — обрезать историю (ablation по длине без переобучения).
    log — печатать прогресс (раз в ~10%); nulls — сюда вес пустого слота каждого запроса (arch v2 с shortcut)."""
    model.eval()
    loader = torch.utils.data.DataLoader(QueryDataset(ex, history_len=history_len), batch_size=batch_size,
                                         shuffle=False, collate_fn=builder.collate)
    out = []
    t0, every = time.time(), max(1, len(loader) // 10)
    parts = model.track_parts() if hasattr(model, "track_parts") else None   # v2: r всех треков — один раз
    for i, batch in enumerate(loader, start=1):
        with autocast(device, precision):
            b = move(batch, device)
            out.append((model.encode_users(b, parts) if parts is not None else model.encode_users(b)).float().cpu())
        if nulls is not None and getattr(model.user, "last_null", None) is not None:
            nulls.append(model.user.last_null.float().cpu().numpy())
        if log and (i % every == 0 or i == len(loader)):
            done = min(i * batch_size, len(ex))
            log(f"[encode] {done}/{len(ex)} запросов, {time.time() - t0:.0f} c, осталось ~"
                f"{(time.time() - t0) / done * (len(ex) - done) / 60:.0f} мин")
    if out and torch.isnan(torch.cat(out)).any():
        raise RuntimeError("NaN в векторах запросов: на mps нужен attn_implementation eager (device_attention), "
                           "в fp16 — проверьте переполнение")
    return torch.cat(out).numpy() if out else np.zeros((0, model.user.proj.out_features), dtype=np.float32)


@torch.no_grad()
def item_vectors(model: TwoTower) -> np.ndarray:
    model.eval()
    return model.encode_items().float().cpu().numpy()


def recall_at(u: np.ndarray, v: np.ndarray, target: np.ndarray, ks=KS, chunk: int = 1024) -> Dict[str, Any]:
    """recall@k по всему каталогу: цель среди k треков с наибольшим u·v. -> средние и место цели по запросам."""
    ranks = np.empty(len(u), dtype=np.int64)
    for s in range(0, len(u), chunk):
        with np.errstate(all="ignore"):  # numpy 2.0 + Accelerate (macOS) даёт ложные warnings в matmul
            sc = u[s:s + chunk] @ v.T
        t = target[s:s + chunk]
        ranks[s:s + chunk] = (sc > sc[np.arange(len(t)), t][:, None]).sum(1) + 1
    return {**{f"recall@{k}": float((ranks <= k).mean()) for k in ks}, "ranks": ranks}


def evaluate(model: TwoTower, ex: Examples, builder: SequenceBuilder, device: torch.device, precision: str,
             batch_size: int = 256, by_type: bool = False) -> Dict[str, Any]:
    nulls: List[np.ndarray] = []
    u = encode_examples(model, ex, builder, device, precision, batch_size, nulls=nulls)
    r = recall_at(u, item_vectors(model), ex.target)
    out = {k: v for k, v in r.items() if k != "ranks"}
    null = np.concatenate(nulls) if nulls else None
    if null is not None:
        out["null"] = float(null.mean())         # средний вес «ничего из истории»
    if by_type:
        out["by_type"] = {t: {f"recall@{k}": float((r["ranks"][ex.query_type == t] <= k).mean()) for k in KS}
                          for t in sorted(set(ex.query_type))}
        if null is not None:
            for t in out["by_type"]:
                out["by_type"][t]["null"] = float(null[ex.query_type == t].mean())
    return out


def param_groups(model: TwoTower, stage: Dict[str, Any], weight_decay: float) -> List[Dict[str, Any]]:
    """Новые параметры — stage.lr, LoRA — stage.lr_lora; без weight decay у τ и смещений."""
    groups: Dict[str, List[torch.nn.Parameter]] = {"new": [], "new_nd": [], "lora": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in name:
            groups["lora"].append(p)
        elif p.ndim < 2 or name == "log_tau":
            groups["new_nd"].append(p)
        else:
            groups["new"].append(p)
    lr = stage["lr"]
    return [g for g in (
        {"params": groups["new"], "lr": lr, "weight_decay": weight_decay},
        {"params": groups["new_nd"], "lr": lr, "weight_decay": 0.0},
        {"params": groups["lora"], "lr": stage.get("lr_lora", lr), "weight_decay": weight_decay},
    ) if g["params"]]


def warmup_cosine(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(total - warmup, 1))))


def save(model: TwoTower, path: str, tcfg: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
    lora = hasattr(model.user.base, "peft_config")
    torch.save({"state": trainable_state(model), "twotower": tcfg, "lora": lora, **(extra or {})}, path + ".tmp")
    os.replace(path + ".tmp", path)   # на Drive не останется половины файла, если сессия оборвётся при записи


def load_weights(model: TwoTower, path: str, lora_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Веса из model.pt / last.pt / checkpoint.pt в уже собранную модель (build_model); LoRA добавляется,
    если она была."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("lora"):
        add_lora(model, lora_cfg or ck["twotower"].get("model", {}).get("lora", {}))
    missing, unexpected = model.load_state_dict(ck["state"], strict=False)
    missing = [k for k in missing if not k.startswith("user.base.") or "lora_" in k]
    if missing or unexpected:
        raise ValueError(f"{path}: не совпали веса: нет {missing[:5]}, лишние {unexpected[:5]}")
    return ck


REWARM_STEPS = 50


def resume_from_best(model: TwoTower, path: str, stages: List[Dict[str, Any]], plan: List[int],
                     lora_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Продолжение без checkpoint.pt (прогон старой версии или checkpoint.pt потерян): веса из model.pt и его
    этап / шаг (val.stage, val.step — общий номер шага); оптимизатора там нет."""
    mk = load_weights(model, path, lora_cfg)
    val = mk["val"]
    names = [st.get("name") for st in stages]
    if val.get("stage") not in names:
        raise ValueError(f"{path}: этап {val.get('stage')} не из конфига ({names})")
    si = names.index(val["stage"])
    step_in_stage = int(val["step"]) - sum(plan[:si])
    if step_in_stage >= plan[si]:                        # сохранено в конце этапа — продолжать со следующего
        si, step_in_stage = si + 1, 0
    return {"stage_idx": si, "step_in_stage": step_in_stage, "step_global": int(val["step"]),
            "best": {k: v for k, v in val.items()}, "done": si >= len(stages)}


def fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def train(model: TwoTower, train_ex: Examples, val_ex: Examples, builder: SequenceBuilder, tcfg: Dict[str, Any],
          run_dir: str, device: Optional[torch.device] = None, log: Callable[[str], None] = print,
          resume: bool = False) -> Dict[str, Any]:
    """Этапы twotower.train.stages по очереди; валидация каждые eval_every шагов и в конце этапа.
    resume — продолжить с run_dir/checkpoint.pt (этап, шаг, оптимизатор, лучшая валидация); порядок батчей
    после продолжения новый. -> лучшие метрики валидации."""
    device = device or pick_device()
    mcfg, lcfg, trc = tcfg.get("model", {}), tcfg.get("loss", {}), tcfg.get("train", {})
    precision = device_precision(device, mcfg.get("precision", "bf16"))
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "log.jsonl")
    ck_path = os.path.join(run_dir, "checkpoint.pt")
    torch.manual_seed(trc.get("seed", 42))

    bs, accum = trc.get("batch_size", 128), trc.get("grad_accum", 1)
    stages = trc.get("stages", [])
    plan = [max(1, int(st.get("epochs", 1.0) * len(train_ex) / (bs * accum))) for st in stages]
    if trc.get("max_steps"):                             # быстрая проверка: не больше max_steps на этап
        plan = [min(n, int(trc["max_steps"])) for n in plan]

    ck: Optional[Dict[str, Any]] = None
    best_path = os.path.join(run_dir, "model.pt")
    if resume and os.path.exists(ck_path):
        ck = load_weights(model, ck_path, mcfg.get("lora", {}))
        log(f"[train] продолжаю с {ck_path}: этап {ck['stage_idx'] + 1}, шаг {ck['step_in_stage']} этапа "
            f"(всего {ck['step_global']}), лучшее recall@100 {ck['best']['recall@100']:.4f}")
    elif resume and os.path.exists(best_path):
        ck = resume_from_best(model, best_path, stages, plan, mcfg.get("lora", {}))
        log(f"[train] {ck_path} нет — продолжаю с весов {best_path} (лучшая валидация): этап "
            f"{ck['stage_idx'] + 1}, шаг {ck['step_in_stage']} этапа (всего {ck['step_global']}); оптимизатор "
            f"заново, lr разгоняется за {REWARM_STEPS} шагов")
    elif resume:
        log(f"[train] {ck_path} и {best_path} нет — начинаю сначала")
    model.to(device)
    if mcfg.get("gradient_checkpointing") and device.type == "cuda":
        model.user.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")

    rng = np.random.default_rng(trc.get("seed", 42))
    n_val = min(trc.get("val_queries", 5000), len(val_ex))
    val_small = val_ex.subset(np.sort(rng.choice(len(val_ex), n_val, replace=False))) if n_val else val_ex
    workers = min(trc.get("num_workers", 0), os.cpu_count() or 1)   # в Colab 2 ядра
    loader = torch.utils.data.DataLoader(
        QueryDataset(train_ex, history_dropout=tcfg.get("input", {}).get("history_dropout", 0.0)),
        batch_size=bs, shuffle=True, drop_last=True, collate_fn=builder.collate,
        num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    log_every = trc.get("log_every", 50)

    best: Dict[str, Any] = ck["best"] if ck else {"recall@100": -1.0}
    step_global = ck["step_global"] if ck else 0
    t0 = time.time()

    def write(rec: Dict[str, Any]) -> None:
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def validate(stage_name: str) -> None:
        nonlocal best
        m = evaluate(model, val_small, builder, device, precision, tcfg.get("encode", {}).get("batch_size", 256))
        write({"stage": stage_name, "step": step_global, "val": m, "elapsed_s": round(time.time() - t0)})
        msg = ", ".join(f"{k} {v:.4f}" for k, v in m.items())
        if m["recall@100"] > best["recall@100"]:
            best = {**m, "stage": stage_name, "step": step_global}
            save(model, os.path.join(run_dir, "model.pt"), tcfg, {"val": best})
            msg += "  -> лучшее, сохранено в model.pt"
        log(f"[val] этап {stage_name}, шаг {step_global}: {msg}  ({fmt_time(time.time() - t0)} от старта)")
        model.train()

    def checkpoint(stage_idx: int, step_in_stage: int, opt, done: bool = False) -> None:
        save(model, ck_path, tcfg, {"stage_idx": stage_idx, "step_in_stage": step_in_stage,
                                    "step_global": step_global, "best": best, "done": done,
                                    "optimizer": opt.state_dict(), "scaler": scaler.state_dict()})

    log(f"[train] план: " + ", ".join(f"этап {st.get('name')} — {n} шагов" for st, n in zip(stages, plan))
        + f"; валидация и checkpoint.pt каждые {trc.get('eval_every') or '—'} шагов; лог каждые {log_every} шагов")
    first_stage = ck["stage_idx"] if ck else 0
    opt = None
    for si, stage in enumerate(stages):
        if si < first_stage:
            continue
        name, steps = stage.get("name", "?"), plan[si]
        if stage.get("lora"):
            add_lora(model, mcfg.get("lora", {}))
            model.to(device)
        opt = torch.optim.AdamW(param_groups(model, stage, trc.get("weight_decay", 0.01)))
        base_lrs = [g["lr"] for g in opt.param_groups]
        start, rewarm = 0, 0
        if ck and si == ck["stage_idx"]:
            start = ck["step_in_stage"]
            if start > 0 and "optimizer" in ck:
                opt.load_state_dict(ck["optimizer"])
                scaler.load_state_dict(ck["scaler"])
            elif start > 0:                              # с model.pt: моменты Adam с нуля — плавный разгон lr
                rewarm = REWARM_STEPS
        warmup = int(trc.get("warmup", 0.03) * steps)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"[train] этап {name}: шаги {start + 1}..{steps} по {bs}x{accum}, обучаемых параметров "
            f"{n_trainable / 1e6:.2f}M, точность {precision}")
        model.train()
        it = iter(loader)
        t_stage, done_here = time.time(), 0
        for step in range(start, steps):
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * warmup_cosine(step, steps, warmup) * (min(1.0, (step - start + 1) / rewarm) if rewarm else 1.0)
            for _ in range(accum):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)
                    batch = next(it)
                with autocast(device, precision):
                    out = model.loss(move(batch, device), negatives=lcfg.get("negatives", "full"),
                                     logq=lcfg.get("logq", False), mask_same_song=lcfg.get("mask_same_song", True))
                if not torch.isfinite(out["loss"]):
                    raise RuntimeError(f"loss {out['loss'].item()} на шаге {step_global + 1} (этап {name}): "
                                       f"в fp16 — переполнение (нужен GPU с bf16 или меньше lr)")
                scaler.scale(out["loss"] / accum).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           trc.get("max_grad_norm", 1.0))
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step_global += 1
            done_here += 1
            if (step + 1) % log_every == 0 or step == start or step + 1 == steps:
                per_step = (time.time() - t_stage) / done_here
                rec = {"stage": name, "step": step_global, "loss": out["loss"].item(), "tau": out["tau"].item(),
                       "lr": opt.param_groups[0]["lr"], "s_per_step": round(per_step, 3),
                       "elapsed_s": round(time.time() - t0)}
                if "null" in out:                        # arch v2: средний вес «ничего из истории»
                    rec["null"] = out["null"].item()
                write(rec)
                log(f"[train] {name} {step + 1}/{steps}  loss {rec['loss']:.4f}  τ {rec['tau']:.4f}  "
                    + (f"пусто {rec['null']:.2f}  " if "null" in rec else "") +
                    f"{per_step:.2f} с/шаг  прошло {fmt_time(time.time() - t0)}  до конца этапа ~"
                    f"{fmt_time(per_step * (steps - step - 1))}")
            if trc.get("eval_every") and step_global % trc["eval_every"] == 0 and step + 1 < steps:
                validate(name)
                checkpoint(si, step + 1, opt)
        validate(name)
        checkpoint(si + 1, 0, opt, done=si + 1 == len(stages))   # этап закончен: продолжать со следующего
    save(model, os.path.join(run_dir, "last.pt"), tcfg)
    return best
