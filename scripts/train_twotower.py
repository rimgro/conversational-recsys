"""Обучение двухбашенного кандгена (docs/twotower.md) на train: GPU, torch + transformers + peft.

    python scripts/train_twotower.py --run-dir artifacts/twotower/main
    python scripts/train_twotower.py --run-dir artifacts/twotower/inbatch --set twotower.loss.negatives=in_batch
    python scripts/train_twotower.py --run-dir artifacts/twotower/random --set twotower.codes.source=random \\
        --set twotower.codes.init=random
    python scripts/train_twotower.py --run-dir /tmp/tt --n-users 200 --set twotower.train.stages=[...]   # проверка

Конфиг: configs/default.yaml + configs/twotower.yaml (+ --config, --set). Результат в --run-dir: model.pt (лучшее
на валидации), last.pt, log.jsonl, metrics.json (recall@10/100/200 на всей валидации, по типам), config.yaml.
Дальше — scripts/encode_twotower.py --run-dir <та же папка>.

Оборвалась сессия (Colab) — та же команда с --resume: продолжит с checkpoint.pt (пишется на каждой валидации)
с тем же конфигом twotower, что был при старте; если обучение уже закончено — только посчитает metrics.json.
Нет checkpoint.pt (прогон старой версии) — продолжит с весов model.pt и его шага, оптимизатор заново.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # recsys из корня репозитория; запускать из корня: python scripts/<скрипт>.py

import numpy as np
import yaml

from recsys.config import config_path, git_commit, load_config, set_key
from recsys.twotower.build import load_codes, load_examples, make_builder, new_model
from recsys.twotower.data import TokenizerSpec
from recsys.twotower.model import load_base
from recsys.twotower.train import device_attention, device_precision, evaluate, pick_device, split_train_val, target_log_q, train


def torch_device_name(device) -> str:
    import torch
    return torch.cuda.get_device_name(device)


def build_cfg(a: argparse.Namespace) -> dict:
    over: dict = {}
    for item in a.set:
        key, _, value = item.partition("=")
        set_key(over, key, yaml.safe_load(value))
    return load_config([config_path("default.yaml"), config_path("twotower.yaml")] + a.config, over)


def flops_report(tcfg: dict, base, ex, builder) -> dict:
    """arch lazy: FLOPs на запрос при средней длине истории и текста этого train (recsys/twotower/lazy.py)."""
    from recsys.twotower.lazy import lazy_flops
    qm = base.get_base_model() if hasattr(base, "get_base_model") else base
    L = float(np.mean([len(ex.histories[u]) for u in ex.user])) if len(ex) else 0.0
    T = float(np.mean([len(q) for q in ex.query_tokens])) + len(builder.prefix) + 1 if len(ex) else 0.0
    lz = tcfg.get("lazy", {})
    f = lazy_flops(qm.config, L, T, lz.get("mode", "encdec"), lz.get("encoder_layers", 0), lz.get("cross_every", 1))
    print(f"[twotower] FLOPs на запрос (GFLOP; история {L:.0f} треков, текст {T:.0f} токенов): "
          + ", ".join(f"{k} {v / 1e9:.1f}" for k, v in f.items() if not k.startswith("cache")))
    return {"mean_history": L, "mean_text_tokens": T, **f}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", required=True, help="куда писать модель и логи (например, artifacts/twotower/main)")
    p.add_argument("--config", action="append", default=[], help="ещё YAML поверх configs/twotower.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="параметр конфига")
    p.add_argument("--n-users", type=int, help="взять столько пользователей train (быстрая проверка)")
    p.add_argument("--device", help="cuda / mps / cpu (по умолчанию cuda, если есть)")
    p.add_argument("--resume", action="store_true", help="продолжить с --run-dir/checkpoint.pt (если он есть)")
    a = p.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)   # лог сразу, а не пачками (вывод в ноутбук — не терминал)

    cfg = build_cfg(a)
    run_dir = Path(a.run_dir)
    saved = [p for p in (run_dir / "checkpoint.pt", run_dir / "model.pt") if p.exists()]
    if not a.resume and (saved or (run_dir / "last.pt").exists()):
        raise SystemExit(f"[twotower] в {run_dir} уже есть обученная модель: новый прогон — новая папка --run-dir, "
                         f"продолжить этот — --resume (иначе model.pt, checkpoint.pt и log.jsonl перезапишутся)")
    if a.resume and saved:           # тот же конфиг, что при старте: иначе веса и этапы не совпадут
        import torch
        old = torch.load(saved[0], map_location="cpu", weights_only=False)["twotower"]
        model_keys = [k for k in set(old) | set(cfg["twotower"]) if k not in ("train", "encode")]
        changed = [k for k in sorted(model_keys) if old.get(k) != cfg["twotower"].get(k)]
        if (a.config or a.set) and changed:
            raise SystemExit(f"[twotower] {saved[0]} обучен с другой моделью ({', '.join(changed)} не совпадают с "
                             f"--config / --set): это другой прогон — нужна своя папка --run-dir")
        cfg["twotower"] = old
        print(f"[twotower] --resume: конфиг twotower из {saved[0]} (--set для twotower игнорируются)")
    tcfg = cfg["twotower"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    device = pick_device(a.device)
    print(f"[twotower] устройство: {device}" + (f" ({torch_device_name(device)})" if device.type == "cuda" else ""))

    codes = load_codes(cfg)
    print(f"[twotower] коды: {tcfg['codes'].get('source', 'rqvae')} {codes.codes.shape}, таблицы {codes.vocab_sizes}")
    precision = device_precision(device, tcfg["model"].get("precision", "bf16"))
    print(f"[twotower] загружаю {tcfg['model']['name']} ({precision})...")
    base, tok = load_base(tcfg["model"]["name"], precision,
                          device_attention(device, tcfg["model"].get("attn_implementation", "sdpa")))
    spec = TokenizerSpec.from_hf(tok)
    t = time.time()
    part = tcfg["train"].get("part", "train")
    print(f"[twotower] читаю {part} и токенизирую запросы (на всём train — пара минут)...")
    ex = load_examples(cfg, part, spec, codes.sid, n_users=a.n_users)
    print(f"[twotower] прочитано за {time.time() - t:.0f} c")
    ex = ex.subset(np.flatnonzero(ex.target >= 0))
    train_ex, val_ex = split_train_val(ex, tcfg["train"].get("val_users", 0.05))
    print(f"[twotower] train: {len(train_ex)} запросов, валидация: {len(val_ex)} "
          f"({len(set(val_ex.user_id))} пользователей)")
    print("[twotower] собираю модель (группы «одна песня» для маски дублей)...")
    model, _ = new_model(cfg, codes, base=base, log_q=target_log_q(train_ex, len(codes.codes)))
    builder = make_builder(cfg, spec, codes.codes.shape[1])
    if tcfg.get("arch") == "lazy":
        flops_report(tcfg, base, train_ex, builder)

    best = train(model, train_ex, val_ex, builder, tcfg, str(run_dir), device, resume=a.resume)
    print(f"[twotower] итоговая валидация на всех {len(val_ex)} запросах валидации...")
    full = evaluate(model, val_ex, builder, device, precision, tcfg.get("encode", {}).get("batch_size", 256),
                    by_type=True)   # модель после последнего шага, вся валидация
    result = {"best_val": best, "last_val": full, "n_train": len(train_ex), "n_val": len(val_ex),
              "git_commit": git_commit()}
    if tcfg.get("arch") == "lazy":
        result["flops"] = flops_report(tcfg, base, train_ex, builder)
    (run_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[twotower] лучшее на валидации: {best}\nрезультат: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
