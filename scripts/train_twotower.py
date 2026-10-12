"""Обучение двухбашенного кандгена (docs/twotower.md) на train: GPU, torch + transformers + peft.

    python scripts/train_twotower.py --run-dir artifacts/twotower/main
    python scripts/train_twotower.py --run-dir artifacts/twotower/inbatch --set twotower.loss.negatives=in_batch
    python scripts/train_twotower.py --run-dir artifacts/twotower/random --set twotower.codes.source=random \\
        --set twotower.codes.init=random
    python scripts/train_twotower.py --run-dir /tmp/tt --n-users 200 --set twotower.train.stages=[...]   # проверка

Конфиг: configs/default.yaml + configs/twotower.yaml (+ --config, --set). Результат в --run-dir: model.pt (лучшее
на валидации), last.pt, log.jsonl, metrics.json (recall@10/100/200 на всей валидации, по типам), config.yaml.
Дальше — scripts/encode_twotower.py --run-dir <та же папка>.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # recsys из корня репозитория; запускать из корня: python scripts/<скрипт>.py

import numpy as np
import yaml

from evaluate import config_path, git_commit, set_key
from recsys.config import load_config
from recsys.twotower.build import load_codes, load_examples, make_builder, new_model
from recsys.twotower.data import TokenizerSpec
from recsys.twotower.model import load_base
from recsys.twotower.train import device_attention, device_precision, evaluate, pick_device, split_train_val, target_log_q, train


def build_cfg(a: argparse.Namespace) -> dict:
    over: dict = {}
    for item in a.set:
        key, _, value = item.partition("=")
        set_key(over, key, yaml.safe_load(value))
    return load_config([config_path("default.yaml"), config_path("twotower.yaml")] + a.config, over)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", required=True, help="куда писать модель и логи (например, artifacts/twotower/main)")
    p.add_argument("--config", action="append", default=[], help="ещё YAML поверх configs/twotower.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="параметр конфига")
    p.add_argument("--n-users", type=int, help="взять столько пользователей train (быстрая проверка)")
    p.add_argument("--device", help="cuda / cpu (по умолчанию cuda, если есть)")
    a = p.parse_args(argv)

    cfg = build_cfg(a)
    tcfg = cfg["twotower"]
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    device = pick_device(a.device)

    codes = load_codes(cfg)
    print(f"[twotower] коды: {tcfg['codes'].get('source', 'rqvae')} {codes.codes.shape}, таблицы {codes.vocab_sizes}")
    precision = device_precision(device, tcfg["model"].get("precision", "bf16"))
    base, tok = load_base(tcfg["model"]["name"], precision,
                          device_attention(device, tcfg["model"].get("attn_implementation", "sdpa")))
    spec = TokenizerSpec.from_hf(tok)
    ex = load_examples(cfg, "train", spec, codes.sid, n_users=a.n_users)
    ex = ex.subset(np.flatnonzero(ex.target >= 0))
    train_ex, val_ex = split_train_val(ex, tcfg["train"].get("val_users", 0.05))
    print(f"[twotower] train: {len(train_ex)} запросов, валидация: {len(val_ex)} "
          f"({len(set(val_ex.user_id))} пользователей)")
    model, _ = new_model(cfg, codes, base=base, log_q=target_log_q(train_ex, len(codes.codes)))
    builder = make_builder(cfg, spec, codes.codes.shape[1])

    best = train(model, train_ex, val_ex, builder, tcfg, str(run_dir), device)
    full = evaluate(model, val_ex, builder, device, precision, tcfg.get("encode", {}).get("batch_size", 256),
                    by_type=True)   # модель после последнего шага, вся валидация
    result = {"best_val": best, "last_val": full, "n_train": len(train_ex), "n_val": len(val_ex),
              "git_commit": git_commit()}
    (run_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[twotower] лучшее на валидации: {best}\nрезультат: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
