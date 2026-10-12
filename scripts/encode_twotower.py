"""Векторы обученного двухбашенного кандгена в файлы: треки и запросы сплитов (GPU; дальше — без GPU).

    python scripts/encode_twotower.py --run-dir artifacts/twotower/main                      # сплиты из конфига
    python scripts/encode_twotower.py --run-dir artifacts/twotower/main --splits test_public --history-len 16

В --run-dir (или --out):
  item_vectors.npy, item_ids.npy           (n_tracks, dim) float32 и m4a_id по строкам
  query_vectors_<сплит>.npz                query_id, vectors (float16); с --history-len — ..._h<L>.npz
  recall_<сплит>.json                      recall@10/100/200 по всему каталогу, всего и по типам запросов

Источник retrieval.twotower (recsys/retrieval/twotower.py) читает эти файлы: метрики в общем пайплайне —
python scripts/evaluate_candgen.py --set retrieval.twotower.enabled=true --set retrieval.twotower.dir=<папка>.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # recsys из корня репозитория; запускать из корня: python scripts/<скрипт>.py

import numpy as np

from train_twotower import build_cfg
from recsys.twotower.build import load_examples, load_trained
from recsys.twotower.train import KS, device_precision, encode_examples, item_vectors, pick_device, recall_at


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", required=True, help="папка прогона scripts/train_twotower.py")
    p.add_argument("--checkpoint", default="model.pt", help="model.pt (лучшее на валидации) или last.pt")
    p.add_argument("--splits", help="через запятую (по умолчанию twotower.encode.splits)")
    p.add_argument("--n-users", type=int, help="столько пользователей сплита (быстрая проверка)")
    p.add_argument("--history-len", type=int, help="обрезать историю до L треков (ablation, без переобучения)")
    p.add_argument("--out", help="куда писать (по умолчанию --run-dir)")
    p.add_argument("--config", action="append", default=[])
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--device")
    a = p.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)   # прогресс сразу, а не пачками

    cfg = build_cfg(a)
    run_dir, out = Path(a.run_dir), Path(a.out or a.run_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(a.device)
    print(f"[encode] загружаю модель из {run_dir / a.checkpoint}...")
    model, builder, codes, tcfg = load_trained(cfg, str(run_dir / a.checkpoint), device)
    precision = device_precision(device, tcfg["model"].get("precision", "bf16"))
    bs = tcfg.get("encode", {}).get("batch_size", 256)

    v = item_vectors(model)
    np.save(out / "item_vectors.npy", v)
    np.save(out / "item_ids.npy", codes.sid.track_ids.astype(str))
    print(f"[encode] треки: {v.shape} -> {out / 'item_vectors.npy'}")

    suffix = f"_h{a.history_len}" if a.history_len is not None else ""
    for split in (a.splits.split(",") if a.splits else tcfg.get("encode", {}).get("splits", ["test_public"])):
        print(f"[encode] {split}: читаю запросы...")
        ex = load_examples(cfg, split, builder.spec, codes.sid, n_users=a.n_users)
        u = encode_examples(model, ex, builder, device, precision, bs, history_len=a.history_len, log=print)
        np.savez(out / f"query_vectors_{split}{suffix}.npz", query_id=ex.query_id.astype(str), vectors=u.astype(np.float16))
        print(f"[encode] {split}: {u.shape} -> {out / f'query_vectors_{split}{suffix}.npz'}")
        has = ex.target >= 0
        if has.any():
            r = recall_at(u[has], v, ex.target[has])
            types = ex.query_type[has]
            res = {**{k: x for k, x in r.items() if k != "ranks"}, "n": int(has.sum()),
                   "by_type": {t: {f"recall@{k}": float((r["ranks"][types == t] <= k).mean()) for k in KS}
                               for t in sorted(set(types))}}
            (out / f"recall_{split}{suffix}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
            print(f"[encode] {split}: " + ", ".join(f"recall@{k} {res[f'recall@{k}']:.4f}" for k in KS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
