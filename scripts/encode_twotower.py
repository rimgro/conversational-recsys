"""Векторы обученного двухбашенного кандгена в файлы: треки и запросы частей датасета (GPU; дальше — без GPU).

    python scripts/encode_twotower.py --run-dir artifacts/twotower/v2              # ranker_train, valid, test, submit
    python scripts/encode_twotower.py --run-dir artifacts/twotower/v2 --parts valid      # только эта часть
    python scripts/encode_twotower.py --run-dir artifacts/twotower/v2 --parts test --history-len 16

В --run-dir (или --out):
  item_vectors.npy, item_ids.npy           (n_tracks, dim) float32 и m4a_id по строкам
  query_vectors_<часть>.npz                query_id, vectors (float16); с --history-len — ..._h<L>.npz
  recall_<часть>.json                      recall@10/100/200 по всему каталогу, всего и по типам запросов
  encode_parts/                            куски по 2000 запросов: оборвалось — та же команда продолжит с места;
                                           не хватило памяти GPU — пачка уменьшается сама

Части — recsys/data/parts.py: модель учится на index, векторы нужны для ranker_train, valid, test (и submit).
Кандген twotower (recsys/candgen/dense.py) читает эти файлы: метрики в общем пайплайне —
python -m recsys candgen --sources +twotower --set candgen.sources.twotower.dir=<папка>.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # recsys из корня репозитория; запускать из корня: python scripts/<скрипт>.py

import numpy as np

from train_twotower import build_cfg
from recsys.twotower.build import load_examples, load_trained
from recsys.twotower.train import KS, device_precision, encode_examples, item_vectors, pick_device, recall_at


CHUNK = 2000   # запросов в куске: каждый кусок сразу на диск, после обрыва считаются только недостающие


def model_stamp(path: Path) -> str:
    """Чем посчитаны векторы: имя чекпоинта, размер и время записи. Модель переобучена в той же папке — штамп
    другой, и старые векторы не берутся."""
    st = path.stat()
    return f"{path.name}:{st.st_size}:{st.st_mtime_ns}"


def same_vectors(path: Path, query_id: np.ndarray, stamp: str) -> bool:
    with np.load(path) as z:
        return "model" in z and str(z["model"]) == stamp and z["query_id"].tolist() == query_id.astype(str).tolist()


def encode_resumable(model, ex, builder, device, precision, batch_size, history_len, parts_dir: Path, name: str,
                     stamp: str = ""):
    """Векторы запросов кусками по CHUNK: готовые куски (parts_dir/<name>_<i>.npz с теми же query_id и той же
    моделью — stamp) берутся с диска. Не хватило памяти GPU — пачка вдвое меньше, тот же кусок заново."""
    import torch
    parts_dir.mkdir(parents=True, exist_ok=True)
    out, n_chunks, t0 = [], (len(ex) + CHUNK - 1) // CHUNK, time.time()
    for i, start in enumerate(range(0, len(ex), CHUNK)):
        part = ex.subset(np.arange(start, min(start + CHUNK, len(ex))))
        path = parts_dir / f"{name}_{i:04d}.npz"
        if path.exists() and same_vectors(path, part.query_id, stamp):
            with np.load(path) as z:
                out.append(z["vectors"])
            continue
        while True:
            try:
                u = encode_examples(model, part, builder, device, precision, batch_size, history_len=history_len)
                break
            except torch.OutOfMemoryError:
                if batch_size <= 1:
                    raise
                batch_size //= 2
                torch.cuda.empty_cache()
                print(f"[encode] не хватило памяти GPU: пачка {batch_size}, кусок {i + 1} заново")
        np.savez(path.with_suffix(".tmp.npz"), query_id=part.query_id.astype(str), vectors=u.astype(np.float16),
                 model=stamp)
        path.with_suffix(".tmp.npz").replace(path)
        out.append(u.astype(np.float16))
        done = min(start + CHUNK, len(ex))
        rate = (time.time() - t0) / max(i + 1, 1)
        print(f"[encode] {name}: {done}/{len(ex)} запросов (кусок {i + 1}/{n_chunks} сохранён), "
              f"осталось ~{rate * (n_chunks - i - 1) / 60:.0f} мин")
    return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float16), batch_size


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", required=True, help="папка прогона scripts/train_twotower.py")
    p.add_argument("--checkpoint", default="model.pt", help="model.pt (лучшее на валидации) или last.pt")
    p.add_argument("--parts", help="части запросов через запятую (по умолчанию twotower.encode.parts)")
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
    stamp = model_stamp(run_dir / a.checkpoint)
    cfg = {**cfg, "twotower": tcfg}        # история и запросы — как при обучении (arch, history_len), а не по --config
    precision = device_precision(device, tcfg["model"].get("precision", "bf16"))
    bs = tcfg.get("encode", {}).get("batch_size", 256)

    v = item_vectors(model)
    np.save(out / "item_vectors.npy", v)
    np.save(out / "item_ids.npy", codes.sid.track_ids.astype(str))
    print(f"[encode] треки: {v.shape} -> {out / 'item_vectors.npy'}")

    suffix = f"_h{a.history_len}" if a.history_len is not None else ""
    for split in (a.parts.split(",") if a.parts else tcfg.get("encode", {}).get("parts", ["ranker_train", "valid", "test"])):
        target_path = out / f"query_vectors_{split}{suffix}.npz"
        print(f"[encode] {split}: читаю запросы...")
        ex = load_examples(cfg, split, builder.spec, codes.sid, n_users=a.n_users)
        if target_path.exists() and same_vectors(target_path, ex.query_id, stamp):
            print(f"[encode] {split}: уже посчитано этой моделью ({target_path}), пропускаю")
            continue
        u, bs = encode_resumable(model, ex, builder, device, precision, bs, a.history_len,
                                 out / "encode_parts", f"{split}{suffix}", stamp)
        np.savez(target_path, query_id=ex.query_id.astype(str), vectors=u.astype(np.float16), model=stamp)
        print(f"[encode] {split}: {u.shape} -> {target_path}")
        has = ex.target >= 0
        if has.any():
            r = recall_at(u[has].astype(np.float32), v, ex.target[has])
            types = ex.query_type[has]
            res = {**{k: x for k, x in r.items() if k != "ranks"}, "n": int(has.sum()),
                   "by_type": {t: {f"recall@{k}": float((r["ranks"][types == t] <= k).mean()) for k in KS}
                               for t in sorted(set(types))}}
            (out / f"recall_{split}{suffix}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
            print(f"[encode] {split}: " + ", ".join(f"recall@{k} {res[f'recall@{k}']:.4f}" for k in KS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
