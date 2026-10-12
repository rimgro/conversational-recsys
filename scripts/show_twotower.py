"""Примеры выдачи двухбашенного кандгена: запрос -> цель и top-k треков модели (по готовым векторам, без GPU).

    python scripts/show_twotower.py --run-dir artifacts/twotower/main                       # 5 случайных запросов
    python scripts/show_twotower.py --run-dir artifacts/twotower/main --type mood -n 3 --k 10
    python scripts/show_twotower.py --run-dir artifacts/twotower/main --query-id 0a1b2c3d4e5f6789

Берёт item_vectors.npy, item_ids.npy и query_vectors_<сплит>.npz из --run-dir (scripts/encode_twotower.py),
тексты запросов и цели — из data.dir. Для каждого запроса: текст, тип, цель (артист — название) и её место
по всему каталогу, top-k модели; цель в выдаче отмечена «<- цель».
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # recsys из корня репозитория; запускать из корня: python scripts/<скрипт>.py

import numpy as np

from recsys.config import config_path, load_config
from recsys.data.dataset import Dataset
from recsys.data.parts import part_split


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", required=True, help="папка с векторами (scripts/encode_twotower.py)")
    p.add_argument("--part", default="test", help="часть запросов (recsys/data/parts.py), как у encode_twotower.py")
    p.add_argument("-n", type=int, default=5, help="сколько запросов показать")
    p.add_argument("--k", type=int, default=10, help="сколько треков выдачи показать")
    p.add_argument("--type", help="только этот тип запроса (mood, genre, similar_to, ...)")
    p.add_argument("--query-id", action="append", default=[], help="конкретные query_id (можно несколько)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    run = Path(a.run_dir)
    v = np.load(run / "item_vectors.npy").astype(np.float32)
    ids = np.load(run / "item_ids.npy").astype(str)
    z = np.load(run / f"query_vectors_{a.part}.npz")
    qvec = dict(zip(z["query_id"].astype(str), range(len(z["query_id"]))))
    u_all = z["vectors"].astype(np.float32)

    cfg = load_config(config_path("default.yaml"))
    ds = Dataset(cfg, verbose=False)
    meta = ds.tracks_meta(["m4a_id", "m4a_artist", "m4a_song"])
    name = dict(zip(meta["m4a_id"], meta["m4a_artist"] + " — " + meta["m4a_song"]))
    queries = ds.part(a.part)[["query_id", "query", "query_type"]]
    qrels = ds.table(part_split(a.part, cfg["data"]["parts"]), "qrels")[["query_id", "target_m4a_id"]]
    q = queries.merge(qrels, on="query_id")
    q = q[q["query_id"].isin(qvec)]
    if a.query_id:
        q = q[q["query_id"].isin(a.query_id)]
    elif a.type:
        q = q[q["query_type"] == a.type]
    if not len(q):
        print("нет таких запросов среди закодированных", file=sys.stderr)
        return 1
    if not a.query_id:
        q = q.sample(min(a.n, len(q)), random_state=a.seed)

    pos = {t: i for i, t in enumerate(ids)}
    for row in q.itertuples(index=False):
        u = u_all[qvec[row.query_id]]
        with np.errstate(all="ignore"):
            scores = v @ u
        top = np.argsort(-scores)[:a.k]
        t = pos.get(str(row.target_m4a_id))
        rank = int((scores > scores[t]).sum()) + 1 if t is not None else None
        print(f"\n[{row.query_type}] {row.query}   ({row.query_id})")
        print(f"  цель: {name.get(row.target_m4a_id, row.target_m4a_id)}  — место {rank} из {len(ids)}")
        for i, j in enumerate(top, start=1):
            mark = "  <- цель" if j == t else ""
            print(f"  {i:>3}. {scores[j]:.3f}  {name.get(ids[j], ids[j])}{mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
