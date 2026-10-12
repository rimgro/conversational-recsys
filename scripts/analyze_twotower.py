"""Разбор векторов двухбашенного кандгена (без GPU): различимы ли треки, какие треки модель выдаёт, где промахи.

    python scripts/analyze_twotower.py --run-dir artifacts/twotower/main
    python scripts/analyze_twotower.py --run-dir artifacts/twotower/main --n-queries 50000

Берёт item_vectors.npy, item_ids.npy, query_vectors_<сплит>.npz (scripts/encode_twotower.py) и сплит из data.crs.dir.
Печатает:
  1. треки: сколько векторов различимы (точно и с косинусом > 0.99 к другому треку), средний косинус пар;
  2. выдача top-k на выборке запросов: сколько разных треков в ней бывает, какую долю мест занимает 1% самых
     частых треков, медианная популярность выдачи против целей;
  3. recall@k по всему каталогу: цель в истории пользователя или новая; в последних 64 треках истории или раньше.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # recsys из корня репозитория; запускать из корня: python scripts/<скрипт>.py

import numpy as np

from evaluate import config_path
from recsys.config import load_config
from recsys.data.crs import read_split


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--split", default="test_public")
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--n-queries", type=int, default=20000, help="выборка запросов для пунктов 2–3")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    run = Path(a.run_dir)

    v = np.load(run / "item_vectors.npy").astype(np.float32)
    ids = np.load(run / "item_ids.npy").astype(str)
    pos = {t: i for i, t in enumerate(ids)}
    n = len(v)

    print("1. векторы треков")
    uniq = len(np.unique(np.round(v, 5), axis=0))
    rng = np.random.default_rng(a.seed)
    sample = rng.choice(n, min(5000, n), replace=False)
    with np.errstate(all="ignore"):
        sims = v[sample] @ v.T
    sims[np.arange(len(sample)), sample] = -1.0
    nearest = sims.max(1)
    print(f"  различных векторов: {uniq} из {n}")
    print(f"  у трека есть сосед с косинусом > 0.99: {(nearest > 0.99).mean():.1%}, > 0.95: {(nearest > 0.95).mean():.1%} "
          f"(выборка {len(sample)} треков); медиана косинуса с ближайшим: {np.median(nearest):.3f}")
    print(f"  средний косинус случайной пары: {sims[sims > -1].mean():.3f} (около 0 — векторы разбросаны, около 1 — слиплись)")

    data_dir = load_config(config_path("default.yaml"))["data"]["crs"]["dir"]
    tables = read_split(data_dir, a.split)
    z = np.load(run / f"query_vectors_{a.split}.npz")
    qrow = {q: i for i, q in enumerate(z["query_id"].astype(str))}
    q = tables["queries"].merge(tables["qrels"][["query_id", "target_m4a_id", "is_new"]], on="query_id")
    q = q[q["query_id"].isin(qrow) & q["target_m4a_id"].isin(pos)]
    q = q.sample(min(a.n_queries, len(q)), random_state=a.seed)
    hist = {u: list(h) for u, h in zip(tables["users"]["user_id"], tables["users"]["history"])}

    u = z["vectors"][[qrow[x] for x in q["query_id"]]].astype(np.float32)
    target = np.array([pos[t] for t in q["target_m4a_id"]])
    tops, ranks = [], np.empty(len(q), dtype=np.int64)
    for s in range(0, len(u), 1024):
        with np.errstate(all="ignore"):
            sc = u[s:s + 1024] @ v.T
        t = target[s:s + 1024]
        ranks[s:s + 1024] = (sc > sc[np.arange(len(t)), t][:, None]).sum(1) + 1
        tops.append(np.argpartition(-sc, a.k - 1, axis=1)[:, :a.k])
    tops = np.concatenate(tops)

    print(f"\n2. выдача top-{a.k} ({len(q)} запросов {a.split})")
    cnt = Counter(tops.ravel().tolist())
    freq = np.array(sorted(cnt.values(), reverse=True))
    top1 = freq[:max(1, n // 100)].sum() / freq.sum()
    print(f"  разных треков в выдачах: {len(cnt)} из {n} ({len(cnt) / n:.1%})")
    print(f"  1% самых частых треков ({max(1, n // 100)}) занимают {top1:.1%} всех мест в выдачах")
    item_pop = Counter(t for h in hist.values() for t in set(h))
    pop = np.array([item_pop.get(t, 0) for t in ids])
    print(f"  медианная популярность (сколько пользователей сплита слушали трек): выдача {np.median(pop[tops.ravel()]):.0f}, "
          f"цели {np.median(pop[target]):.0f}, каталог {np.median(pop):.0f}")

    print(f"\n3. recall@{a.k} по всему каталогу, где цель")
    in_hist, in_recent = [], []
    for user, t in zip(q["user_id"], q["target_m4a_id"]):
        h = hist.get(user, [])
        in_hist.append(t in set(h))
        in_recent.append(t in set(list(dict.fromkeys(reversed(h)))[:64]))
    in_hist, in_recent = np.array(in_hist), np.array(in_recent)
    hit = ranks <= a.k
    for name, m in (("все", np.ones(len(q), bool)), ("цель в истории пользователя", in_hist),
                    ("  — в последних 64 треках (видна модели)", in_recent),
                    ("  — раньше, модель её не видит", in_hist & ~in_recent), ("цель новая (не в истории)", ~in_hist)):
        if m.any():
            print(f"  {name:<42} доля {m.mean():6.1%}   recall@{a.k} {hit[m].mean():.3f}   "
                  f"recall@10 {(ranks[m] <= 10).mean():.3f}   медианное место {np.median(ranks[m]):.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
