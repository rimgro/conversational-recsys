"""Метрики кандгенов: каждый источник кандидатов отдельно и все вместе, без ранкера и описания (быстрее evaluate.py).

    python evaluate_candgen.py --n-users 1000                    # локальные источники
    python evaluate_candgen.py --n-users 1000 --candgen          # + BM25 и HNSW с сервера
    python evaluate_candgen.py --sources bm25,audio              # только эти источники
    python evaluate_candgen.py --set retrieval.audio.top_k=500   # любой параметр конфига

Списки: каждый источник и три этапа слияния — rrf (все источники, до фильтров), filtered (после жёстких
фильтров fusion.py), fused (первые fusion.top_n: то, что получает ранкер). Для каждого списка:
recall (цель где угодно в списке), recall@20/50/100/200, nDCG@20 (если бы выдачей был этот список как есть;
у fused это выдача stub-ранкера), only_this (цель нашёл только этот источник), median_rank, сколько кандидатов.
Цель для similar_to считается как в датасете: без трека-образца и его артиста.

Результат — папка outputs/<время>_<сплит>_candgen/ (или --run-dir):
  candgen.csv                строка = список: метрики выше
  candgen_by_query_type.csv  то же по типам запросов
  recall_by_query_type.csv   recall каждого списка: строка = тип запроса, колонка = список
  ndcg_by_query_type.csv     nDCG@20 каждого списка так же
  filter_losses.csv          доля запросов, где цель нашли, но выкинул фильтр (какой именно), по типам
  per_list.csv               строка = запрос × список: место цели, nDCG, сколько кандидатов; пишется по ходу
  metrics.json, config.yaml  метрики и как запускали
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from evaluate import add_common_args, build_config, check_candgen, git_commit
from recsys.data import load_data
from recsys.eval import STAGES, candidates_summary, evaluate_candidates, filter_losses
from recsys.pipeline import Pipeline


def keep_sources(cfg: dict, names: str) -> None:
    """--sources a,b: остальные источники выключаются."""
    wanted = {s.strip() for s in names.split(",") if s.strip()}
    unknown = wanted - set(cfg["retrieval"])
    if unknown:
        raise SystemExit(f"неизвестные источники: {sorted(unknown)}; есть: {sorted(cfg['retrieval'])}")
    for name, c in cfg["retrieval"].items():
        c["enabled"] = name in wanted


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(p)
    p.add_argument("--sources", help="через запятую: считать только эти источники (по умолчанию — включённые в конфиге)")
    a = p.parse_args(argv)

    started = datetime.now()
    cfg, files, over = build_config(a)
    if a.sources:
        keep_sources(cfg, a.sources)
    data = load_data(cfg)
    requests = [r for r in data.requests if r.target_ids]
    if not requests:
        print(f"в сплите {a.split} нет запросов с целями", file=sys.stderr)
        return 1
    pipe = Pipeline.from_config(cfg, data.catalog)
    remote, health = check_candgen(pipe)
    if health is None:
        return 2

    run_dir = Path(a.run_dir or Path(a.out) / f"{started:%Y%m%d_%H%M%S}_{a.split}_candgen")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    per_path = run_dir / "per_list.csv"
    per_path.unlink(missing_ok=True)

    n_users = len({r.user_id for r in requests})
    sources = [r.name for r in pipe.retrievers]
    print(f"[candgen] {len(requests)} запросов от {n_users} пользователей, источники: {', '.join(sources)} -> {run_dir}")
    t0, frames = time.time(), []
    for start in range(0, len(requests), a.chunk):
        per = evaluate_candidates(pipe, requests[start:start + a.chunk], k=a.k)
        per.to_csv(per_path, mode="a", header=not frames, index=False)
        frames.append(per)
        done, spent = min(start + a.chunk, len(requests)), time.time() - t0
        rate = done / spent if spent else 0.0
        left = (len(requests) - done) / rate if rate else 0.0
        print(f"[candgen] {done}/{len(requests)}  {rate:.1f} запр/с  осталось ~{left / 60:.0f} мин", flush=True)

    per = pd.concat(frames, ignore_index=True)
    order = sources + STAGES
    summary = candidates_summary(per, k=a.k).reindex(order)
    by_type = candidates_summary(per, by="query_type", k=a.k)
    summary.to_csv(run_dir / "candgen.csv")
    by_type.to_csv(run_dir / "candgen_by_query_type.csv")
    for metric, name in (("recall", "recall_by_query_type.csv"), (f"ndcg@{a.k}", "ndcg_by_query_type.csv")):
        table = by_type[metric].unstack("list").reindex(columns=order)
        table.loc["ALL"] = summary[metric]
        table.to_csv(run_dir / name)
    losses = filter_losses(per)
    losses.to_csv(run_dir / "filter_losses.csv")

    errors = {r.name: r.n_errors for r in remote}
    result = {
        "lists": {name: {m: (None if pd.isna(v) else round(float(v), 4)) for m, v in row.items()}
                  for name, row in summary.iterrows()},
        "filter_losses": {f: round(float(v), 4) for f, v in losses.loc["ALL"].items()} if len(losses.columns) else {},
        "split": a.split, "n_requests": len(requests), "n_users": n_users, "sources": sources,
        "data": {"source": cfg["data"]["source"], "dir": cfg["data"]["crs"]["dir"]},
        "summarizer": cfg["summarizer"]["type"], "config_files": files, "overrides": over,
        "candgen": health, "candgen_errors": errors,
        "git_commit": git_commit(), "started_at": started.isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    cols = ["recall", "recall@20", "recall@100", f"ndcg@{a.k}", "only_this", "median_rank", "mean_candidates"]
    print("\n" + summary[cols].round(3).to_string())
    print("\nrecall по типам запросов:\n" + pd.read_csv(run_dir / "recall_by_query_type.csv", index_col=0).round(3).to_string())
    if len(losses.columns):
        print("\nцель выкинул фильтр (доля запросов):\n" + losses.round(3).to_string())
    if any(errors.values()):
        print(f"\nвнимание: запросы без кандидатов сервиса (ошибка сети или сервиса): {errors}")
    print(f"\nрезультат: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
