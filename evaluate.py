"""Прогон валидации: метрики пайплайна на сплите датасета (по умолчанию test_public).

    python evaluate.py --data-dir "CRS dataset" --n-users 1000             # быстро: 1000 пользователей
    python evaluate.py --data-dir "CRS dataset" --n-users all --candgen    # весь сплит, BM25 и HNSW с VPS
    python evaluate.py --synthetic                                          # без данных: проверить, что всё работает
    python evaluate.py --data-dir data/crs --set ranker.type=heuristic      # любой параметр конфига

Результат — папка outputs/<время>_<сплит>/ (или --run-dir):
  metrics.json          средние метрики и как запускали: конфиги, сколько запросов, время, git-коммит,
                        версии индексов сервисов и сколько запросов прошло без их кандидатов
  by_query_family.csv   метрики по типам запросов
  by_is_new.csv         по новым и уже знакомым трекам
  per_request.csv       метрики каждого запроса; дописывается по ходу, при обрыве прогона не пропадает
  config.yaml           полный конфиг прогона

С --candgen (BM25 и HNSW с сервера) перед стартом проверяется /health: если сервис недоступен, прогон не начинается,
чтобы метрики не посчитались молча без его кандидатов. Адреса и ключи — BM25_URL, BM25_API_KEY, HNSW_URL,
HNSW_API_KEY (из окружения или файла .env).
Полный прогон в облаке без открытого ноутбука: jobs/evaluate.yaml (DataSphere Jobs).
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from recsys.config import load_config
from recsys.data import load_data
from recsys.eval import evaluate, metrics_by
from recsys.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
ID_COLUMNS = ["request_id", "query_family", "is_new"]


def config_path(name: str) -> str:
    """configs/ рядом с текущей папкой (так в DataSphere Jobs) или рядом со скриптом."""
    local = Path("configs") / name
    return str(local if local.exists() else HERE / "configs" / name)


def count(value: str):
    """'all' -> None (без ограничения), иначе число."""
    return None if value == "all" else int(value)


def set_key(cfg: dict, dotted: str, value) -> None:
    *path, last = dotted.split(".")
    for key in path:
        cfg = cfg.setdefault(key, {})
    cfg[last] = value


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def build_config(a: argparse.Namespace):
    files = [config_path("default.yaml")] + ([config_path("candgen.yaml")] if a.candgen else []) + a.config
    over: dict = {"data": {"eval_split": a.split}}
    if a.synthetic:
        over["data"]["source"] = "synthetic"
        if a.n_users is not None:
            over["data"]["synthetic"] = {"n_users": count(a.n_users)}
    else:
        crs: dict = {"splits": [a.split]}  # читаем только нужный сплит
        if a.data_dir:
            crs["dir"] = a.data_dir
        if a.n_users is not None:
            crs["n_users"] = count(a.n_users)
        over["data"].update(source="crs", crs=crs)
    if a.max_positives is not None:
        over["data"]["max_positives_per_user"] = count(a.max_positives)
    for item in a.set:
        key, _, value = item.partition("=")
        set_key(over, key, yaml.safe_load(value))
    return load_config(files, over), files, over


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--split", default="test_public", help="сплит датасета (test_public, train)")
    p.add_argument("--data-dir", help="папка датасета (файлы целиком или частями); по умолчанию data.crs.dir")
    p.add_argument("--synthetic", action="store_true", help="синтетика вместо датасета")
    p.add_argument("--n-users", help="сколько пользователей взять из сплита; all — все (по умолчанию из конфига)")
    p.add_argument("--max-positives", help="запросов на пользователя; all — все (по умолчанию из конфига)")
    p.add_argument("--candgen", action="store_true", help="BM25 и HNSW с сервера по HTTP (configs/candgen.yaml)")
    p.add_argument("--config", action="append", default=[], help="ещё YAML поверх (можно несколько)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="параметр конфига, например ranker.type=heuristic (можно несколько)")
    p.add_argument("--k", type=int, default=10, help="метрики @k")
    p.add_argument("--out", default="outputs", help="куда класть папку прогона")
    p.add_argument("--run-dir", help="точная папка результата вместо outputs/<время>_<сплит>")
    p.add_argument("--chunk", type=int, default=500, help="через сколько запросов дописывать результат")
    a = p.parse_args(argv)

    started = datetime.now()
    cfg, files, over = build_config(a)
    data = load_data(cfg)
    requests = [r for r in data.requests if r.target_ids]
    if not requests:
        print(f"в сплите {a.split} нет запросов с целями", file=sys.stderr)
        return 1
    pipe = Pipeline.from_config(cfg, data.catalog)

    remote = [r for r in pipe.retrievers if getattr(r, "remote", False)]
    health = {r.name: r.health() for r in remote}
    down = {name: h for name, h in health.items() if h.get("status") != "ok"}
    for name, h in health.items():
        print(f"[candgen] {name}: {h}")
    if down:
        print(f"сервисы недоступны: {sorted(down)}; проверьте адреса и ключи в .env (BM25_URL, HNSW_URL, ...)", file=sys.stderr)
        return 2

    run_dir = Path(a.run_dir or Path(a.out) / f"{started:%Y%m%d_%H%M%S}_{a.split}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    per_path = run_dir / "per_request.csv"
    per_path.unlink(missing_ok=True)

    n_users = len({r.user_id for r in requests})
    print(f"[eval] {len(requests)} запросов от {n_users} пользователей -> {run_dir}")
    t0, frames = time.time(), []
    for start in range(0, len(requests), a.chunk):
        per, _ = evaluate(pipe, requests[start:start + a.chunk], k=a.k, verbose=False)
        per.to_csv(per_path, mode="a", header=not frames, index=False)
        frames.append(per)
        done, spent = start + len(per), time.time() - t0
        rate = done / spent if spent else 0.0
        left = (len(requests) - done) / rate if rate else 0.0
        print(f"[eval] {done}/{len(requests)}  {rate:.1f} запр/с  осталось ~{left / 60:.0f} мин", flush=True)

    per = pd.concat(frames, ignore_index=True)
    mean = per.drop(columns=ID_COLUMNS).mean(numeric_only=True)
    by_family, by_new = metrics_by(per, "query_family"), metrics_by(per, "is_new")
    by_family.to_csv(run_dir / "by_query_family.csv")
    by_new.to_csv(run_dir / "by_is_new.csv")
    errors = {r.name: r.n_errors for r in remote}
    summary = {
        "metrics": {name: round(float(v), 4) for name, v in mean.items()},
        "split": a.split, "n_requests": len(per), "n_users": n_users,
        "data": {"source": cfg["data"]["source"], "dir": cfg["data"]["crs"]["dir"]},
        "ranker": cfg["ranker"]["type"], "summarizer": cfg["summarizer"]["type"],
        "config_files": files, "overrides": over,
        "candgen": health, "candgen_errors": errors,
        "git_commit": git_commit(), "started_at": started.isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    main_cols = [c for c in (f"hit@{a.k}", f"ndcg@{a.k}", f"mrr@{a.k}", "recall@fused") if c in mean]
    print("\n" + mean[main_cols].round(3).to_string())
    print("\n" + by_family[["n"] + main_cols].round(3).to_string())
    if any(errors.values()):
        print(f"\nвнимание: запросы без кандидатов сервиса (ошибка сети или сервиса): {errors}")
    print(f"\nрезультат: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
