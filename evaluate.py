"""Прогон валидации: метрики пайплайна на сплите датасета (по умолчанию test_public).

    python evaluate.py --n-users 1000                       # 1000 пользователей test_public, BM25 и HNSW с сервера
    python evaluate.py --n-users all                        # весь сплит
    python evaluate.py --n-users 1000 --offline             # без сервисов: только relisten и audio, быстро
    python evaluate.py --set ranker.type=heuristic          # любой параметр конфига
    python evaluate.py --synthetic                          # без данных: проверить, что код работает

Метрика как в датасете — nDCG@20 (для similar_to без трека-образца и его артиста), по query_type.
Результат — папка outputs/<время>_<сплит>/ (или --run-dir):
  metrics.json          средние метрики и как запускали: конфиги, сколько запросов, время, git-коммит,
                        версии индексов сервисов и сколько запросов прошло без их кандидатов
  submission.parquet    сабмит в формате датасета: query_id, top20 (m4a_id по убыванию), response
  by_query_type.csv     метрики по типам запросов
  by_is_new.csv         по новым и уже знакомым трекам
  by_source.csv         кандгены: recall, уникальный вклад, место цели, сколько кандидатов
  by_source_query_type.csv  recall каждого кандгена по типам запросов
  per_request.csv       метрики каждого запроса; дописывается по ходу, при обрыве прогона не пропадает
  config.yaml           полный конфиг прогона

Перед стартом проверяется /health сервисов BM25 и HNSW: если сервис недоступен, прогон не начинается (код выхода 2),
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
from recsys.eval import OUTPUT_COLUMNS, evaluate, metrics_by, sources_by, sources_summary
from recsys.pipeline import Pipeline

HERE = Path(__file__).resolve().parent


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


REMOTE_SOURCES = ("bm25", "hnsw")


def build_config(a: argparse.Namespace):
    files = [config_path("default.yaml")] + a.config
    over: dict = {"data": {"eval_split": a.split}}
    if a.offline or a.synthetic:  # у сервисов каталог датасета: синтетических треков они не знают
        over["retrieval"] = {name: {"enabled": False} for name in REMOTE_SOURCES}
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
    if a.max_queries is not None:
        over["data"]["max_queries_per_user"] = count(a.max_queries)
    for item in a.set:
        key, _, value = item.partition("=")
        set_key(over, key, yaml.safe_load(value))
    return load_config(files, over), files, over


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Аргументы, общие для evaluate.py и evaluate_candgen.py: данные, конфиг, куда писать."""
    p.add_argument("--split", default="test_public", help="сплит датасета (test_public, train)")
    p.add_argument("--data-dir", help="папка датасета (файлы целиком или частями); по умолчанию data.crs.dir")
    p.add_argument("--synthetic", action="store_true", help="синтетика вместо датасета")
    p.add_argument("--n-users", help="сколько пользователей взять из сплита; all — все (по умолчанию из конфига)")
    p.add_argument("--max-queries", help="запросов на пользователя; all — все (по умолчанию из конфига)")
    p.add_argument("--offline", action="store_true", help="без сервисов BM25 и HNSW: только локальные источники")
    p.add_argument("--config", action="append", default=[], help="ещё YAML поверх (можно несколько)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="параметр конфига, например ranker.type=heuristic (можно несколько)")
    p.add_argument("--k", type=int, default=20, help="метрики @k (в датасете — 20)")
    p.add_argument("--out", default="outputs", help="куда класть папку прогона")
    p.add_argument("--run-dir", help="точная папка результата вместо outputs/<время>_<сплит>")
    p.add_argument("--chunk", type=int, default=500, help="через сколько запросов дописывать результат")


def check_candgen(pipe: Pipeline):
    """/health удалённых источников -> (источники, ответы); ответы None, если какой-то сервис недоступен."""
    remote = [r for r in pipe.retrievers if getattr(r, "remote", False)]
    health = {r.name: r.health() for r in remote}
    down = {name: h for name, h in health.items() if h.get("status") != "ok"}
    for name, h in health.items():
        print(f"[candgen] {name}: {h}")
    if down:
        print(f"сервисы недоступны: {sorted(down)}; проверьте адреса и ключи в .env (BM25_URL, HNSW_URL, ...)", file=sys.stderr)
        return remote, None
    return remote, health


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(p)
    a = p.parse_args(argv)

    started = datetime.now()
    cfg, files, over = build_config(a)
    data = load_data(cfg)
    requests = [r for r in data.requests if r.target_ids]
    if not requests:
        print(f"в сплите {a.split} нет запросов с целями", file=sys.stderr)
        return 1
    pipe = Pipeline.from_config(cfg, data.catalog)

    remote, health = check_candgen(pipe)
    if health is None:
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
        per, _ = evaluate(pipe, requests[start:start + a.chunk], k=a.k, verbose=False, outputs=True)
        per.drop(columns=OUTPUT_COLUMNS).to_csv(per_path, mode="a", header=not frames, index=False)
        frames.append(per)
        done, spent = start + len(per), time.time() - t0
        rate = done / spent if spent else 0.0
        left = (len(requests) - done) / rate if rate else 0.0
        print(f"[eval] {done}/{len(requests)}  {rate:.1f} запр/с  осталось ~{left / 60:.0f} мин", flush=True)

    per = pd.concat(frames, ignore_index=True)
    pd.DataFrame({"query_id": per["request_id"], f"top{a.k}": per["top"], "response": per["response"]}) \
        .to_parquet(run_dir / "submission.parquet", index=False)
    per = per.drop(columns=OUTPUT_COLUMNS)
    mean = per[[c for c in per.columns if "@" in c]].mean(numeric_only=True)
    by_family, by_new = metrics_by(per, "query_type"), metrics_by(per, "is_new")
    by_family.to_csv(run_dir / "by_query_type.csv")
    by_new.to_csv(run_dir / "by_is_new.csv")
    by_source, source_by_type = sources_summary(per), sources_by(per, "query_type")
    by_source.to_csv(run_dir / "by_source.csv")
    source_by_type.to_csv(run_dir / "by_source_query_type.csv")
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

    main_cols = [c for c in (f"ndcg@{a.k}", f"hit@{a.k}", f"mrr@{a.k}", "recall@fused") if c in mean]
    print("\n" + mean[main_cols].round(3).to_string())
    print("\n" + by_family[["n"] + main_cols].round(3).to_string())
    print("\nкандгены:\n" + by_source.round(3).to_string())
    if any(errors.values()):
        print(f"\nвнимание: запросы без кандидатов сервиса (ошибка сети или сервиса): {errors}")
    print(f"\nрезультат: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
