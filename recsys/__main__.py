"""Точка запуска: python -m recsys <команда> [опции]. Запускать из корня репозитория.

    python -m recsys build-index                    # индексы кандгенов (иначе соберутся при первом запуске)
    python -m recsys candgen                        # кандгены и отбор без ранкера: отчёт по каждому кандгену
    python -m recsys tune-policy                    # подобрать веса отбора на valid (после добавления кандгенов)
    python -m recsys train                          # обучить ранкер (ranker_train, early stopping по valid)
    python -m recsys evaluate                       # метрики всех шагов на valid и test
    python -m recsys run                            # train + evaluate
    python -m recsys submit                         # сабмит по всему test_public

Опции:
    --run NAME                 папка прогона runs/NAME/ (модель, отчёты, сабмит)
    --parts valid,test         на каких частях считать метрики (recsys/data/parts.py)
    --sources +hnsw,-tfidf     включить / выключить кандгены; без знаков (bm25,history) — только эти
    --set a.b=c                любой параметр конфига (значение — YAML: 3, true, [a, b])
    --config extra.yaml        ещё конфиг поверх configs/default.yaml (можно несколько)
    --synthetic                синтетика вместо датасета: проверить, что код работает

Примеры:
    python -m recsys candgen --sources +card_bm25,+relisten,+audio_seed --run more_candgens
    python -m recsys tune-policy --sources +hnsw --run hnsw     # затем run с --config runs/hnsw/selection.yaml
    python -m recsys run --set ranker.catboost.depth=8 --run depth8
    python -m recsys evaluate --set ranker.type=stub --run stub
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List

import yaml

from recsys.config import config_path, load_config, set_key

COMMANDS = ("build-index", "candgen", "tune-policy", "train", "evaluate", "run", "submit")


def apply_sources(cfg: Dict[str, Any], spec: str) -> None:
    """'+a,-b' — включить a, выключить b; 'a,b' — только a и b."""
    items = [s.strip() for s in spec.split(",") if s.strip()]
    sources = cfg["candgen"]["sources"]
    names = [s.lstrip("+-") for s in items]
    unknown = sorted(set(names) - set(sources))
    if unknown:
        raise SystemExit(f"неизвестные кандгены: {unknown}; есть: {', '.join(sources)}")
    if all(s[0] not in "+-" for s in items):
        for name, c in sources.items():
            c["enabled"] = name in names
        return
    if not all(s[0] in "+-" for s in items):
        raise SystemExit("--sources: либо все со знаком (+a,-b), либо все без (a,b)")
    for s in items:
        sources[s[1:]]["enabled"] = s[0] == "+"


def build_config(a: argparse.Namespace) -> Dict[str, Any]:
    over: Dict[str, Any] = {}
    for item in a.set:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set {item}: нужно KEY=VALUE")
        set_key(over, key, yaml.safe_load(value))
    if a.run:
        set_key(over, "run.name", a.run)
    if a.synthetic:
        set_key(over, "data.source", "synthetic")
    cfg = load_config([config_path("default.yaml")] + a.config, over)
    if a.sources:
        apply_sources(cfg, a.sources)
    return cfg


def main(argv: List[str] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m recsys", description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("command", choices=COMMANDS)
    p.add_argument("--run", help="имя прогона: runs/<имя>/ (по умолчанию run.name)")
    p.add_argument("--parts", default="valid,test", help="части для метрик (candgen, evaluate, run)")
    p.add_argument("--sources", help="+a,-b — включить / выключить кандгены; a,b — только эти")
    p.add_argument("--config", action="append", default=[], help="ещё YAML поверх configs/default.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="параметр конфига")
    p.add_argument("--synthetic", action="store_true", help="синтетика вместо датасета")
    a = p.parse_args(argv)

    from recsys import run
    cfg = build_config(a)
    parts = [x.strip() for x in a.parts.split(",") if x.strip()]
    if a.command == "build-index":
        run.build_indexes(cfg)
    elif a.command == "candgen":
        run.candgen(cfg, parts)
    elif a.command == "tune-policy":
        from recsys.tune import tune_policy
        tune_policy(cfg)
    elif a.command == "train":
        run.train(cfg)
    elif a.command == "evaluate":
        run.evaluate(cfg, parts)
    elif a.command == "run":
        ranker = run.train(cfg)
        run.evaluate(cfg, parts, ranker=ranker)
    elif a.command == "submit":
        run.submit(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
