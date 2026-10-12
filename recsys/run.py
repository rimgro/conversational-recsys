"""Этапы прогона (их запускает python -m recsys <команда>, recsys/__main__.py). Результаты — в runs/<прогон>/:

  config.yaml                 полный конфиг прогона
  ranker/                     модель ранкера (model.cbm, features.json, importance.csv)
  <часть>/report.md           главное: метрики каждого кандгена, объединения, отбора и ранкера; по типам запросов
  <часть>/lists.csv           строка = список (кандген / union / selection / ranker), все метрики
  <часть>/<метрика>_by_type.csv   тип запроса × список (recall, recall@150, ndcg@20)
  <часть>/per_list.parquet    строка = запрос × список: место цели, длина списка
  <часть>/metrics.json        главные числа
  submission.parquet          сабмит (query_id, top20, response)

Выдача кандгенов кэшируется в data.cache_dir/candidates/: повторный прогон с другим ранкером или весами отбора
кандгены не пересчитывает.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from recsys import metrics
from recsys.pipeline import Pipeline, Result
from recsys.ranker import CatBoostRanker, model_dir
from recsys.resources import Resources


def run_dir(cfg: Dict[str, Any]) -> str:
    run = cfg.get("run", {})
    d = os.path.join(run.get("dir", "runs"), run.get("name", "baseline"))
    os.makedirs(d, exist_ok=True)
    return d


def save_config(cfg: Dict[str, Any]) -> None:
    with open(os.path.join(run_dir(cfg), "config.yaml"), "w", encoding="utf8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def recommend_part(pipe: Pipeline, part: str, rank: bool, labels: bool = True) -> Iterator[List[Result]]:
    """Выдача по всем запросам части кусками по run.chunk запросов (иначе списки кандгенов сабмита не влезут
    в память). Выдача кандгенов кэшируется по кускам (data.cache_dir/candidates/), признаки ранкера — блоками
    (data.cache_dir/features/): готовое берётся с диска."""
    res = pipe.res
    requests = res.dataset.requests(part, labels=labels)
    size = int(pipe.cfg.get("run", {}).get("chunk", 20000))
    t0 = time.time()
    for n, start in enumerate(range(0, len(requests), size)):
        contexts = res.contexts(requests[start:start + size])
        name = part if len(requests) <= size else f"{part}/{n:03d}"
        lists = pipe.retrieve(contexts, name)
        out = []
        for i, ctx in enumerate(contexts):
            own = {s: lists[s][i] for s in pipe.sources}
            sel = pipe.selector.select(ctx, own)
            out.append(Result(ctx, own, sel, np.zeros(0), pipe.order(ctx, sel, sel.score)))
        if rank:
            score(pipe, part, out)
        res.log(f"{part}: {start + len(contexts)}/{len(requests)} запросов за {time.time() - t0:.0f} c")
        yield out


def score(pipe: Pipeline, part: str, results: List[Result]) -> None:
    """Скоры ранкера и итоговая выдача (r.scores, r.top) по признакам из кэша; ранкер — одним вызовом на блок."""
    if pipe.ranker is None:
        raise ValueError("ранкер не загружен")
    for frames, block in features_for(pipe, part, results):
        n = [len(f) for f in frames]
        if not sum(n):
            continue
        scores = np.asarray(pipe.ranker.score(pd.concat(frames, ignore_index=True)), dtype=np.float64)
        for r, a, b in zip(block, np.cumsum([0] + n[:-1]), np.cumsum(n)):
            if b > a:
                r.scores = scores[a:b]
                r.top = pipe.order(r.ctx, r.selection, r.scores)


# ---------------------------------------------------------------- отчёт

def per_list(cfg: Dict[str, Any], pipe: Pipeline, part: str, rank: bool) -> pd.DataFrame:
    """Строка = запрос × список (metrics.per_list) по всей части."""
    k = int(cfg["eval"].get("k", 20))
    return pd.concat([metrics.per_list(chunk, pipe.selector, k=k) for chunk in recommend_part(pipe, part, rank)],
                     ignore_index=True)


def report(cfg: Dict[str, Any], pipe: Pipeline, per: pd.DataFrame, part: str) -> Dict[str, Any]:
    """Метрики по шагам -> runs/<прогон>/<часть>/; возвращает главные числа."""
    k, recall_at = int(cfg["eval"].get("k", 20)), list(cfg["eval"]["recall_at"])
    limit = int(cfg["selection"]["limit"])
    if limit not in recall_at:
        recall_at = sorted(recall_at + [limit])
    out_dir = os.path.join(run_dir(cfg), part)
    os.makedirs(out_dir, exist_ok=True)
    if per.empty:
        raise ValueError(f"в части {part} нет запросов с ответами")
    per.to_parquet(os.path.join(out_dir, "per_list.parquet"), index=False)
    lists = metrics.summary(per, recall_at, k)
    lists.to_csv(os.path.join(out_dir, "lists.csv"))
    tables = {}
    for metric in ("recall", f"recall@{limit}", f"ndcg@{k}"):
        tables[metric] = metrics.by_type(per, metric, k, recall_at)
        tables[metric].to_csv(os.path.join(out_dir, f"{metric.replace('@', '_at_')}_by_type.csv"))
    losses = metrics.filter_losses(per)
    main = {
        "part": part, "queries": int(per["request_id"].nunique()),
        "candgens": pipe.sources, "ranker": cfg["ranker"]["type"],
        f"ndcg@{k}": _get(lists, "ranker", f"ndcg@{k}"), f"hit@{k}": _get(lists, "ranker", f"hit@{k}"),
        f"stub_ndcg@{k}": _get(lists, "selection", f"ndcg@{k}"),
        f"recall@{limit}": _get(lists, "selection", "recall"), "recall_union": _get(lists, "union", "recall"),
        "filter_losses": {f: round(float(v), 4) for f, v in losses.items()},
        "not_in_policy": pipe.selector.unused,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf8") as f:
        json.dump(main, f, ensure_ascii=False, indent=2)
    md = render_report(main, lists, tables, k, limit, recall_at, losses)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf8") as f:
        f.write(md)
    pipe.res.log(f"отчёт: {out_dir}/report.md")
    print("\n" + md)
    return main


def _get(table: pd.DataFrame, row: str, col: str) -> Optional[float]:
    return round(float(table.loc[row, col]), 4) if row in table.index else None


def render_report(main, lists, tables, k, limit, recall_at, losses) -> str:
    cols = [f"recall@{n}" for n in recall_at] + ["recall", "only_this", f"ndcg@{k}", "median_rank", "mean_len"]
    lines = [f"# {main['part']}: {main['queries']} запросов", ""]
    if main[f"ndcg@{k}"] is not None:
        lines.append(f"**Ранкер ({main['ranker']}) nDCG@{k} = {main[f'ndcg@{k}']:.4f}**, hit@{k} = {main[f'hit@{k}']:.4f}; "
                     f"порядок отбора (stub) nDCG@{k} = {main[f'stub_ndcg@{k}']:.4f}")
    lines += [f"Отбор: recall@{limit} = {main[f'recall@{limit}']:.4f}; все кандгены вместе: recall = "
              f"{main['recall_union']:.4f}"]
    if main["not_in_policy"]:
        lines.append(f"Без веса в отборе (только признаки ранкера): {', '.join(main['not_in_policy'])}")
    lines.append("")
    lines += ["## Списки", "", "recall@N — цель в первых N; recall — где угодно; only_this — нашёл только этот кандген; "
              f"nDCG@{k} — если бы выдачей был этот список как есть.", "", _md(lists[cols]), ""]
    for metric, title in ((f"ndcg@{k}", f"nDCG@{k} по типам запросов"), ("recall", "recall по типам запросов")):
        lines += [f"## {title}", "", _md(tables[metric]), ""]
    if len(losses):
        lines += ["## Цель выкинул фильтр отбора (доля запросов)", "", _md(losses.to_frame("share")), ""]
    return "\n".join(lines)


COUNT_COLUMNS = {"queries", "median_rank", "mean_len"}


def _md(df: pd.DataFrame) -> str:
    """Таблица markdown: доли — 4 знака, счётчики — целые."""
    def cell(col, v) -> str:
        if pd.isna(v):
            return ""
        if col in COUNT_COLUMNS:
            return f"{v:.0f}"
        return f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)

    head = "| " + " | ".join([str(df.index.name or "")] + [str(c) for c in df.columns]) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    rows = ["| " + " | ".join([str(idx)] + [cell(c, v) for c, v in row.items()]) + " |" for idx, row in df.iterrows()]
    return "\n".join([head, sep] + rows)


# ---------------------------------------------------------------- команды

def candgen(cfg: Dict[str, Any], parts: List[str]) -> Dict[str, Any]:
    """Кандгены и отбор без ранкера: отчёт по спискам."""
    save_config(cfg)
    pipe = Pipeline(cfg, load_model=False)
    return {part: report(cfg, pipe, per_list(cfg, pipe, part, rank=False), part) for part in parts}


FEATURES_VERSION = "2"   # поднять, если поменялся код признаков, а конфиг — нет
FEATURE_BLOCK = 5000     # признаки сохраняются блоками по столько запросов: после остановки прогон продолжит с блока


def feature_block_path(pipe: Pipeline, part: str, results: List[Result]) -> Optional[str]:
    """data.cache_dir/features/<часть>/<ключ>.parquet. Ключ — всё, от чего зависят признаки: запросы блока,
    кандгены и их параметры, отбор, группы признаков, версия данных (содержимое tracks_meta) и кода."""
    cache_dir = pipe.cfg["data"].get("cache_dir")
    if not cache_dir:
        return None
    deps = {"queries": [str(r.ctx.request.request_id) for r in results], "data": pipe.res.dataset.source_key,
            "candgens": [c.cache_key() for c in pipe.candgens], "selection": pipe.cfg["selection"],
            "features": pipe.cfg["features"], "v": FEATURES_VERSION}
    key = hashlib.md5(json.dumps(deps, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return os.path.join(cache_dir, "features", part.split("/")[0], f"{key}.parquet")


def features_for(pipe: Pipeline, part: str, results: List[Result]) -> Iterator[Tuple[List[pd.DataFrame], List[Result]]]:
    """Признаки отобранных кандидатов каждого запроса (строки — в порядке selection.ids), блоками по
    FEATURE_BLOCK запросов: блок с диска, если он уже посчитан с теми же кандгенами, отбором и признаками."""
    for start in range(0, len(results), FEATURE_BLOCK):
        block = results[start:start + FEATURE_BLOCK]
        path = feature_block_path(pipe, part, block)
        t0 = time.time()
        if path and os.path.exists(path):
            x = pd.read_parquet(path)
            how = "с диска"
        else:
            parts = []
            for i, r in enumerate(block):
                f = pipe.features.build(r.ctx, r.lists, r.selection) if len(r.selection.ids) else pd.DataFrame()
                parts.append(f.assign(_query=np.int32(i)) if len(f) else f)
            x = pd.concat([f for f in parts if len(f)], ignore_index=True) if any(len(f) for f in parts) \
                else pd.DataFrame({"_query": np.zeros(0, np.int32)})
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                x.to_parquet(path + ".tmp", index=False)
                os.replace(path + ".tmp", path)
            how = "посчитаны"
        q = x.pop("_query").to_numpy()
        bounds = np.searchsorted(q, np.arange(len(block) + 1))   # строки запроса i — [bounds[i], bounds[i+1])
        frames = [x.iloc[bounds[i]:bounds[i + 1]].reset_index(drop=True) for i in range(len(block))]
        pipe.res.log(f"{part}: признаки запросов {start + len(block)}/{len(results)} ({how}) "
                     f"за {time.time() - t0:.0f} c")
        yield frames, block


def training_data(pipe: Pipeline, part: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, int]]:
    """Признаки, метки и группы для ранкера: запросы, где цель среди отобранных (как у team1: без сэмплирования
    негативов и без принудительного добавления цели). Признаки — из кэша (features_for)."""
    xs, labels, groups, n_queries = [], [], [], 0
    for chunk in recommend_part(pipe, part, rank=False):
        n_queries += len(chunk)
        for frames, block in features_for(pipe, part, chunk):
            for r, f in zip(block, frames):
                target = pipe.catalog.pos(r.ctx.request.target_ids[0])
                if target not in r.selection.ids:
                    continue
                xs.append(f)
                labels.append((r.selection.ids == target).astype(np.int8))
                groups.append(np.full(len(f), len(xs) - 1, dtype=np.int32))
    x = pd.concat(xs, ignore_index=True)
    stats = {"queries": n_queries, "groups": len(xs), "rows": len(x)}
    pipe.res.log(f"{part}: {stats}")
    return x, np.concatenate(labels), np.concatenate(groups), stats


def train(cfg: Dict[str, Any]) -> CatBoostRanker:
    """Обучение ранкера на ranker_train, early stopping по valid; модель -> ranker.model_dir."""
    if cfg["ranker"]["type"] != "catboost":
        raise ValueError(f"ranker.type: {cfg['ranker']['type']} не обучается")
    save_config(cfg)
    pipe = Pipeline(cfg, load_model=False)
    train_data = training_data(pipe, "ranker_train")
    valid_data = training_data(pipe, "valid")
    d = model_dir(cfg)
    os.makedirs(d, exist_ok=True)
    ranker = CatBoostRanker.fit(train_data[:3], valid_data[:3], cfg["ranker"]["catboost"],
                                threads=cfg["ranker"].get("threads", 6), train_dir=os.path.join(d, "catboost_info"))
    ranker.save(d, {"candgens": pipe.sources, "groups": cfg["features"]["groups"],
                    "train": train_data[3], "valid": valid_data[3], "trees": int(ranker.model.tree_count_)})
    ranker.importance.to_csv(os.path.join(d, "importance.csv"), index=False)
    pipe.res.log(f"ранкер: {ranker.model.tree_count_} деревьев -> {d}")
    return ranker


def evaluate(cfg: Dict[str, Any], parts: List[str], ranker=None) -> Dict[str, Any]:
    save_config(cfg)
    pipe = Pipeline(cfg, ranker=ranker)
    return {part: report(cfg, pipe, per_list(cfg, pipe, part, rank=True), part) for part in parts}


def submit(cfg: Dict[str, Any], part: str = "submit", path: Optional[str] = None) -> str:
    """Сабмит: все запросы test_split без ответов -> query_id, top20 (m4a_id), response (explainer)."""
    save_config(cfg)
    pipe = Pipeline(cfg)
    k = int(cfg["eval"].get("k", 20))
    if pipe.top_k < k:
        raise ValueError(f"ranker.top_k = {pipe.top_k} < {k}")
    ids = pipe.catalog.track_ids
    rows = []
    for chunk in recommend_part(pipe, part, rank=True, labels=False):
        for r in chunk:
            tracks = pipe.to_tracks(r)
            rows.append({"query_id": r.ctx.request.request_id, f"top{k}": [str(ids[p]) for p in r.top[:k]],
                         "response": pipe.explainer.describe(tracks, r.ctx)})
    df = pd.DataFrame(rows)
    short = int((df[f"top{k}"].map(len) < k).sum())
    if short:
        pipe.res.log(f"внимание: у {short} запросов меньше {k} треков")
    path = path or os.path.join(run_dir(cfg), "submission.parquet")
    df.to_parquet(path, index=False)
    pipe.res.log(f"сабмит: {len(df)} запросов -> {path}")
    return path


def build_indexes(cfg: Dict[str, Any]) -> None:
    """Собрать (или проверить в кэше) индексы включённых кандгенов и признаков."""
    res = Resources(cfg)
    names = [n for n, c in cfg["candgen"]["sources"].items() if c.get("enabled")]
    from recsys.candgen.text import TEXT_INDEXES
    for name in sorted(set(names) & set(TEXT_INDEXES) | {"bm25", "tfidf"}):   # bm25 и tfidf нужны признакам
        res.text_index(name, cfg["candgen"]["sources"].get(name, {}).get("index"))
    if {"card_bm25", "relisten"} & set(names):
        _ = res.card_index
    res.log("индексы готовы")
