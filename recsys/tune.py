"""Подбор весов отбора (selection.rrf_k, selection.weights) по recall@limit на части valid: python -m recsys tune-policy.

Жадно добавляем кандгены: на каждом шаге — тот, что сильнее всего поднимает recall@limit, вместе с лучшими rrf_k
(tune.rrf_ks) и весом текстовых и плотных кандгенов (tune.text_weights; остальные — вес 1). Останавливаемся, когда
прирост меньше tune.min_gain. Одни веса для всех запросов: тип запроса модели не известен.

Результат — runs/<прогон>/selection.yaml (подключить: --config runs/<прогон>/selection.yaml или перенести веса в
configs/default.yaml) и recall по типам запросов до и после (selection_by_type.csv, тип — только для отчёта).
Подбирается на valid: на test он не смотрит, поэтому оценка на test остаётся честной.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import yaml

from recsys.candgen import REGISTRY
from recsys.pipeline import Pipeline

SEMANTIC_KINDS = ("text", "dense")   # их вес — text_weight


@dataclass
class Query:
    query_type: str          # только для отчёта по типам; в подборе не участвует
    ranks: np.ndarray        # (n_union, n_sources) int32: место в списке кандгена, 0 — нет в списке
    target: int              # строка цели в ranks; -1 — цели нет среди кандидатов (после фильтров)


class PolicyTuner:
    def __init__(self, sources: Sequence[str], queries: List[Query], limit: int,
                 rrf_ks: Sequence[float] = (20, 60), text_weights: Sequence[float] = (1, 2, 4),
                 min_gain: float = 0.002):
        self.sources = list(sources)
        self.queries = queries
        self.limit = limit
        self.rrf_ks, self.text_weights, self.min_gain = list(rrf_ks), list(text_weights), min_gain
        self.semantic = np.array([REGISTRY[s].kind in SEMANTIC_KINDS for s in self.sources])

    def recall(self, queries: List[Query], weights: np.ndarray, k: float) -> float:
        """Доля запросов, где цель попала в первые limit (скор > 0; при равенстве — меньшая позиция)."""
        hit = 0
        for q in queries:
            if q.target < 0:
                continue
            r = q.ranks
            score = (np.where(r > 0, 1.0 / (k + r), 0.0) * weights).sum(axis=1)
            st = score[q.target]
            if st <= 0:
                continue
            better = (score > st).sum() + (score[:q.target] == st).sum()   # строки ranks — по возрастанию позиции
            hit += better < self.limit
        return hit / max(len(queries), 1)

    def weights(self, chosen: List[int], tw: float) -> np.ndarray:
        w = np.zeros(len(self.sources))
        for i in chosen:
            w[i] = tw if self.semantic[i] else 1.0
        return w

    def tune(self, queries: List[Query]) -> Dict[str, Any]:
        chosen: List[int] = []
        best = {"recall": 0.0, "rrf_k": self.rrf_ks[-1], "tw": 1.0}
        while len(chosen) < len(self.sources):
            step = None
            for i in (j for j in range(len(self.sources)) if j not in chosen):
                for k in self.rrf_ks:
                    for tw in self.text_weights:
                        rec = self.recall(queries, self.weights(chosen + [i], tw), k)
                        if step is None or rec > step[0] + 1e-12:
                            step = (rec, i, k, tw)
            if step is None or step[0] - best["recall"] < self.min_gain:
                break
            chosen.append(step[1])
            best = {"recall": step[0], "rrf_k": step[2], "tw": step[3]}
        w = self.weights(chosen, best["tw"])
        return {"rrf_k": int(best["rrf_k"]),
                "weights": {self.sources[i]: _num(w[i]) for i in chosen}, "recall": best["recall"]}


def _num(x: float):
    return int(x) if float(x).is_integer() else float(x)


def collect(pipe: Pipeline, part: str) -> List[Query]:
    """Для каждого запроса части: места кандидатов (после фильтров отбора) в списках каждого кандгена."""
    from recsys.run import recommend_part
    out = []
    for chunk in recommend_part(pipe, part, rank=False):
        for r in chunk:
            union = r.selection.union
            ranks = np.zeros((len(union), len(pipe.sources)), dtype=np.int32)
            for j, name in enumerate(pipe.sources):
                c = r.lists[name]
                if len(c):
                    at = np.searchsorted(union, c.ids)
                    ok = (at < len(union)) & (union[np.minimum(at, len(union) - 1)] == c.ids)
                    ranks[at[ok], j] = np.flatnonzero(ok) + 1
            target = pipe.catalog.pos(r.ctx.request.target_ids[0]) if r.ctx.request.target_ids else None
            row = np.flatnonzero(union == target) if target is not None else []
            out.append(Query(str(r.ctx.request.meta.get("query_type") or ""), ranks, int(row[0]) if len(row) else -1))
    return out


def tune_policy(cfg: Dict[str, Any], part: str = "valid") -> str:
    from recsys.run import run_dir, save_config
    save_config(cfg)
    pipe = Pipeline(cfg, load_model=False)
    queries = collect(pipe, part)
    tcfg = cfg.get("tune", {})
    tuner = PolicyTuner(pipe.sources, queries, pipe.selector.limit, tcfg.get("rrf_ks", (20, 60)),
                        tcfg.get("text_weights", (1, 2, 4)), tcfg.get("min_gain", 0.002))
    tuned = tuner.tune(queries)
    sel = pipe.selector
    current = (np.array([sel.weights.get(s, 0.0) for s in pipe.sources]), sel.rrf_k)
    new = (np.array([float(tuned["weights"].get(s, 0.0)) for s in pipe.sources]), tuned["rrf_k"])
    rows = []
    for t in ["ALL"] + sorted({q.query_type for q in queries}):
        qs = queries if t == "ALL" else [q for q in queries if q.query_type == t]
        rows.append({"query_type": t, "queries": len(qs), "recall_current": tuner.recall(qs, *current),
                     "recall_tuned": tuner.recall(qs, *new)})
    d = run_dir(cfg)
    path = os.path.join(d, "selection.yaml")
    with open(path, "w", encoding="utf8") as f:
        f.write(f"# python -m recsys tune-policy: часть {part}, {len(queries)} запросов, recall@{tuner.limit} "
                f"{rows[0]['recall_current']:.4f} -> {rows[0]['recall_tuned']:.4f}\n"
                f"# кандгены: {', '.join(pipe.sources)}\n")
        # все кандгены явно (0 — не в отборе): конфиги накладываются рекурсивно, иначе останутся веса из default
        weights = {s: tuned["weights"].get(s, 0) for s in dict.fromkeys(list(pipe.sources) + list(sel.weights))}
        yaml.safe_dump({"selection": {"rrf_k": tuned["rrf_k"], "weights": weights}}, f,
                       default_flow_style=None, allow_unicode=True, sort_keys=False, width=200)
    table = pd.DataFrame(rows).set_index("query_type")
    table.to_csv(os.path.join(d, "selection_by_type.csv"))
    print("\n" + table.round(4).to_string())
    pipe.res.log(f"веса отбора: {tuned['weights']}, rrf_k {tuned['rrf_k']} -> {path}; использовать — --config {path}")
    return path
