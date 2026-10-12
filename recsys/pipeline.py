"""Пайплайн: запрос -> разбор -> кандгены -> отбор -> признаки -> ранкер -> ответ.

  1. разбор запроса (dialog.py) и состояние пользователя (data/users.py)   Resources.context
  2. кандгены, каждый отдельно (candgen/)                                  Pipeline.retrieve
  3. отбор: объединение, фильтры, RRF по политике, первые limit            selection.py
  4. признаки и ранкер                                                     features.py, ranker.py
  5. ответ: шаблон или LLM                                                 explain.py

Один код для всего: пачка запросов части датасета (оценка, обучение ранкера, сабмит — recsys/run.py) и
одна реплика в диалоге (Pipeline.run -> Response, ноутбук inference.ipynb).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from recsys.candgen import Cands, build_candgens, retrieve
from recsys.explain import build_explainer, reasons_for
from recsys.features import FeatureBuilder
from recsys.ranker import BaseRanker, load_ranker
from recsys.resources import Resources
from recsys.schemas import Context, RankedTrack, Request, Response
from recsys.selection import Selection, Selector


@dataclass
class Result:
    """Выдача по одному запросу со всеми промежуточными шагами (для метрик по шагам и разбора)."""
    ctx: Context
    lists: Dict[str, Cands]                  # кандген -> его список
    selection: Selection                     # отобранные для ранкера
    scores: np.ndarray                       # скор ранкера у selection.ids (пусто — ранкер не считали)
    top: np.ndarray                          # итоговая выдача: позиции каталога, top_k
    features: Optional[pd.DataFrame] = None
    timings: Dict[str, float] = field(default_factory=dict)


class Pipeline:
    def __init__(self, cfg: Dict[str, Any], res: Optional[Resources] = None, ranker: Optional[BaseRanker] = None,
                 load_model: bool = True):
        """ranker — готовый (после обучения); load_model=False — без ранкера: только кандгены и отбор."""
        self.cfg = cfg
        self.res = res or Resources(cfg)
        self.candgens = build_candgens(cfg, self.res)
        self.sources = [c.name for c in self.candgens]
        self.selector = Selector(cfg, self.res.catalog, self.sources)
        self.features = FeatureBuilder(cfg, self.res, self.sources)
        self.ranker = ranker if ranker is not None else (load_ranker(cfg) if load_model else None)
        self.top_k = int(cfg["ranker"].get("top_k", 20))
        self._explainer = None

    @property
    def catalog(self):
        return self.res.catalog

    @property
    def explainer(self):
        if self._explainer is None:
            self._explainer = build_explainer(self.cfg, self.res.llm, self.res)
        return self._explainer

    # ------------------------------------------------------------ пачкой

    def retrieve(self, contexts: List[Context], part: Optional[str] = None) -> Dict[str, List[Cands]]:
        """Шаг 2. part — часть датасета: выдача кэшируется на диске."""
        return retrieve(self.candgens, contexts, self.res, part=part,
                        batch_size=int(self.cfg["candgen"].get("batch_size", 256)))

    def recommend(self, contexts: List[Context], part: Optional[str] = None, rank: bool = True,
                  keep_features: bool = False) -> List[Result]:
        """Шаги 2–4 для пачки. rank=False — без признаков и ранкера (метрики кандгенов и отбора)."""
        lists = self.retrieve(contexts, part)
        out = []
        for i, ctx in enumerate(contexts):
            own = {name: lists[name][i] for name in self.sources}
            sel = self.selector.select(ctx, own)
            out.append(self.rank(ctx, own, sel, keep_features) if rank else
                       Result(ctx, own, sel, np.zeros(0), self.order(ctx, sel, sel.score)))
        return out

    def rank(self, ctx: Context, lists: Dict[str, Cands], sel: Selection, keep_features: bool = False) -> Result:
        if self.ranker is None:
            raise ValueError("ранкер не загружен (Pipeline(..., load_model=False))")
        if not len(sel.ids):
            return Result(ctx, lists, sel, np.zeros(0), np.zeros(0, dtype=np.int32))
        frame = self.features.build(ctx, lists, sel)
        scores = np.asarray(self.ranker.score(frame), dtype=np.float64)
        return Result(ctx, lists, sel, scores, self.order(ctx, sel, scores), frame if keep_features else None)

    def order(self, ctx: Context, sel: Selection, scores: np.ndarray) -> np.ndarray:
        """Позиции по убыванию скора (при равенстве — меньшая позиция), первые top_k. Уже показанное и
        скипнутое в диалоге не повторяем (в датасете таких нет)."""
        ids = sel.ids[np.lexsort((sel.ids, -scores))]
        seen = ctx.request.shown_ids + ctx.request.skipped_ids
        if seen:
            ids = ids[~np.isin(ids, self.catalog.positions(seen))]
        return ids[:self.top_k]

    # ------------------------------------------------------------ одна реплика (диалог)

    def run(self, request: Request, debug: bool = False) -> Response:
        """Одна реплика: Request (диалог, история, показанное) -> Response (текст + треки)."""
        t: Dict[str, float] = {}
        t0 = time.perf_counter()
        ctx = self.res.context(request)
        t["1_context"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        lists = {name: got[0] for name, got in self.retrieve([ctx]).items()}
        t["2_candgen"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        sel = self.selector.select(ctx, lists)
        t["3_selection"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        result = self.rank(ctx, lists, sel, keep_features=debug)
        tracks = self.to_tracks(result)
        t["4_ranker"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        text = self.explainer.describe(tracks, ctx)
        t["5_explain"] = time.perf_counter() - t0
        dbg: Dict[str, Any] = {"timings": t, "n_candidates": {s: len(c) for s, c in lists.items()},
                               "n_selected": len(sel.ids)}
        if debug:
            dbg.update(result=result, profile=ctx.profile)
        return Response(text=text, tracks=tracks, summary=ctx.summary, debug=dbg)

    def to_tracks(self, result: Result) -> List[RankedTrack]:
        catalog = self.catalog
        score_of = dict(zip(result.selection.ids.tolist(), result.scores.tolist()))
        out = []
        for rank, p in enumerate(result.top, start=1):
            row = catalog.df.iloc[int(p)]
            tid = str(row["track_id"])
            out.append(RankedTrack(
                track_id=tid, rank=rank, score=float(score_of.get(int(p), 0.0)),
                title=row["title"], artist=row["artist"],
                tags=(list(row["tags"]) or list(row["genres"]))[:8],
                reasons=reasons_for(tid, result.ctx, catalog),
            ))
        return out
