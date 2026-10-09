"""main из схемы: разбор диалога -> кандидаты -> RRF -> ранкер -> описание от LLM."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from recsys.data.catalog import Catalog
from recsys.data.history import build_profile
from recsys.dialog import BaseSummarizer, build_summarizer
from recsys.explain import BaseExplainer, build_explainer
from recsys.fusion import apply_constraints, apply_filters, rrf, source_weights
from recsys.llm import BaseLLM, StubLLM, build_llm
from recsys.ranking import BaseRanker, build_features, build_ranker, cap_per_artist, reasons_for
from recsys.retrieval import BaseRetriever, CardIndex, build_retrievers
from recsys.schemas import Candidate, Context, FusedCandidate, RankedTrack, Request, Response


@dataclass
class Retrieval:
    """Шаги 1–3 пайплайна: что нашли источники и что из этого дошло до ранкера."""
    ctx: Context
    candidates: Dict[str, List[Candidate]]   # источник -> список
    weights: Dict[str, float]                # веса источников в RRF для этого запроса
    merged: List[FusedCandidate]             # RRF всех источников, до фильтров
    filtered: List[FusedCandidate]           # после фильтров и ограничений запроса (apply_filters, apply_constraints)
    fused: List[FusedCandidate]              # первые fusion.top_n: вход ранкера
    timings: Dict[str, float]


class Pipeline:
    def __init__(self, cfg: Dict[str, Any], catalog: Catalog, summarizer: BaseSummarizer,
                 retrievers: List[BaseRetriever], ranker: BaseRanker, explainer: BaseExplainer):
        self.cfg = cfg
        self.catalog = catalog
        self.summarizer = summarizer
        self.retrievers = retrievers
        self.ranker = ranker
        self.explainer = explainer

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], catalog: Catalog, llm: Optional[BaseLLM] = None,
                    bm25_index: Optional[CardIndex] = None) -> "Pipeline":
        """llm и bm25_index можно передать готовыми, чтобы не грузить/строить их заново."""
        needs_llm = cfg.get("summarizer", {}).get("type") == "llm" or cfg.get("explainer", {}).get("type") == "llm"
        if llm is None:
            llm = build_llm(cfg) if needs_llm else StubLLM()
        return cls(
            cfg=cfg,
            catalog=catalog,
            summarizer=build_summarizer(cfg, catalog, llm),
            retrievers=build_retrievers(cfg, catalog, bm25_index),
            ranker=build_ranker(cfg),
            explainer=build_explainer(cfg, llm),
        )

    @property
    def bm25_index(self) -> Optional[CardIndex]:
        """Индекс карточек (bm25 и relisten): передать в следующий Pipeline.from_config, чтобы не загружать заново."""
        return next((r.index for r in self.retrievers if r.name in ("bm25", "relisten")), None)

    def retrieve(self, request: Request) -> Retrieval:
        """Шаги 1–3: разбор диалога, кандидаты, RRF + фильтры. Без ранкера и описания (метрики кандгенов)."""
        t: Dict[str, float] = {}
        fcfg = self.cfg.get("fusion", {})

        # 1. разбор диалога
        t0 = time.perf_counter()
        profile = build_profile(request.history, self.catalog)
        summary = self.summarizer.summarize(request, profile)
        ctx = Context.build(request, summary, profile, exclude_listened=fcfg.get("exclude_listened", True))
        t["1_summary"] = time.perf_counter() - t0

        # 2. кандидаты
        t0 = time.perf_counter()
        candidates = self._candidates(ctx)
        t["2_candidates"] = time.perf_counter() - t0

        # 3. RRF + фильтры + дедуп
        t0 = time.perf_counter()
        weights = source_weights(self.cfg, ctx)
        merged = rrf(candidates, k=fcfg.get("rrf_k", 60), weights=weights)
        filtered = apply_filters(merged, ctx, self.catalog, fcfg.get("exclude_top_tags", 0))
        filtered = apply_constraints(filtered, ctx, self.catalog, fcfg.get("constraints", {}))
        fused = filtered[:fcfg.get("top_n", 500)]
        t["3_fusion"] = time.perf_counter() - t0
        return Retrieval(ctx, candidates, weights, merged, filtered, fused, t)

    def run(self, request: Request, debug: bool = False) -> Response:
        top_k = self.cfg.get("ranker", {}).get("top_k", 10)
        r = self.retrieve(request)
        ctx, candidates, fused, t = r.ctx, r.candidates, r.fused, r.timings
        profile, summary, weights = ctx.profile, ctx.summary, r.weights

        # 4. ранкер
        t0 = time.perf_counter()
        features = build_features(fused, ctx, self.catalog, sources=list(candidates))
        if features.empty:
            ranked = features.assign(rank_score=[])
        else:
            ranked = self.ranker.rank(features, ctx)
            ranked = cap_per_artist(ranked, self.catalog, self.cfg.get("diversity", {}).get("max_per_artist", 0))
        tracks = self._to_tracks(ranked.head(top_k), ctx)
        t["4_ranker"] = time.perf_counter() - t0

        # 5. описание
        t0 = time.perf_counter()
        text = self.explainer.describe(tracks, ctx)
        t["5_explain"] = time.perf_counter() - t0

        dbg: Dict[str, Any] = {"timings": t, "weights": weights,
                               "n_candidates": {s: len(c) for s, c in candidates.items()},
                               "n_fused": len(fused)}
        if debug:
            dbg.update(profile=profile, candidates=candidates, fused=fused, features=ranked)
        return Response(text=text, tracks=tracks, summary=summary, debug=dbg)

    def _candidates(self, ctx: Context) -> Dict[str, List[Candidate]]:
        """Удалённые источники (HTTP к VPS) опрашиваются параллельно, пока считаются локальные."""
        remote = [r for r in self.retrievers if getattr(r, "remote", False)]
        if not remote:
            return {r.name: r.search(ctx) for r in self.retrievers}
        with ThreadPoolExecutor(max_workers=len(remote)) as pool:
            futures = {r.name: pool.submit(r.search, ctx) for r in remote}
            local = {r.name: r.search(ctx) for r in self.retrievers if r.name not in futures}
            return {r.name: futures[r.name].result() if r.name in futures else local[r.name]
                    for r in self.retrievers}

    def _to_tracks(self, ranked: pd.DataFrame, ctx: Context) -> List[RankedTrack]:
        out = []
        feat_cols = [c for c in ranked.columns if c != "track_id"]
        for i, row in enumerate(ranked.to_dict("records"), start=1):
            tid = row["track_id"]
            info = self.catalog.row(tid) or {}
            out.append(RankedTrack(
                track_id=tid,
                rank=i,
                score=float(row["rank_score"]),
                title=info.get("title", ""),
                artist=info.get("artist", ""),
                tags=(list(info.get("tags", [])) or list(info.get("genres", [])))[:8],
                reasons=reasons_for(tid, ctx, self.catalog),
                features={c: row[c] for c in feat_cols},
            ))
        return out
