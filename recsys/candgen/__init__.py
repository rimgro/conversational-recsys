"""Кандгены (шаг 2): каждый отдельно, с кэшем на диске; включаются в candgen.sources.<имя>.enabled.

  text     bm25, metadata_bm25, tfidf, query_tfidf, metadata_char (text.py), card_bm25 (card.py)
  history  history, history_recent, history_frequency, favorite_artists (history.py), relisten (card.py)
  audio    audio_taste, audio_recent, audio_seed (audio.py)
  dense    hnsw, twotower (dense.py)
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from recsys.candgen import audio, card, dense, history, text  # noqa: F401  (регистрация кандгенов)
from recsys.candgen import cache
from recsys.candgen.base import REGISTRY, Candgen, Cands, top_positions
from recsys.schemas import Context


def enabled_sources(cfg: Dict[str, Any]) -> List[str]:
    return [name for name, c in cfg["candgen"]["sources"].items() if c.get("enabled", False)]


def build_candgens(cfg: Dict[str, Any], res) -> List[Candgen]:
    """Включённые кандгены в порядке конфига; top_k по умолчанию — candgen.top_k."""
    ccfg = cfg["candgen"]
    out = []
    for name in enabled_sources(cfg):
        if name not in REGISTRY:
            raise ValueError(f"неизвестный кандген {name}; есть: {', '.join(sorted(REGISTRY))}")
        c = {"top_k": ccfg.get("top_k", 1000), **ccfg["sources"][name]}
        out.append(REGISTRY[name](c, res))
    return out


def retrieve(candgens: List[Candgen], contexts: List[Context], res, part: Optional[str] = None,
             batch_size: int = 256) -> Dict[str, List[Cands]]:
    """{кандген: [Cands на каждый Context]}. part — часть датасета: выдача кэшируется на диске
    (data.cache_dir/candidates/<part>/); без part (диалог, тесты) — без кэша."""
    query_ids = [str(ctx.request.request_id) for ctx in contexts]
    cache_dir = res.cfg["data"].get("cache_dir")
    out: Dict[str, List[Cands]] = {}
    for cg in candgens:
        path = cache.cache_path(cache_dir, part, cg, query_ids, res.dataset.source_key) if part else None
        got = cache.load(path, query_ids) if path else None
        if got is None:
            t0 = time.time()
            got = []
            for start in range(0, len(contexts), batch_size):
                got.extend(cg.retrieve(contexts[start:start + batch_size]))
            if path:
                cache.save(path, query_ids, got)
            if part:
                res.log(f"{part}: {cg.name} за {time.time() - t0:.0f} c")
        out[cg.name] = got
    return out


__all__ = ["Candgen", "Cands", "REGISTRY", "build_candgens", "enabled_sources", "retrieve", "top_positions"]
