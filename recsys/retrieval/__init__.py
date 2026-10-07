from __future__ import annotations

from typing import Any, Dict, List, Optional

from recsys.data.catalog import Catalog
from recsys.retrieval.base import BaseRetriever
from recsys.retrieval.bm25 import BM25Index, BM25Retriever, build_bm25_index
from recsys.retrieval.history import HistoryRetriever
from recsys.retrieval.hnsw_stub import HNSWStubRetriever
from recsys.retrieval.popular import PopularRetriever


def build_retrievers(cfg: Dict[str, Any], catalog: Catalog,
                     bm25_index: Optional[BM25Index] = None) -> List[BaseRetriever]:
    """Источники из cfg['retrieval'] с enabled: true, в порядке конфига."""
    rcfg = cfg.get("retrieval", {})
    excl = cfg.get("fusion", {}).get("exclude_listened", True)
    need_bm25 = any(rcfg.get(n, {}).get("enabled", False) for n in ("bm25", "history"))
    if need_bm25 and bm25_index is None:
        b = rcfg.get("bm25", {})
        bm25_index = build_bm25_index(catalog, k1=b.get("k1", 1.2), b=b.get("b", 0.75))
    out: List[BaseRetriever] = []
    for name, c in rcfg.items():
        if not c.get("enabled", False):
            continue
        k = c.get("top_k", 200)
        if name == "bm25":
            out.append(BM25Retriever(catalog, bm25_index, top_k=k, exclude_listened=excl))
        elif name == "hnsw":
            out.append(HNSWStubRetriever(catalog, top_k=k, exclude_listened=excl))
        elif name == "history":
            out.append(HistoryRetriever(catalog, bm25_index, top_k=k, exclude_listened=excl,
                                        n_tags=c.get("n_tags", 15), n_artists=c.get("n_artists", 5)))
        elif name == "popular":
            out.append(PopularRetriever(catalog, top_k=k, exclude_listened=excl))
        else:
            raise ValueError(f"Неизвестный источник кандидатов: {name}")
    return out


__all__ = ["BaseRetriever", "BM25Index", "BM25Retriever", "HNSWStubRetriever", "HistoryRetriever",
           "PopularRetriever", "build_bm25_index", "build_retrievers"]
