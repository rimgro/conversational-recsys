"""Пул кандидатов: hnsw + bm25_genres + bm25_tags, по top_k от каждого -> RRF -> top_n.

Источник — любой объект с `name` и `search(query, words, k) -> [(track_id, score), ...]`
(по убыванию score). query — русский текст запроса, words — слова для BM25
(жанры/теги на английском), которые даёт words_fn.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

Hits = List[Tuple[str, float]]
WordsFn = Callable[[str], List[str]]


class CandidateSource(Protocol):
    name: str

    def search(self, query: str, words: List[str], k: int) -> Hits: ...


class BM25IndexSource:
    """Читает индекс в формате ветки dev/bm25 (index.npz, item_ids.npy, meta.json) и ищет так же,
    как bm25.infer_bm25: каждое слово запроса учитывается один раз, ничьи — по порядку каталога."""

    def __init__(self, name: str, index_dir: str | Path):
        path = Path(index_dir)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        self.name = name
        self.index = load_npz(path / "index.npz").tocsc()
        self.item_ids = np.load(path / "item_ids.npy", allow_pickle=False)
        self.vocab: Dict[str, int] = meta["vocab"]

    def search(self, query: str, words: List[str], k: int) -> Hits:
        terms = {w for s in words for w in s.strip().casefold().split()}
        cols = sorted(self.vocab[t] for t in terms if t in self.vocab)
        if not cols:
            return []
        scores = np.asarray(self.index[:, cols].sum(axis=1)).ravel()
        cand = np.flatnonzero(scores > 0)
        order = cand[np.lexsort((cand, -scores[cand]))[:k]]
        return [(str(self.item_ids[i]), float(scores[i])) for i in order]


class BM25HttpSource:
    """Тот же поиск через сервис dev/bm25: POST /bm25/search."""

    def __init__(self, name: str, url: str, index: str, api_key: Optional[str] = None, timeout: float = 10.0):
        self.name, self.url, self.index, self.api_key, self.timeout = name, url, index, api_key, timeout

    def search(self, query: str, words: List[str], k: int) -> Hits:
        if not words:
            return []
        body = json.dumps({"words": words[:256], "k": k, "index": self.index}).encode()
        headers = {"Content-Type": "application/json", **({"X-API-Key": self.api_key} if self.api_key else {})}
        req = urllib.request.Request(self.url, data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read())
        return list(zip(out["ids"], out["scores"]))


class FunctionSource:
    """Обёртка для hnsw (или любого поиска по тексту запроса): fn(query, k) -> [(track_id, score)]."""

    def __init__(self, name: str, fn: Callable[[str, int], Hits]):
        self.name, self.fn = name, fn

    def search(self, query: str, words: List[str], k: int) -> Hits:
        return list(self.fn(query, k))


def rrf(lists: Dict[str, Sequence[str]], k: int = 60, top_n: int = 200):
    """score(d) = sum_s 1/(k + rank_s(d)), rank с 1. Возвращает [(track_id, rrf_score, n_sources)]."""
    score: Dict[str, float] = {}
    n_src: Dict[str, int] = {}
    for ids in lists.values():
        for rank, tid in enumerate(dict.fromkeys(ids), start=1):
            score[tid] = score.get(tid, 0.0) + 1.0 / (k + rank)
            n_src[tid] = n_src.get(tid, 0) + 1
    top = sorted(score, key=lambda t: (-score[t], t))[:top_n]
    return [(t, score[t], n_src[t]) for t in top]


def parse_words(queries: pd.DataFrame, words_fn: WordsFn) -> pd.Series:
    """Слова для BM25 по каждому запросу (одинаковый текст парсится один раз)."""
    cache: Dict[str, List[str]] = {}
    out = []
    for q in queries["query"]:
        if q not in cache:
            cache[q] = [w.strip().lower() for w in words_fn(q) if w and w.strip()]
        out.append(cache[q])
    return pd.Series(out, index=queries.index)


def build_pools(queries: pd.DataFrame, sources: Sequence[CandidateSource], tracks: pd.DataFrame,
                top_k: int = 100, top_n: int = 200, rrf_k: int = 60,
                progress_every: int = 0) -> pd.DataFrame:
    """queries нужны колонки qid, query, words. -> qid, track_idx, rrf_score, n_sources."""
    idx_of = dict(zip(tracks["track_id"], tracks["track_idx"]))
    qids, tidx, rs, ns = [], [], [], []
    for i, (qid, query, words) in enumerate(zip(queries["qid"], queries["query"], queries["words"])):
        lists = {s.name: [t for t, _ in s.search(query, list(words), top_k)] for s in sources}
        for tid, score, n in rrf(lists, k=rrf_k, top_n=top_n):
            j = idx_of.get(tid)
            if j is None:
                continue
            qids.append(qid), tidx.append(j), rs.append(score), ns.append(n)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"[pools] {i + 1}/{len(queries)}", flush=True)
    return pd.DataFrame({"qid": qids, "track_idx": np.asarray(tidx, dtype=np.int32),
                         "rrf_score": np.asarray(rs, dtype=np.float32),
                         "n_sources": np.asarray(ns, dtype=np.int8)})


def rule_words_fn(data_dir: str | Path) -> WordsFn:
    """Временный парсер «русский запрос -> английские теги»: RuleSummarizer из recsys (include_tags).
    Заменить на парсер команды, когда он появится; в train и test должен быть один и тот же."""
    from recsys.data.crs import catalog_from_meta, read_tracks_meta
    from recsys.dialog import RuleSummarizer
    from recsys.schemas import Message, Request

    from ranker.data import find_files
    meta = pd.concat([read_tracks_meta(str(f)) for f in find_files(data_dir, "tracks_meta")], ignore_index=True)
    summarizer = RuleSummarizer(catalog_from_meta(meta.drop_duplicates("m4a_id").reset_index(drop=True)))

    def fn(query: str) -> List[str]:
        s = summarizer.summarize(Request(dialog=[Message("user", query)]))
        return list(s.include_tags)
    return fn
