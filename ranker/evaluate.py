"""nDCG / hit / MRR @k по запросам; запрос, у которого таргета нет в топе (или в пуле), даёт 0."""
from __future__ import annotations

import numpy as np
import pandas as pd


def per_query(scored: pd.DataFrame, queries: pd.DataFrame, score_col: str, k: int = 20) -> pd.DataFrame:
    """scored: qid, track_idx, score_col. queries: qid, target_idx, query_family, is_new."""
    df = scored[["qid", "track_idx", score_col]].sort_values(["qid", score_col], ascending=[True, False],
                                                             kind="stable")
    df["rank"] = df.groupby("qid", sort=False).cumcount() + 1
    tgt = queries.set_index("qid")["target_idx"]
    hit = df[df["track_idx"].to_numpy() == tgt.reindex(df["qid"]).to_numpy()]
    rank = hit.groupby("qid")["rank"].min().reindex(queries["qid"]).to_numpy()   # NaN = нет в пуле
    in_top = rank <= k
    out = queries[["qid", "query_family", "is_new"]].copy()
    out["in_pool"] = ~np.isnan(rank)
    out[f"ndcg@{k}"] = np.where(in_top, 1.0 / np.log2(np.nan_to_num(rank, nan=1.0) + 1.0), 0.0)
    out[f"hit@{k}"] = in_top.astype(float)
    out[f"mrr@{k}"] = np.where(in_top, 1.0 / np.nan_to_num(rank, nan=1.0), 0.0)
    return out


def summary(pq_: pd.DataFrame, by: str = "query_family") -> pd.DataFrame:
    metrics = [c for c in pq_.columns if "@" in c] + ["in_pool"]
    total = pq_[metrics].mean().to_frame("all").T
    total["n"] = len(pq_)
    grouped = pq_.groupby(by)[metrics].mean()
    grouped["n"] = pq_.groupby(by).size()
    return pd.concat([total, grouped]).round(4)


def compare(feats: pd.DataFrame, scores: np.ndarray, queries: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """Ранкер vs порядок RRF на том же пуле, по query_family."""
    scored = feats[["qid", "track_idx", "rrf_score"]].assign(ranker=scores)
    r = summary(per_query(scored, queries, "ranker", k))
    b = summary(per_query(scored, queries, "rrf_score", k))
    col = f"ndcg@{k}"
    return pd.DataFrame({"n": r["n"], "in_pool": r["in_pool"], f"rrf_{col}": b[col], f"ranker_{col}": r[col],
                         "delta": (r[col] - b[col]).round(4), f"ranker_hit@{k}": r[f"hit@{k}"],
                         f"ranker_mrr@{k}": r[f"mrr@{k}"]})


# ---------------------------------------------------------------- новый формат (query_id, label)

def per_query_labels(scored: pd.DataFrame, queries: pd.DataFrame, score_col: str, k: int = 20) -> pd.DataFrame:
    """scored: query_id, label, score_col (строки пула; X и артист X для similar_to уже выкинуты).
    queries: query_id, query_type, is_new — ВСЕ запросы сплита: без таргета в пуле запрос даёт 0."""
    df = scored[["query_id", "label", score_col]].sort_values(["query_id", score_col], ascending=[True, False],
                                                              kind="stable")
    df["rank"] = df.groupby("query_id", sort=False).cumcount() + 1
    rank = df[df["label"] == 1].groupby("query_id")["rank"].min().reindex(queries["query_id"]).to_numpy()
    in_top = rank <= k
    out = queries[["query_id", "query_type", "is_new"]].copy()
    out["in_pool"] = ~np.isnan(rank)
    out[f"ndcg@{k}"] = np.where(in_top, 1.0 / np.log2(np.nan_to_num(rank, nan=1.0) + 1.0), 0.0)
    out[f"hit@{k}"] = in_top.astype(float)
    out[f"mrr@{k}"] = np.where(in_top, 1.0 / np.nan_to_num(rank, nan=1.0), 0.0)
    return out


def compare_labels(scored: pd.DataFrame, queries: pd.DataFrame, score_cols, k: int = 20) -> pd.DataFrame:
    """nDCG@k по query_type и is_new для каждого скора (например ranker и rrf_score) + доля таргетов в пуле."""
    col = f"ndcg@{k}"
    parts = {}
    for sc in score_cols:
        pq_ = per_query_labels(scored, queries, sc, k)
        by_type = pq_.groupby("query_type")[[col, "in_pool"]].mean()
        by_new = pq_.groupby("is_new")[[col, "in_pool"]].mean().rename(index=lambda x: f"is_new={x}")
        total = pq_[[col, "in_pool"]].mean().to_frame("all").T
        t = pd.concat([total, by_type, by_new])
        t["n"] = [len(pq_)] + pq_.groupby("query_type").size().tolist() + pq_.groupby("is_new").size().tolist()
        parts[sc] = t
    first = next(iter(parts.values()))
    out = pd.DataFrame({"n": first["n"], "in_pool": first["in_pool"]})
    for sc, t in parts.items():
        out[f"{sc}_{col}"] = t[col]
    return out.round(4)
