"""Organizer Ranking/Discovery; Diversity requires the official implementation."""
import math

import numpy as np
import pandas as pd

from recsys.track_tower import text_value


class CatalogScorer:
    def __init__(self, records, query_types, diversity=None):
        self.ids = [str(r["m4a_id"]) for r in records]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Duplicate m4a IDs")
        self.lookup = {t: i for i, t in enumerate(self.ids)}
        self.keys = [("spotify:" + text_value(r.get("spotify_id"))) if text_value(r.get("spotify_id"))
                     else "m4a:" + str(r["m4a_id"]) for r in records]
        self.artists = [(text_value(r.get("m4a_artist")) or text_value(r.get("artist"))).casefold()
                        for r in records]
        groups = {key: i for i, key in enumerate(sorted(set(self.keys)))}
        self.group_ids = np.array([groups[k] for k in self.keys], dtype=np.int64)
        self.query_types, self.diversity = tuple(sorted(query_types)), diversity

    def history_keys(self, history):
        return {self.keys[self.lookup[str(t)]] for t in history if str(t) in self.lookup}

    def clean_ranking(self, ranking, row, listened=frozenset(), discovery=False, k=20):
        excluded_id = text_value(row.get("exclude_m4a_id"))
        excluded_key = self.keys[self.lookup[excluded_id]] if excluded_id in self.lookup else None
        excluded_artist = text_value(row.get("exclude_artist")).casefold()
        seen, result = set(), []
        for tid in ranking:
            index = self.lookup.get(str(tid))
            if index is None:
                continue
            key = self.keys[index]
            if key in seen:
                continue
            seen.add(key)
            if row["query_type"] == "similar_to":
                if key == excluded_key or (excluded_artist and self.artists[index] == excluded_artist):
                    continue
            if discovery and key in listened:
                continue
            result.append(self.ids[index])
            if len(result) >= k:
                break
        return result

    def score(self, ranking, row, listened):
        target = self.lookup[str(row["target_m4a_id"])]
        key = self.keys[target]
        normal = self.clean_ranking(ranking, row)
        def ndcg(rec):
            for position, tid in enumerate(rec, 1):
                if self.keys[self.lookup[tid]] == key:
                    return 1. / math.log2(position + 1)
            return 0.
        is_new = key not in listened
        novel = self.clean_ranking(ranking, row, listened, discovery=True) if is_new else []
        diversity = self.diversity(normal[:10], row["query_type"]) if self.diversity else np.nan
        if self.diversity and (not np.isfinite(diversity) or not 0 <= diversity <= 1):
            raise ValueError("Diversity scorer must return a finite score in [0,1]")
        return {"query_id": row["query_id"], "query_type": row["query_type"],
                "ranking": ndcg(normal), "discovery": ndcg(novel) if is_new else np.nan,
                "diversity": diversity, "is_new_canonical": is_new,
                "top20": normal, "new_top20": novel}

    def summarize(self, rows):
        frame = pd.DataFrame(rows)
        by_type = frame.groupby("query_type").agg(
            queries=("query_id", "size"), new_queries=("discovery", "count"),
            ranking=("ranking", "mean"), discovery=("discovery", "mean"),
            diversity=("diversity", "mean")).reindex(self.query_types)
        def complete_macro(column):
            values = by_type[column]
            return float(values.mean()) if values.notna().all() else None
        summary = {"macro_ranking_ndcg20": complete_macro("ranking"),
                   "macro_discovery_ndcg20": complete_macro("discovery"),
                   "macro_diversity": complete_macro("diversity"),
                   "ranking_types_present": int(by_type.ranking.notna().sum()),
                   "discovery_types_present": int(by_type.discovery.notna().sum()),
                   "expected_types": len(self.query_types), "queries": len(frame),
                   "diversity_status": "official_plugin" if self.diversity else "not_available"}
        return summary, frame, by_type
