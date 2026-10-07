"""Клиент ранкера по API (на схеме «ранкер по api»).

POST {url}  {"features": [{...}, ...], "context": {...}}  ->  {"scores": [float, ...]}
При ошибке сети или неверном ответе используется fallback-ранкер.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Optional

import pandas as pd

from recsys.ranking.base import BaseRanker
from recsys.ranking.stub import StubRanker
from recsys.schemas import Context


class APIRanker(BaseRanker):
    def __init__(self, url: str, timeout: float = 2.0, api_key: Optional[str] = None,
                 fallback: Optional[BaseRanker] = None):
        self.url = url
        self.timeout = timeout
        self.api_key = api_key
        self.fallback = fallback or StubRanker()
        self.last_error: Optional[str] = None

    def rank(self, features: pd.DataFrame, ctx: Context) -> pd.DataFrame:
        payload = {
            "features": json.loads(features.to_json(orient="records")),
            "context": {"summary": ctx.summary.text, "include_tags": ctx.summary.include_tags,
                        "user_id": ctx.request.user_id},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        try:
            req = urllib.request.Request(self.url, data=json.dumps(payload).encode("utf8"), headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                scores = json.loads(r.read().decode("utf8"))["scores"]
            if len(scores) != len(features):
                raise ValueError(f"ожидали {len(features)} скоров, получили {len(scores)}")
        except Exception as e:
            self.last_error = str(e)
            return self.fallback.rank(features, ctx)
        out = features.copy()
        out["rank_score"] = [float(s) for s in scores]
        return out.sort_values("rank_score", ascending=False, kind="stable").reset_index(drop=True)
