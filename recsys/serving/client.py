"""Bounded HTTP client and left-tower-to-right-tower candidate retrieval."""
from urllib.parse import urlsplit
import math

import httpx
from pydantic import ValidationError
from .contract import SearchRequest, SearchResponse


class CandidateAPIError(RuntimeError):
    pass


class CandidateAPIClient:
    def __init__(self, url, api_key, bundle_id, release_id, timeout=10., transport=None):
        parsed = urlsplit(url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Expected an HTTP(S) base URL without credentials or query parameters")
        if not api_key or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("API key and a finite positive timeout are required")
        self.bundle_id, self.release_id = bundle_id, release_id
        self.http = httpx.Client(base_url=url.rstrip("/") + "/", transport=transport,
            headers={"X-API-Key": api_key}, timeout=httpx.Timeout(timeout, connect=min(timeout, 5.)),
            follow_redirects=False, trust_env=False)
        self.checked = False

    def close(self):
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _call(self, method, path, **kwargs):
        try:
            # Stream so an invalid peer cannot supply an unbounded response.
            with self.http.stream(method, path, **kwargs) as response:
                if response.status_code != 200:
                    raise CandidateAPIError(f"Candidate API HTTP {response.status_code}")
                raw = bytearray()
                for block in response.iter_bytes():
                    raw.extend(block)
                    if len(raw) > 1024 * 1024:
                        raise CandidateAPIError("Candidate API response too large")
                import json
                return json.loads(raw)
        except (httpx.HTTPError, ValueError) as error:
            # Do not leak the API key, user history, vector or URL credentials.
            raise CandidateAPIError("Candidate API transport or JSON error") from None

    def check_health(self):
        health = self._call("GET", "health")
        if not isinstance(health, dict) or (health.get("status"), health.get("api_version"),
                health.get("dimension"), health.get("bundle_id"), health.get("release_id")) != (
                "ok", 1, 256, self.bundle_id, self.release_id):
            raise CandidateAPIError("Query encoder and index releases differ")
        self.checked = True
        return health

    def search(self, vector, **options):
        body = SearchRequest(bundle_id=self.bundle_id, release_id=self.release_id,
            vector=list(vector), **options)
        if not self.checked:
            self.check_health()
        try:
            result = SearchResponse.model_validate(self._call("POST", "v1/search", json=body.model_dump()))
        except ValidationError:
            raise CandidateAPIError("Invalid candidate API response") from None
        if result.bundle_id != self.bundle_id or result.release_id != self.release_id:
            raise CandidateAPIError("Query encoder and index releases differ")
        candidates = result.candidates
        if (result.returned != len(candidates) or result.returned > body.k or result.mode != body.mode
                or result.ids != [x.m4a_id for x in candidates]
                or result.scores != [x.score for x in candidates]
                or len(set(result.ids)) != len(result.ids)
                or any(a < b - 1e-6 for a, b in zip(result.scores, result.scores[1:]))):
            raise CandidateAPIError("Inconsistent candidate API response")
        return result.model_dump()


class RemoteQwenRetriever:
    def __init__(self, encoder, client):
        if (encoder.serving_manifest["bundle_id"], encoder.serving_manifest["release_id"]) != (
                client.bundle_id, client.release_id):
            raise ValueError("Encoder and client identities differ")
        self.encoder, self.client = encoder, client

    @classmethod
    def from_bundle(cls, bundle_dir, url, api_key, device="cuda", cache_dir=None, timeout=10.):
        from .artifacts import load_subset
        serving, _ = load_subset(bundle_dir, "left")
        client = CandidateAPIClient(url, api_key, serving["bundle_id"], serving["release_id"], timeout)
        try:
            client.check_health()  # Reject wrong index before downloading/loading the Qwen base.
            from .query import QwenQueryEncoder
            return cls(QwenQueryEncoder(bundle_dir, device, cache_dir), client)
        except BaseException:
            client.close()
            raise

    def search(self, query, history=(), k=200, **options):
        # Validate network options before expensive encoding.
        history = list(history)
        SearchRequest(bundle_id=self.client.bundle_id, release_id=self.client.release_id,
            vector=[1.] + [0.] * 255, history=history, k=k, **options)
        vector = self.encoder.encode_query(query, history)
        return self.client.search(vector.tolist(), history=history, k=k, **options)

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
