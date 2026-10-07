"""HTTP API for the HNSW (embedding) candidate generator.

    HNSW_INDEXES_DIR=../indexes/hnsw uvicorn service:app --port 8001     # standalone

In the candidate-generator image it is mounted next to BM25 (candgen/main.py).

Env: HNSW_INDEXES_DIR (every subdirectory with a meta.json is an index named
after the subdirectory; 'audio' is the default), HNSW_API_KEY (falls back to
BM25_API_KEY; empty disables auth). Indexes are loaded once at startup.

POST /hnsw/search  {"track_ids": [...], "weights": [...], "k": 200, "exclude_ids": [...], "index": "audio"}
               or  {"vector": [...], "k": 200, ...}
               ->  {"ids": [...], "scores": [...], "index": "audio", "index_version": "...", "n_query_tracks": 2}
GET  /hnsw/health
"""
import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hnsw_index import _positions, infer_hnsw, load_hnsw

log = logging.getLogger('hnsw')

DEFAULT_INDEX = 'audio'
MAX_K = 1000
MAX_TRACKS = 10_000
MAX_EXCLUDE = 100_000


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    k: int = Field(ge=1, le=MAX_K)
    vector: list[float] | None = None
    track_ids: list[str] = Field(default_factory=list, max_length=MAX_TRACKS)
    weights: list[float] | None = Field(default=None, max_length=MAX_TRACKS)
    exclude_ids: list[str] = Field(default_factory=list, max_length=MAX_EXCLUDE)
    index: str | None = None

    @model_validator(mode='after')
    def _one_query(self):
        if (self.vector is None) == (not self.track_ids):
            raise ValueError('pass exactly one of: vector, track_ids')
        if self.weights is not None and len(self.weights) != len(self.track_ids):
            raise ValueError('weights and track_ids must have the same length')
        return self


class SearchResponse(BaseModel):
    ids: list[str]
    scores: list[float]
    index: str
    index_version: str
    n_query_tracks: int


def load_indexes(root) -> dict:
    root = Path(root)
    return {p.name: load_hnsw(p) for p in sorted(root.iterdir()) if (p / 'meta.json').is_file()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads indexes into app.state.hnsw_models; an empty/missing dir disables HNSW (BM25 still works)."""
    root = os.environ.get('HNSW_INDEXES_DIR', '')
    app.state.hnsw_models = load_indexes(root) if root and Path(root).is_dir() else {}
    if not app.state.hnsw_models:
        log.warning('No HNSW indexes in %r: /hnsw/search will return 503', root)
    app.state.hnsw_api_key = os.environ.get('HNSW_API_KEY', os.environ.get('BM25_API_KEY', ''))
    for name, model in app.state.hnsw_models.items():
        log.info('Loaded HNSW index %s (%s, %d items)', name, model['index_version'], len(model['item_ids']))
    yield


def check_api_key(request: Request, x_api_key: Annotated[str | None, Header()] = None) -> None:
    expected = request.app.state.hnsw_api_key
    if expected and not secrets.compare_digest((x_api_key or '').encode(), expected.encode()):
        raise HTTPException(status_code=401, detail='Invalid or missing X-API-Key')


router = APIRouter()


@router.post('/hnsw/search', response_model=SearchResponse, dependencies=[Depends(check_api_key)])
def search(body: SearchRequest, request: Request) -> SearchResponse:
    models = request.app.state.hnsw_models
    if not models:
        raise HTTPException(status_code=503, detail='No HNSW indexes loaded')
    name = body.index or DEFAULT_INDEX
    model = models.get(name)
    if model is None:
        raise HTTPException(status_code=404, detail=f'Unknown index {name!r}; available: {sorted(models)}')
    try:
        result = infer_hnsw(model, vector=body.vector, track_ids=body.track_ids or None, weights=body.weights,
                            top_k=body.k, exclude_ids=body.exclude_ids)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None
    pos = _positions(model)
    n_known = 1 if body.vector is not None else sum(str(t) in pos for t in body.track_ids)
    return SearchResponse(ids=list(result), scores=list(result.values()), index=name,
                          index_version=model['index_version'], n_query_tracks=n_known)


@router.get('/hnsw/health')
def health(request: Request) -> dict:
    return {'status': 'ok' if request.app.state.hnsw_models else 'no_indexes', 'default_index': DEFAULT_INDEX,
            'indexes': {name: {'index_version': m['index_version'], 'n_items': len(m['item_ids']),
                               'dim': int(m['vectors'].shape[1]), 'vector_col': m['vector_col']}
                        for name, m in request.app.state.hnsw_models.items()}}


app = FastAPI(title='HNSW candidate generator', lifespan=lifespan)
app.include_router(router)
