"""HTTP API for the BM25 candidate generator.

    BM25_INDEXES_DIR=indexes BM25_API_KEY=... uvicorn app:app --port 8000

Env: BM25_INDEXES_DIR (required; every subdirectory with a meta.json is an
index named after the subdirectory, e.g. indexes/genres, indexes/tags; the
'genres' index must exist and is the default), BM25_API_KEY (empty disables
auth). Indexes are loaded once at startup, read-only.

POST /bm25/search  {"words": [...], "k": 20, "exclude_ids": [...], "index": "tags"}
                -> {"ids": [...], "scores": [...], "index": "tags", "index_version": "..."}
GET  /health
"""
import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from bm25 import infer_bm25, load_bm25

log = logging.getLogger('bm25')

DEFAULT_INDEX = 'genres'
MAX_K = 1000
MAX_WORDS = 256
MAX_EXCLUDE = 10_000


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    words: list[str] = Field(max_length=MAX_WORDS)
    k: int = Field(ge=1, le=MAX_K)
    exclude_ids: list[str] = Field(default_factory=list, max_length=MAX_EXCLUDE)
    index: str | None = None


class SearchResponse(BaseModel):
    ids: list[str]
    scores: list[float]
    index: str
    index_version: str


def load_indexes(root) -> dict:
    root = Path(root)
    models = {p.name: load_bm25(p) for p in sorted(root.iterdir())
              if (p / 'meta.json').is_file()}
    if not models:
        raise RuntimeError(f'No indexes found in {root}')
    return models


@asynccontextmanager
async def lifespan(app: FastAPI):
    root = os.environ.get('BM25_INDEXES_DIR')
    if not root:
        raise RuntimeError('BM25_INDEXES_DIR is not set')
    app.state.models = load_indexes(root)
    if DEFAULT_INDEX not in app.state.models:
        raise RuntimeError(f'Default index {DEFAULT_INDEX!r} not in '
                           f'{sorted(app.state.models)}')
    app.state.api_key = os.environ.get('BM25_API_KEY', '')
    if not app.state.api_key:
        log.warning('BM25_API_KEY is empty: authentication is disabled')
    for name, model in app.state.models.items():
        log.info('Loaded BM25 index %s (%s)', name, model['index_version'])
    yield


app = FastAPI(title='BM25 candidate generator', lifespan=lifespan)


def check_api_key(request: Request,
                  x_api_key: Annotated[str | None, Header()] = None) -> None:
    expected = request.app.state.api_key
    if expected and not secrets.compare_digest((x_api_key or '').encode(),
                                               expected.encode()):
        raise HTTPException(status_code=401, detail='Invalid or missing X-API-Key')


@app.post('/bm25/search', response_model=SearchResponse,
          dependencies=[Depends(check_api_key)])
def search(body: SearchRequest, request: Request) -> SearchResponse:
    name = body.index or DEFAULT_INDEX
    model = request.app.state.models.get(name)
    if model is None:
        raise HTTPException(status_code=404, detail=(
            f'Unknown index {name!r}; available: {sorted(request.app.state.models)}'))
    result = infer_bm25(model, body.words, top_k=body.k,
                        exclude_ids=body.exclude_ids)
    return SearchResponse(ids=list(result), scores=list(result.values()),
                          index=name, index_version=model['index_version'])


@app.get('/health')
def health(request: Request) -> dict:
    return {'status': 'ok', 'default_index': DEFAULT_INDEX,
            'indexes': {name: {'index_version': m['index_version'],
                               'n_items': len(m['item_ids']),
                               'vocab_size': len(m['vocab']),
                               'id_col': m['id_col'],
                               'feature_col': m['feature_col']}
                        for name, m in request.app.state.models.items()}}
