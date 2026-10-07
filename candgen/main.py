"""Образ 1: кандидат-генераторы BM25 + HNSW в одном HTTP-сервисе.

    uvicorn main:app --host 0.0.0.0 --port 8000

Код кандгенов не меняется: здесь только сборка. Маршруты BM25 берутся из bm25/app.py,
HNSW — из hnsw/service.py; их lifespan (загрузка индексов) запускаются по очереди.

POST /bm25/search   {"words": [...], "k": 200, "exclude_ids": [...], "index": "tags"}       (см. bm25/RUN.md)
POST /hnsw/search   {"track_ids": [...], "weights": [...], "k": 200, "index": "audio"}      (см. hnsw/service.py)
GET  /health        состояние BM25;  GET /hnsw/health  состояние HNSW

Env: BM25_INDEXES_DIR, HNSW_INDEXES_DIR, BM25_API_KEY (HNSW_API_KEY, если ключ другой).
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.routing import APIRoute

# в образе: /app/main.py рядом с /app/bm25, /app/hnsw; в репозитории: candgen/main.py рядом с ../bm25, ../hnsw
_HERE = Path(__file__).resolve().parent
for _name in ('bm25', 'hnsw'):
    _dir = next(d for d in (_HERE / _name, _HERE.parent / _name) if d.is_dir())
    sys.path.insert(0, str(_dir))

import app as bm25_app  # noqa: E402  bm25/app.py
import service as hnsw_service  # noqa: E402  hnsw/service.py


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with bm25_app.lifespan(app):
        async with hnsw_service.lifespan(app):
            yield


app = FastAPI(title='Candidate generators: BM25 + HNSW', lifespan=lifespan)
# только API-маршруты BM25 (без его /docs и /openapi.json, они есть у этого приложения)
app.router.routes.extend(r for r in bm25_app.app.router.routes if isinstance(r, APIRoute))
app.include_router(hnsw_service.router)
