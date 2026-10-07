# Образ 1: кандидат-генераторы BM25 + HNSW. Собирать из КОРНЯ репозитория:
#   docker build -f candgen/Dockerfile -t recsys-candgen .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BM25_INDEXES_DIR=/indexes/bm25 \
    HNSW_INDEXES_DIR=/indexes/hnsw

WORKDIR /app

COPY candgen/requirements.txt .
RUN pip install -r requirements.txt

# Только код сервисов и сборки индексов. Индексы монтируются в /indexes, а не запекаются в образ.
COPY bm25/bm25.py bm25/app.py bm25/build_index.py bm25/
COPY hnsw/hnsw_index.py hnsw/service.py hnsw/build_hnsw.py hnsw/
COPY candgen/main.py candgen/build_indexes.sh ./

RUN useradd --no-create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
