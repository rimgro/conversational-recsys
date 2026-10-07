FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BM25_INDEXES_DIR=/indexes

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Only the service code. Indexes are mounted at /indexes, not baked in.
COPY bm25.py app.py build_index.py ./

RUN useradd --no-create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
