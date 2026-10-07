# Образ 2: наша часть — пакет recsys и main.ipynb (разбор запроса, локальные источники, RRF, ранкер, ответ).
# BM25 и HNSW вызываются по HTTP из образа 1 (candgen/): configs/candgen.yaml, адрес в CANDGEN_URL.
#   docker build -t recsys-app .
#   docker run --rm -e CANDGEN_URL=http://<образ 1>:8000 -v "$PWD/data:/app/data" recsys-app pytest -q
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CANDGEN_URL=http://localhost:8000

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt ipykernel

COPY recsys/ recsys/
COPY configs/ configs/
COPY examples/ examples/
COPY tests/ tests/
COPY candgen/build_indexes.sh candgen/
COPY main.ipynb scheme.png ./

CMD ["python", "-c", "import recsys; print('recsys', recsys.__version__, '- открыть main.ipynb или запустить pytest -q')"]
