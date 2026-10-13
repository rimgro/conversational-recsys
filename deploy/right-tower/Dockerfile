FROM python:3.12-slim AS wheels
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*
ENV HNSWLIB_NO_NATIVE=1
COPY requirements-serving.txt /tmp/requirements.txt
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r /tmp/requirements.txt

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app && useradd --uid 10001 --gid app --create-home app
COPY --from=wheels /wheels /wheels
COPY requirements-serving.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-index --find-links /wheels -r /tmp/requirements.txt \
    && rm -rf /wheels /tmp/requirements.txt
WORKDIR /app
COPY recsys/__init__.py /app/recsys/__init__.py
COPY recsys/serving /app/recsys/serving
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    QWEN_RIGHT_BUNDLE=/data/right QWEN_API_KEY_FILE=/run/secrets/qwen_api_key
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).read()"
CMD ["python", "-m", "uvicorn", "recsys.serving.app:create_env_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "8", "--timeout-keep-alive", "5", "--no-access-log"]
