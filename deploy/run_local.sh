#!/bin/sh
# Локально то же, что на VPS, только без systemd: BM25 и HNSW двумя процессами на 8001 и 8002.
#   sh deploy/run_local.sh <tracks_meta.parquet> [папка индексов, по умолчанию indexes]
# Индексы строятся, если их ещё нет (пересобрать — удалить папку). Ctrl+C останавливает оба сервиса.
# Python с зависимостями bm25/ и hnsw/: BM25_PYTHON, HNSW_PYTHON (по умолчанию python из PATH).
set -e
SRC=${1:?путь к tracks_meta.parquet}
IDX=${2:-indexes}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BM25_PY=$(command -v "${BM25_PYTHON:-python}")
HNSW_PY=$(command -v "${HNSW_PYTHON:-python}")
for PY in "$BM25_PY" "$HNSW_PY"; do
  "$PY" -c 'import sys, fastapi, uvicorn; sys.exit(sys.version_info < (3, 10))' 2>/dev/null || {
    echo "$PY: нужен Python 3.10+ с fastapi и uvicorn, например:" >&2
    echo "  python3.12 -m venv .venv-candgen && .venv-candgen/bin/pip install -r bm25/requirements.txt -r hnsw/requirements.txt" >&2
    echo "  BM25_PYTHON=.venv-candgen/bin/python HNSW_PYTHON=.venv-candgen/bin/python sh deploy/run_local.sh $SRC" >&2
    exit 1; }
done

[ -f "$IDX/bm25/genres/meta.json" ] || BM25_PYTHON=$BM25_PY sh "$ROOT/deploy/build_indexes.sh" bm25 "$SRC" "$IDX"
[ -f "$IDX/hnsw/audio/meta.json" ] || HNSW_PYTHON=$HNSW_PY sh "$ROOT/deploy/build_indexes.sh" hnsw "$SRC" "$IDX"
IDX=$(cd "$IDX" && pwd)

# те же команды, что в ExecStart юнитов deploy/recsys-*.service
(cd "$ROOT/bm25" && BM25_INDEXES_DIR="$IDX/bm25" exec "$BM25_PY" -m uvicorn app:app --host 127.0.0.1 --port 8001) &
BM25_PID=$!
(cd "$ROOT/hnsw" && HNSW_INDEXES_DIR="$IDX/hnsw" exec "$HNSW_PY" -m uvicorn service:app --host 127.0.0.1 --port 8002) &
HNSW_PID=$!
trap 'kill $BM25_PID $HNSW_PID 2>/dev/null' INT TERM EXIT
echo "BM25 http://127.0.0.1:8001  HNSW http://127.0.0.1:8002  проверка: python deploy/check.py"
wait
