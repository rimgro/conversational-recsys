#!/bin/sh
# Строит индексы кандгенов из tracks_meta.parquet; bm25 и hnsw независимо.
#   sh deploy/build_indexes.sh <bm25|hnsw|all> <tracks_meta.parquet> <папка индексов>
# У каждого кандгена свой venv: BM25_PYTHON, HNSW_PYTHON (по умолчанию python).
# Сервис читает индексы один раз при старте: после пересборки — systemctl restart recsys-<кандген>.
set -e
WHAT=${1:?bm25, hnsw или all}
SRC=${2:?путь к tracks_meta.parquet}
OUT=${3:?папка для индексов}
ROOT=$(cd "$(dirname "$0")/.." && pwd)

case $WHAT in bm25|hnsw|all) ;; *) echo "первый аргумент: bm25, hnsw или all, а не $WHAT" >&2; exit 2 ;; esac

if [ "$WHAT" != hnsw ]; then
  PY=${BM25_PYTHON:-python}
  "$PY" "$ROOT/bm25/build_index.py" --input "$SRC" --field m4a_genres_full --out "$OUT/bm25/genres"
  "$PY" "$ROOT/bm25/build_index.py" --input "$SRC" --field m4a_tags_full --out "$OUT/bm25/tags"
  "$PY" "$ROOT/bm25/build_index.py" --input "$SRC" --field m4a_artist m4a_song --out "$OUT/bm25/title"
fi
if [ "$WHAT" != bm25 ]; then
  "${HNSW_PYTHON:-python}" "$ROOT/hnsw/build_hnsw.py" --input "$SRC" --field muq_embedding --out "$OUT/hnsw/audio"
fi
