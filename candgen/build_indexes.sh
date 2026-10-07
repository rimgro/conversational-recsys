#!/bin/sh
# Строит все индексы кандгенов из tracks_meta.parquet.
#   sh build_indexes.sh <tracks_meta.parquet> <папка индексов>
# В образе: sh /app/build_indexes.sh /data/tracks_meta.parquet /indexes
set -e
SRC=${1:?путь к tracks_meta.parquet}
OUT=${2:?папка для индексов}
HERE=$(cd "$(dirname "$0")" && pwd)
# в образе bm25/ и hnsw/ лежат рядом со скриптом, в репозитории — уровнем выше
BM25=$HERE/bm25; [ -d "$BM25" ] || BM25=$HERE/../bm25
HNSW=$HERE/hnsw; [ -d "$HNSW" ] || HNSW=$HERE/../hnsw

python "$BM25/build_index.py" --input "$SRC" --field m4a_genres_full --out "$OUT/bm25/genres"
python "$BM25/build_index.py" --input "$SRC" --field m4a_tags_full --out "$OUT/bm25/tags"
python "$BM25/build_index.py" --input "$SRC" --field m4a_artist m4a_song --out "$OUT/bm25/title"
python "$HNSW/build_hnsw.py" --input "$SRC" --field muq_embedding --out "$OUT/hnsw/audio"
