# BM25 service

HTTP API for lexical candidate search over `tracks_meta`. Python 3.10+.

## Setup

```bash
pip install -r requirements.txt
```

## Build indexes

Point `--input` at `tracks_meta.jsonl`, `tracks_meta.parquet` or a folder of `tracks_meta-*.parquet` shards:

```bash
python build_index.py --input ../data/tracks_meta.jsonl --field m4a_genres_full --out indexes/genres
python build_index.py --input ../data/tracks_meta.jsonl --field m4a_tags_full --out indexes/tags
```

Each index is a folder in `indexes/`; its name is the `index` value in requests.
The `genres` index is required (it is the default). Other indexes are picked up automatically.
Add one with `build_index.py --field <column> ... --out indexes/<name>` (see its docstring).

## Run

```bash
BM25_INDEXES_DIR=indexes uvicorn app:app --port 8000
```

- API page: http://localhost:8000/docs
- Health and loaded indexes: http://localhost:8000/health

Indexes are read once at startup: after rebuilding, restart the service.

## API

`POST /bm25/search`

```json
{"words": ["rock", "indie"], "k": 20, "exclude_ids": [], "index": "genres"}
```

```json
{"ids": ["..."], "scores": [0.0], "index": "genres", "index_version": "..."}
```

- Ids are `m4a_id`. Words are matched exactly (lowercased, split on spaces); unknown words are ignored.
- `k` 1..1000, `words` up to 256, `exclude_ids` up to 100 000.
- Errors: 404 unknown index, 422 invalid body, 401 bad key (only if `BM25_API_KEY` is set).

## Env

| Variable | Meaning |
|---|---|
| `BM25_INDEXES_DIR` | Folder with indexes. |
| `BM25_API_KEY` | If set, requests need header `X-API-Key`. Empty = no auth. |
