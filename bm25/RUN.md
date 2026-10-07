# BM25 service

HTTP API for lexical candidate search over `tracks_meta`. Python 3.10+.
On the VPS it runs as its own service next to HNSW: `deploy/README.md`.

## Build indexes

```bash
pip install -r requirements.txt
sh ../deploy/build_indexes.sh bm25 ../data/tracks_meta.parquet ../indexes   # genres, tags, title
```

Each index is a folder in `indexes/`; its name is the `index` value in requests.
The `genres` index is required (it is the default). Other indexes are picked up automatically.
Add one with `build_index.py --field <column> ... --out indexes/<name>` (see its docstring).

## Run

```bash
BM25_INDEXES_DIR=../indexes/bm25 uvicorn app:app --port 8001
```

- API page: http://localhost:8001/docs
- Health and loaded indexes: http://localhost:8001/health

Indexes are read once at startup: after rebuilding, restart the service (`systemctl restart recsys-bm25` on the VPS).

## API

`POST /bm25/search`

```json
{"words": ["rock", "indie"], "k": 20, "exclude_ids": [], "index": "genres"}
```

```json
{"ids": ["..."], "scores": [0.0], "index": "genres", "index_version": "..."}
```

- Ids are `m4a_id`. Words are matched exactly (lowercased, split on spaces; when indexing, `tracks_meta` strings are first split on commas); unknown words are ignored.
- `k` 1..1000, `words` up to 256, `exclude_ids` up to 100 000.
- Errors: 404 unknown index, 422 invalid body, 401 bad key (only if `BM25_API_KEY` is set).

## Env

| Variable | Meaning |
|---|---|
| `BM25_INDEXES_DIR` | Folder with indexes (`/opt/recsys/indexes/bm25` on the VPS). |
| `BM25_API_KEY` | If set, requests need header `X-API-Key`. Empty = no auth. |
