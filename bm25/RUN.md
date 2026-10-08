# Music4All card BM25 service

This service loads one already trained `bm25s==0.3.13` model. It does not
retrain the notebook index. `genres` and `tags` are request aliases for
`cards`: all three search the complete card and return the same results.
An omitted or null `index` selects `cards`. Responses preserve an explicitly
requested alias in `index`; `index_version` identifies the actual card model.
`/health` reports only the single physical model, `cards`.

## Prepare artifacts locally

```bash
pip install -r requirements.txt
python package_cards.py --source /path/to/bm25_index --output /path/to/bm25-indexes-cards.zip
```

The archive contains `indexes/cards/`. It includes the original index arrays,
vocabulary, parameters, tokenization settings and `items.parquet`, plus a
manifest with SHA256 checksums and a deterministic `index_version`.

## Deploy

Stop the existing service using its existing service manager. Back up its code
and the whole existing indexes directory before replacing anything. Install the
updated `app.py`, `cards_model.py` and `requirements.txt` in the server's `bm25`
directory and run `pip install -r requirements.txt` in its service environment.
Extract the archive into a staging directory and verify the files against the
SHA256 values in `indexes/cards/meta.json`. Replace the existing indexes directory
with the staged `indexes/` directory (keep the backup for rollback).

Keep the existing `BM25_API_KEY` environment setting. Keep:

```bash
export BM25_INDEXES_DIR=/home/pasha_bel/apps/conversational-recsys-v1.1/bm25/indexes
```

Restart with the existing service manager. For a manual launch from `bm25/`:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Indexes load once at startup. Verify `/health`: default `cards`, 64016 items,
and the packaged `index_version`. Rollback requires restoring BOTH the previous
code and indexes, then restarting. No upload or reload endpoint is provided.

## API

`POST /bm25/search` with header `X-API-Key`:

```json
{"words": ["upbeat", "dreamy", "indie", "rock"], "k": 200, "exclude_ids": []}
```

Response fields: `ids` (Music4All `m4a_id`), `scores`, `index`, `index_version`.
`ids[i]` and `scores[i]` refer to the same track. Queries use the notebook's
bm25s tokenizer, English stopwords and no stemming. Repeated terms contribute
repeatedly, just as in the notebook. Scores are not normalized or rescaled.
Exclusions apply before top-k. Only positive-scoring matches are returned:
there is no zero-score backfill. Equal scores use document order, which in the
supplied model is sorted `m4a_id` order.

Limits remain: `k` 1..1000, up to 256 word/phrase strings and 100000 excluded IDs.
Unknown index names return 404, invalid requests 422, invalid API keys 401.
Old `build_index.py` and `bm25.py` remain offline legacy utilities; the service
does not import them or load their models.

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests -q
```
