"""Build a BM25 index from tracks_meta (.jsonl, .parquet, or a folder of tracks_meta-*.parquet shards).

    python build_index.py --input ../data/tracks_meta.jsonl --field m4a_genres_full --out indexes/genres
    python build_index.py --input ../data/tracks_meta.jsonl --field m4a_tags_full --out indexes/tags
    python build_index.py --input ../data/tracks_meta.jsonl --field m4a_artist m4a_song --out indexes/title

The index is named after its output folder; the service default is 'genres'.
A track's terms are the words of the chosen field(s) (see bm25.py). Rows
without an id are dropped, repeated ids keep their first row.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bm25 import save_bm25, train_bm25

ID_COL = 'm4a_id'


def _terms(cell):
    if isinstance(cell, (list, tuple, np.ndarray)):
        return [str(x) for x in cell]
    return [] if pd.isna(cell) else str(cell).split(',')


def _read(path, keep):
    """Yield DataFrames with the `keep` columns, one per file."""
    path = Path(path)
    files = sorted(path.glob('tracks_meta-*.parquet')) if path.is_dir() else [path]
    if not files:
        raise ValueError(f'No tracks_meta-*.parquet files in {path}')
    for f in files:
        if f.suffix == '.parquet':
            import pyarrow.parquet as pq
            names = pq.read_schema(f).names
        else:
            with open(f, encoding='utf-8') as fh:
                names = list(json.loads(next(l for l in fh if l.strip())))
        missing = [c for c in keep if c not in names]
        if missing:
            raise ValueError(f'{f} lacks columns {missing}')
        if f.suffix == '.parquet':
            yield pd.read_parquet(f, columns=keep)
        else:
            with open(f, encoding='utf-8') as fh:
                records = [json.loads(l) for l in fh if l.strip()]
            yield pd.DataFrame([{c: r.get(c) for c in keep} for r in records],
                               columns=keep)


def load_items(path, fields):
    """Read tracks_meta into [m4a_id, terms]; returns (items, stats)."""
    keep = [ID_COL, *fields]
    raw = pd.concat(_read(path, keep), ignore_index=True)
    items = pd.DataFrame({
        ID_COL: raw[ID_COL],
        'terms': [[t for cell in row for t in _terms(cell)]
                  for row in raw[fields].itertuples(index=False, name=None)]})

    stats = {'n_rows': len(items)}
    null_ids = items[ID_COL].isna()
    stats['n_dropped_null_ids'] = int(null_ids.sum())
    items = items[~null_ids]
    stats['n_dropped_duplicate_ids'] = int(items[ID_COL].duplicated().sum())
    items = items.drop_duplicates(ID_COL).reset_index(drop=True)
    items[ID_COL] = items[ID_COL].astype(str)
    stats['n_items'] = len(items)
    stats['n_empty_docs'] = int(sum(not any(t.strip() for t in ts)
                                    for ts in items['terms']))
    return items, stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--input', required=True,
                        help='tracks_meta.jsonl / .parquet / folder of shards')
    parser.add_argument('--field', required=True, nargs='+',
                        help='column(s) to index, e.g. m4a_genres_full')
    parser.add_argument('--out', required=True, help='output index directory')
    parser.add_argument('--k1', type=float, default=1.2)
    parser.add_argument('--b', type=float, default=0.75)
    args = parser.parse_args(argv)

    feature_col = '+'.join(args.field)
    items, stats = load_items(args.input, args.field)
    model = train_bm25(items.rename(columns={'terms': feature_col}), k1=args.k1,
                       b=args.b, feature_col=feature_col, id_col=ID_COL)
    stats['vocab_size'] = len(model['vocab'])
    info = {'input_file': args.input, 'fields': args.field,
            'built_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            **stats}
    version = save_bm25(model, args.out, info=info)

    for key, value in info.items():
        print(f'{key}: {value}')
    print(f'index_version: {version}')
    print(f'saved to {args.out}')


if __name__ == '__main__':
    main()
