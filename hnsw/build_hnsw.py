"""Build a vector index from tracks_meta (.parquet or a folder of tracks_meta-*.parquet shards).

    python build_hnsw.py --input ../data/tracks_meta.parquet --field muq_embedding --out ../indexes/hnsw/audio

The index is named after its output folder; the service default is 'audio'.
Rows without an id or a vector are dropped, repeated ids keep their first row.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from hnsw_index import save_hnsw, train_hnsw

ID_COL = 'm4a_id'


def load_items(path, field):
    path = Path(path)
    files = sorted(path.glob('tracks_meta-*.parquet')) if path.is_dir() else [path]
    if not files:
        raise ValueError(f'No tracks_meta-*.parquet files in {path}')
    raw = pd.concat([pd.read_parquet(f, columns=[ID_COL, field]) for f in files], ignore_index=True)
    stats = {'n_rows': len(raw), 'n_dropped_null_ids': int(raw[ID_COL].isna().sum())}
    raw = raw[raw[ID_COL].notna()]
    stats['n_dropped_duplicate_ids'] = int(raw[ID_COL].duplicated().sum())
    raw = raw.drop_duplicates(ID_COL).reset_index(drop=True)
    has_vec = raw[field].map(lambda v: v is not None and len(v) > 0)
    stats['n_dropped_no_vector'] = int((~has_vec).sum())
    return raw[has_vec].reset_index(drop=True), stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--input', required=True, help='tracks_meta.parquet or folder of shards')
    parser.add_argument('--field', default='muq_embedding', help='column with vectors')
    parser.add_argument('--out', required=True, help='output index directory')
    args = parser.parse_args(argv)

    items, stats = load_items(args.input, args.field)
    model = train_hnsw(items, vector_col=args.field, id_col=ID_COL)
    stats.update(n_items=len(model['item_ids']), dim=int(model['vectors'].shape[1]))
    info = {'input_file': args.input, 'field': args.field,
            'built_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), **stats}
    version = save_hnsw(model, args.out, info=info)
    for key, value in info.items():
        print(f'{key}: {value}')
    print(f'index_version: {version}')
    print(f'saved to {args.out}')


if __name__ == '__main__':
    main()
