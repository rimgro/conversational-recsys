"""Build a BM25 index from Music4All-Onion TSV files (zenodo.org/records/15394646).

    python build_index.py --source onion-genres \
        --input ../data/id_genres_tf-idf.tsv.bz2 --out indexes/genres
    python build_index.py --source onion-tags \
        --input ../data/id_tags_dict.tsv.bz2 --out indexes/tags

Ids are Music4All `id`. Every source becomes a table [id, list of terms]:
rows without an id are dropped, duplicate ids are merged by uniting their
terms, tracks without terms stay as empty documents. Then the index is trained
and saved with build stats in meta.json.
"""
import argparse
import ast
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from bm25 import save_bm25, train_bm25


def _union(lists):
    """Concatenate lists, dropping repeats, keeping first-seen order."""
    return list(dict.fromkeys(g for genres in lists for g in genres))


def _require(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f'{name} lacks columns {missing}; has {list(df.columns)}')


def _finalize(items, feature_col, stats):
    """Drop null ids, merge duplicate ids, count stats."""
    null_ids = items['id'].isna()
    stats['n_dropped_null_ids'] = int(null_ids.sum())
    items = items[~null_ids].copy()
    items['id'] = items['id'].astype(str)
    n_rows = len(items)
    items = (items.groupby('id', sort=False)[feature_col]
             .agg(_union).reset_index())
    stats['n_merged_duplicates'] = n_rows - len(items)
    stats['n_items'] = len(items)
    stats['n_empty_docs'] = int(sum(len(g) == 0 for g in items[feature_col]))
    return items, stats


def items_from_genre_matrix(matrix: pd.DataFrame, feature_col: str = 'genres'):
    """Onion id_genres_tf-idf: wide [id, <genre>...] -> genres with weight > 0."""
    _require(matrix, ['id'], 'genre matrix')
    genre_names = np.array([c for c in matrix.columns if c != 'id'])
    values = matrix[genre_names].to_numpy(dtype=float)
    genres = [list(genre_names[np.flatnonzero(row > 0)]) for row in values]
    items = pd.DataFrame({'id': matrix['id'], feature_col: genres})
    return _finalize(items, feature_col, {})


def items_from_tag_dicts(tags: pd.DataFrame, feature_col: str = 'tags',
                         min_weight: float = 0):
    """Onion id_tags_dict: [id, "{'tag': weight}"] -> tags with weight >= min_weight."""
    _require(tags, ['id'], 'tag file')
    if len(tags.columns) != 2:
        raise ValueError(f'tag file must have 2 columns; has {list(tags.columns)}')
    parsed = []
    for row, cell in enumerate(tags[tags.columns[1]]):
        try:
            d = ast.literal_eval(cell) if isinstance(cell, str) else {}
        except (SyntaxError, ValueError):
            raise ValueError(f'Invalid tag dict at row {row}') from None
        if not isinstance(d, dict):
            raise ValueError(f'Invalid tag dict at row {row}')
        parsed.append([t for t, w in d.items() if w >= min_weight])
    items = pd.DataFrame({'id': tags['id'], feature_col: parsed})
    return _finalize(items, feature_col, {'min_weight': min_weight})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--source', required=True,
                        choices=['onion-genres', 'onion-tags'])
    parser.add_argument('--input', required=True, help='the .tsv(.bz2) file')
    parser.add_argument('--out', required=True, help='output index directory')
    # onion-tags only: keep tags with weight >= this (see warning.md).
    parser.add_argument('--min-weight', type=float, default=0,
                        help=argparse.SUPPRESS)
    parser.add_argument('--k1', type=float, default=1.2)
    parser.add_argument('--b', type=float, default=0.75)
    args = parser.parse_args(argv)

    if args.source == 'onion-genres':
        feature_col = 'genres'
        items, stats = items_from_genre_matrix(
            pd.read_csv(args.input, sep='\t', dtype={'id': str}), feature_col)
    else:
        feature_col = 'tags'
        items, stats = items_from_tag_dicts(
            pd.read_csv(args.input, sep='\t', dtype=str), feature_col,
            args.min_weight)

    model = train_bm25(items, k1=args.k1, b=args.b,
                       feature_col=feature_col, id_col='id')
    stats['vocab_size'] = len(model['vocab'])
    info = {'source': args.source, 'input_file': args.input,
            'built_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            **stats}
    version = save_bm25(model, args.out, info=info)

    for key, value in info.items():
        print(f'{key}: {value}')
    df = np.diff(model['index'].indptr)
    terms = np.array(list(model['vocab']))  # vocab values are 0..n-1 in order
    top = np.argsort(-df, kind='stable')[:20]
    print('top terms by df:', ', '.join(f'{terms[i]}={df[i]}' for i in top))
    print(f'index_version: {version}')
    print(f'saved to {args.out}')


if __name__ == '__main__':
    main()
