"""Read-only inference for the Music4All bm25s card index."""
import json
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd


def load_cards(path):
    path = Path(path)
    meta = json.loads((path / 'meta.json').read_text())
    if meta.get('format') != 'bm25s-cards-v1':
        raise ValueError('Expected a bm25s-cards-v1 index')
    if bm25s.__version__ != meta['library_version']:
        raise ValueError('bm25s version does not match the saved index')
    tokenization = json.loads((path / 'tokenization.json').read_text())
    if tokenization.get('stopwords') != 'en' or tokenization.get('stemming') is not False:
        raise ValueError('Expected English stopwords and no stemming')
    bm = bm25s.BM25.load(str(path), mmap=True)
    items = pd.read_parquet(path / 'items.parquet').sort_values('item_col')
    if not np.array_equal(items.item_col.to_numpy(), np.arange(bm.scores['num_docs'])):
        raise ValueError('item_col must cover every document position exactly once')
    if items.m4a_id.isna().any() or items.m4a_id.duplicated().any():
        raise ValueError('Expected unique non-null m4a_id values')
    return {'bm': bm, 'item_ids': items.m4a_id.astype(str).to_numpy(),
            'vocab': bm.vocab_dict, 'id_col': 'm4a_id', 'feature_col': 'card',
            'index_version': meta['index_version']}


def infer_cards(model, words, top_k, exclude_ids):
    # Keep repeated terms: this is exactly the notebook query processing.
    tokens = bm25s.tokenize([' '.join(words)], stopwords='en',
                           return_ids=False, show_progress=False)[0]
    if not tokens:
        return {}
    scores = model['bm'].get_scores(tokens)
    candidates = np.flatnonzero(scores > 0)
    if exclude_ids:
        candidates = candidates[~np.isin(model['item_ids'][candidates], exclude_ids)]
    order = np.lexsort((candidates, -scores[candidates]))[:top_k]
    return {str(model['item_ids'][i]): float(scores[i]) for i in candidates[order]}
