"""BM25 query/document retrieval over an item id column and a text/genre column.

Dependencies: numpy, pandas, scipy.

Tokenization (same for index and query): every cell is a list of strings
(or a comma-separated string, or a string with a Python/JSON list). Each
string is casefolded and split on whitespace into WORDS, so the genre
'alternative rock' is indexed as two terms, 'alternative' and 'rock'; a query
'rock' therefore matches 'hard rock' and 'alternative rock' alike. Whole
multiword genres are NOT kept as single tokens. Null cells become empty
documents (they stay in the catalog but never match).

Scoring: BM25 with positive smoothed (Lucene-style) IDF and the (k1 + 1)
numerator factor. Each distinct query term contributes once, so repeated
words in a query do not change the result. Scores are comparable only within
one index (same k1, b and corpus).

Ids come from `id_col` (default 'spotify_id'). save_bm25 can store an extra
`info` dict (build stats) and always stores `index_version`, a short hash of
the index contents; load_bm25 returns both.

Typical use:
    model = train_bm25(items, feature_col='genres')
    save_bm25(model, 'indexes/genres')          # once, offline
    model = load_bm25('indexes/genres')         # at service start
    infer_bm25(model, ['pop', 'chill'], top_k=20, exclude_ids={'abc'})
"""
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, load_npz, save_npz

FORMAT_VERSION = 1

__all__ = ['train_bm25', 'infer_bm25', 'save_bm25', 'load_bm25']


def _cell_to_list(cell, row):
    if isinstance(cell, str):
        cell = cell.strip()
        if cell.startswith('['):
            try:
                cell = ast.literal_eval(cell)
            except (SyntaxError, ValueError):
                raise ValueError(f'Invalid genres at row {row}') from None
        else:
            cell = cell.split(',')
    elif cell is None or cell is pd.NA or (np.isscalar(cell) and pd.isna(cell)):
        cell = []
    if not isinstance(cell, (list, tuple, set, np.ndarray)):
        raise ValueError(f'Invalid genres at row {row}')
    if any(not isinstance(g, str) for g in cell):
        raise ValueError(f'Genre names must be strings at row {row}')
    return cell


def _words(strings):
    return [word for s in strings for word in s.strip().casefold().split()]


def train_bm25(items: pd.DataFrame, k1: float = 1.2, b: float = 0.75,
               feature_col: str = 'genres', id_col: str = 'spotify_id') -> dict:
    """Build a BM25 index from id_col and the selected feature column."""
    if not np.isfinite(k1) or k1 <= 0 or not np.isfinite(b) or not 0 <= b <= 1:
        raise ValueError('Require finite k1 > 0 and 0 <= b <= 1')
    if not {id_col, feature_col}.issubset(items.columns):
        raise ValueError(f'Expected {id_col} and {feature_col} columns')
    ids = items[id_col]
    if items.empty or ids.isna().any() or ids.duplicated().any():
        raise ValueError(f'Expected nonempty data with unique, non-null {id_col}')
    vocab, rows, cols, frequencies = {}, [], [], []
    lengths = np.zeros(len(items), dtype=float)
    for row, cell in enumerate(items[feature_col]):
        terms = _words(_cell_to_list(cell, row))
        lengths[row] = len(terms)
        for term, tf in Counter(terms).items():
            column = vocab.setdefault(term, len(vocab))
            rows.append(row)
            cols.append(column)
            frequencies.append(tf)
    if not vocab:
        raise ValueError(f'No terms available to index in {feature_col}')
    counts = coo_matrix((np.asarray(frequencies, dtype=float), (rows, cols)),
                        shape=(len(items), len(vocab))).tocsc()
    df = np.diff(counts.indptr)
    idf = np.log1p((len(items) - df + 0.5) / (df + 0.5))
    weighted = counts.tocoo()
    length_norm = 1 - b + b * lengths / lengths.mean()
    weighted.data = (weighted.data * (k1 + 1) /
                     (weighted.data + k1 * length_norm[weighted.row]) *
                     idf[weighted.col])
    return {'index': weighted.tocsc(), 'vocab': vocab,
            'item_ids': ids.astype(str).to_numpy(copy=True),
            'k1': float(k1), 'b': float(b), 'feature_col': feature_col,
            'id_col': id_col}


def infer_bm25(model: dict, query, top_k: int = 20, exclude_ids=None) -> dict:
    """Return {item_id: score} in descending score order.

    query: one string or a list of strings (words or phrases; everything is
    split into words the same way as the index). Unknown words are ignored;
    no overlap returns {}. exclude_ids: ids (e.g. already played tracks) removed
    BEFORE the top-k cut, so up to top_k results are still returned. Ties are
    broken by catalog order, so results are deterministic.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 0:
        raise ValueError('top_k must be a nonnegative integer')
    if isinstance(query, str):
        query = [query]
    if not isinstance(query, (list, tuple, set, np.ndarray)):
        raise ValueError('query must be a string or a sequence of strings')
    if any(not isinstance(g, str) for g in query):
        raise ValueError('Query items must be strings')
    terms = set(_words(query))
    # Sorted so the float summation order does not depend on the hash seed.
    columns = sorted(model['vocab'][t] for t in terms if t in model['vocab'])
    if top_k == 0 or not columns:
        return {}
    # One BM25 contribution per distinct matching query term.
    scores = np.asarray(model['index'][:, columns].sum(axis=1)).ravel()
    candidates = np.flatnonzero(scores > 0)
    if exclude_ids is not None and len(candidates):
        if isinstance(exclude_ids, str):
            exclude_ids = [exclude_ids]
        banned = np.isin(model['item_ids'][candidates], [str(i) for i in exclude_ids])
        candidates = candidates[~banned]
    order = np.lexsort((candidates, -scores[candidates]))[:top_k]
    return {model['item_ids'][i]: float(scores[i]) for i in candidates[order]}


def _index_version(index, item_ids, meta) -> str:
    h = hashlib.sha256()
    index = index.tocsc()
    for arr in (index.data, index.indices, index.indptr, np.asarray(index.shape)):
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update('\n'.join(map(str, item_ids)).encode('utf-8'))
    h.update(json.dumps(meta, sort_keys=True, ensure_ascii=False).encode('utf-8'))
    return h.hexdigest()[:12]


def save_bm25(model: dict, path, info: dict = None) -> str:
    """Save the index to directory `path` (index.npz, item_ids.npy, meta.json).

    info: optional JSON-serializable dict (e.g. build stats) kept in meta.json.
    Returns index_version, a hash of the index contents and parameters.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    item_ids = np.asarray(model['item_ids'], dtype=str)
    save_npz(path / 'index.npz', model['index'])
    np.save(path / 'item_ids.npy', item_ids, allow_pickle=False)
    meta = {'format_version': FORMAT_VERSION, 'vocab': model['vocab'],
            'k1': model['k1'], 'b': model['b'], 'feature_col': model['feature_col'],
            'id_col': model.get('id_col', 'spotify_id')}
    meta['index_version'] = _index_version(model['index'], item_ids, meta)
    meta['info'] = info or {}
    (path / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False),
                                    encoding='utf-8')
    return meta['index_version']


def load_bm25(path) -> dict:
    """Load an index written by save_bm25."""
    path = Path(path)
    meta = json.loads((path / 'meta.json').read_text(encoding='utf-8'))
    if meta.get('format_version') != FORMAT_VERSION:
        raise ValueError(f"Unsupported index format: {meta.get('format_version')}")
    index = load_npz(path / 'index.npz').tocsc()
    item_ids = np.load(path / 'item_ids.npy', allow_pickle=False)
    if index.shape != (len(item_ids), len(meta['vocab'])):
        raise ValueError('Index files are inconsistent')
    return {'index': index, 'vocab': meta['vocab'], 'item_ids': item_ids,
            'k1': meta['k1'], 'b': meta['b'], 'feature_col': meta['feature_col'],
            'id_col': meta.get('id_col', 'spotify_id'),
            'index_version': meta.get('index_version', 'unknown'),
            'info': meta.get('info', {})}
