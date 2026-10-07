"""Vector (embedding) index for the HNSW candidate generator.

Dependencies: numpy, pandas.

Vectors are L2-normalized at build time, scores are cosine similarities.
Search is exact (brute force): 64k x 128 float32 takes ~1-2 ms per query, so
no approximate index is needed yet. To switch to hnswlib/faiss keep the same
functions (train/infer/save/load) and the service API does not change.

A query is either a vector or a list of item ids with optional weights: the
weighted mean of their vectors is used (e.g. the user's listening history).

Typical use:
    model = train_hnsw(items, vector_col='muq_embedding', id_col='m4a_id')
    save_hnsw(model, 'indexes/hnsw/audio')         # once, offline
    model = load_hnsw('indexes/hnsw/audio')        # at service start
    infer_hnsw(model, track_ids=['a', 'b'], weights=[2, 1], top_k=20, exclude_ids={'a'})
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FORMAT_VERSION = 1

__all__ = ['train_hnsw', 'infer_hnsw', 'query_vector', 'save_hnsw', 'load_hnsw']


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return np.divide(m, norms, out=np.zeros_like(m), where=norms > 0)


def train_hnsw(items: pd.DataFrame, vector_col: str = 'muq_embedding', id_col: str = 'm4a_id') -> dict:
    """Stack the vectors of `vector_col`; rows without a vector are dropped."""
    if not {id_col, vector_col}.issubset(items.columns):
        raise ValueError(f'Expected {id_col} and {vector_col} columns')
    ok = items[vector_col].map(lambda v: v is not None and len(v) > 0)
    items = items[ok]
    ids = items[id_col]
    if items.empty or ids.isna().any() or ids.duplicated().any():
        raise ValueError(f'Expected nonempty data with unique, non-null {id_col}')
    vectors = np.stack([np.asarray(v, dtype=np.float32) for v in items[vector_col]])
    if not np.isfinite(vectors).all():
        raise ValueError('Vectors contain NaN or inf')
    return {'vectors': _normalize(vectors), 'item_ids': ids.astype(str).to_numpy(copy=True),
            'vector_col': vector_col, 'id_col': id_col}


def _positions(model: dict) -> dict:
    if '_pos' not in model:
        model['_pos'] = {tid: i for i, tid in enumerate(model['item_ids'])}
    return model['_pos']


def query_vector(model: dict, vector=None, track_ids=None, weights=None):
    """Normalized query vector, or None if no known track / zero vector."""
    if vector is not None:
        v = np.asarray(vector, dtype=np.float32)
        if v.shape != (model['vectors'].shape[1],):
            raise ValueError(f"vector must have dim {model['vectors'].shape[1]}")
    else:
        track_ids = list(track_ids or [])
        weights = [1.0] * len(track_ids) if weights is None else list(weights)
        if len(weights) != len(track_ids):
            raise ValueError('weights and track_ids must have the same length')
        pos = _positions(model)
        known = [(pos[str(t)], float(w)) for t, w in zip(track_ids, weights) if str(t) in pos]
        if not known:
            return None
        idx = np.array([p for p, _ in known])
        w = np.array([x for _, x in known], dtype=np.float32)
        v = (model['vectors'][idx] * w[:, None]).sum(axis=0)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 and np.isfinite(n) else None


def infer_hnsw(model: dict, vector=None, track_ids=None, weights=None, top_k: int = 20,
               exclude_ids=None) -> dict:
    """Return {item_id: cosine} in descending order; exclude_ids are removed before the top-k cut."""
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 0:
        raise ValueError('top_k must be a nonnegative integer')
    v = query_vector(model, vector, track_ids, weights)
    if v is None or top_k == 0:
        return {}
    with np.errstate(all='ignore'):  # numpy 2.0 + Accelerate (macOS) gives spurious matmul warnings
        scores = model['vectors'] @ v
    candidates = np.arange(len(scores))
    if exclude_ids:
        if isinstance(exclude_ids, str):
            exclude_ids = [exclude_ids]
        banned = np.isin(model['item_ids'], [str(i) for i in exclude_ids])
        candidates = candidates[~banned]
    order = np.lexsort((candidates, -scores[candidates]))[:top_k]
    return {model['item_ids'][i]: float(scores[i]) for i in candidates[order]}


def _index_version(vectors, item_ids, meta) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(vectors).tobytes())
    h.update('\n'.join(map(str, item_ids)).encode('utf-8'))
    h.update(json.dumps(meta, sort_keys=True).encode('utf-8'))
    return h.hexdigest()[:12]


def save_hnsw(model: dict, path, info: dict = None) -> str:
    """Save to directory `path` (vectors.npy, item_ids.npy, meta.json); returns index_version."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    item_ids = np.asarray(model['item_ids'], dtype=str)
    np.save(path / 'vectors.npy', model['vectors'].astype(np.float32), allow_pickle=False)
    np.save(path / 'item_ids.npy', item_ids, allow_pickle=False)
    meta = {'format_version': FORMAT_VERSION, 'kind': 'hnsw', 'dim': int(model['vectors'].shape[1]),
            'vector_col': model['vector_col'], 'id_col': model['id_col'], 'metric': 'cosine'}
    meta['index_version'] = _index_version(model['vectors'], item_ids, meta)
    meta['info'] = info or {}
    (path / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
    return meta['index_version']


def load_hnsw(path) -> dict:
    path = Path(path)
    meta = json.loads((path / 'meta.json').read_text(encoding='utf-8'))
    if meta.get('format_version') != FORMAT_VERSION or meta.get('kind') != 'hnsw':
        raise ValueError(f'Not an hnsw index of format {FORMAT_VERSION}: {path}')
    vectors = np.load(path / 'vectors.npy', allow_pickle=False)
    item_ids = np.load(path / 'item_ids.npy', allow_pickle=False)
    if vectors.shape != (len(item_ids), meta['dim']):
        raise ValueError('Index files are inconsistent')
    return {'vectors': vectors, 'item_ids': item_ids, 'vector_col': meta['vector_col'],
            'id_col': meta['id_col'], 'index_version': meta['index_version'], 'info': meta.get('info', {})}
