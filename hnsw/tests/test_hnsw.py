import numpy as np
import pytest

from build_hnsw import main as build_main
from hnsw_index import infer_hnsw, load_hnsw, save_hnsw, train_hnsw


def test_train_drops_missing_and_normalizes(items):
    m = train_hnsw(items)
    assert list(m['item_ids']) == ['a', 'b', 'c']
    assert np.allclose(np.linalg.norm(m['vectors'], axis=1), 1.0)


def test_infer_by_tracks_and_vector(items):
    m = train_hnsw(items)
    assert list(infer_hnsw(m, track_ids=['a'], top_k=3)) == ['a', 'b', 'c']
    assert list(infer_hnsw(m, track_ids=['a'], top_k=2, exclude_ids=['a'])) == ['b', 'c']
    assert list(infer_hnsw(m, vector=[0, 0, 1], top_k=1)) == ['c']
    # веса: c с большим весом перетягивает центр
    assert list(infer_hnsw(m, track_ids=['a', 'c'], weights=[1, 5], top_k=1)) == ['c']
    assert infer_hnsw(m, track_ids=['unknown'], top_k=3) == {}
    with pytest.raises(ValueError):
        infer_hnsw(m, vector=[1, 0], top_k=1)


def test_save_load_roundtrip(items, tmp_path):
    m = train_hnsw(items)
    version = save_hnsw(m, tmp_path / 'audio', info={'x': 1})
    loaded = load_hnsw(tmp_path / 'audio')
    assert loaded['index_version'] == version and loaded['info'] == {'x': 1}
    assert infer_hnsw(loaded, track_ids=['b'], top_k=3) == infer_hnsw(m, track_ids=['b'], top_k=3)


def test_build_script(items, tmp_path):
    src = tmp_path / 'tracks_meta.parquet'
    items.assign(muq_embedding=[None if v is None else list(v) for v in items['muq_embedding']]).to_parquet(src)
    build_main(['--input', str(src), '--out', str(tmp_path / 'idx' / 'audio')])
    assert list(load_hnsw(tmp_path / 'idx' / 'audio')['item_ids']) == ['a', 'b', 'c']
