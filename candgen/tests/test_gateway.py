"""Один сервис отдаёт и BM25, и HNSW; индексы строятся теми же скриптами, что в образе."""
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
for d in ('candgen', 'bm25', 'hnsw'):
    sys.path.insert(0, str(ROOT / d))

KEY = 'secret'


@pytest.fixture
def client(tmp_path, monkeypatch):
    meta = pd.DataFrame({
        'm4a_id': ['a', 'b', 'c'],
        'm4a_genres_full': ['rock,hard rock', 'pop', 'jazz'],
        'm4a_tags_full': ['guitar,loud', 'catchy', 'smooth night'],
        'muq_embedding': [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
    })
    src = tmp_path / 'tracks_meta.parquet'
    meta.to_parquet(src)
    import build_index
    import build_hnsw
    build_index.main(['--input', str(src), '--field', 'm4a_genres_full', '--out', str(tmp_path / 'bm25' / 'genres')])
    build_index.main(['--input', str(src), '--field', 'm4a_tags_full', '--out', str(tmp_path / 'bm25' / 'tags')])
    build_hnsw.main(['--input', str(src), '--out', str(tmp_path / 'hnsw' / 'audio')])
    monkeypatch.setenv('BM25_INDEXES_DIR', str(tmp_path / 'bm25'))
    monkeypatch.setenv('HNSW_INDEXES_DIR', str(tmp_path / 'hnsw'))
    monkeypatch.setenv('BM25_API_KEY', KEY)
    monkeypatch.delenv('HNSW_API_KEY', raising=False)
    from main import app
    with TestClient(app) as c:
        yield c


def test_both_generators(client):
    h = {'X-API-Key': KEY}
    r = client.post('/bm25/search', json={'words': ['rock'], 'k': 5}, headers=h).json()
    assert r['ids'] == ['a'] and r['index'] == 'genres'
    r = client.post('/bm25/search', json={'words': ['night'], 'k': 5, 'index': 'tags'}, headers=h).json()
    assert r['ids'] == ['c'] and r['index'] == 'tags'
    r = client.post('/hnsw/search', json={'track_ids': ['a'], 'k': 2, 'exclude_ids': ['a']}, headers=h).json()
    assert r['ids'] == ['b', 'c']
    assert client.get('/health').json()['indexes'].keys() == {'genres', 'tags'}
    assert client.get('/hnsw/health').json()['indexes'].keys() == {'audio'}


def test_shared_key(client):
    # HNSW_API_KEY не задан -> используется BM25_API_KEY
    assert client.post('/hnsw/search', json={'track_ids': ['a'], 'k': 1}).status_code == 401
    assert client.post('/bm25/search', json={'words': ['rock'], 'k': 1}).status_code == 401


def test_one_openapi_with_all_routes(client):
    assert client.get('/docs').status_code == 200
    paths = set(client.get('/openapi.json').json()['paths'])
    assert {'/bm25/search', '/health', '/hnsw/search', '/hnsw/health'} <= paths


@pytest.mark.xfail(strict=True, reason='bm25/build_index.py не делит строку "a,b" по запятым: в индексе слово '
                                       '"guitar,loud", поэтому "loud" не находится. Снять xfail после исправления.')
def test_bm25_splits_comma_separated_tags(client):
    r = client.post('/bm25/search', json={'words': ['loud'], 'k': 5, 'index': 'tags'},
                    headers={'X-API-Key': KEY}).json()
    assert r['ids'] == ['a']
