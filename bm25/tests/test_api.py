import pandas as pd
import pytest
from fastapi.testclient import TestClient

from bm25 import save_bm25, train_bm25

KEY = 'secret'


@pytest.fixture
def client(items, tmp_path, monkeypatch):
    save_bm25(train_bm25(items), tmp_path / 'genres')
    tags = pd.DataFrame({'spotify_id': ['t1', 't2'], 'genres': ['chill', 'sad']})
    save_bm25(train_bm25(tags), tmp_path / 'tags')
    (tmp_path / 'not-an-index').mkdir()
    monkeypatch.setenv('BM25_INDEXES_DIR', str(tmp_path))
    monkeypatch.setenv('BM25_API_KEY', KEY)
    from app import app
    with TestClient(app) as c:
        yield c


def search(client, body, key=KEY):
    headers = {'X-API-Key': key} if key is not None else {}
    return client.post('/bm25/search', json=body, headers=headers)


def test_search(client):
    r = search(client, {'words': ['pop', 'indie'], 'k': 2})
    assert r.status_code == 200
    data = r.json()
    assert len(data['ids']) == 2 == len(data['scores'])
    assert data['scores'] == sorted(data['scores'], reverse=True)
    assert len(data['index_version']) == 12
    assert data['index'] == 'genres'


def test_select_index(client):
    r = search(client, {'words': ['chill'], 'k': 5, 'index': 'tags'}).json()
    assert r['ids'] == ['t1'] and r['index'] == 'tags'
    assert search(client, {'words': ['chill'], 'k': 5}).json()['ids'] == []
    r = search(client, {'words': ['chill'], 'k': 5, 'index': 'nope'})
    assert r.status_code == 404


def test_exclude_ids(client):
    first = search(client, {'words': ['pop'], 'k': 1}).json()['ids']
    r = search(client, {'words': ['pop'], 'k': 1, 'exclude_ids': first}).json()
    assert len(r['ids']) == 1 and r['ids'] != first


def test_exclude_ids_limit(client):
    ok = {'words': ['pop'], 'k': 1, 'exclude_ids': [f'x{i}' for i in range(100_000)]}
    assert search(client, ok).status_code == 200
    ok['exclude_ids'].append('x')
    assert search(client, ok).status_code == 422


def test_empty_result(client):
    r = search(client, {'words': ['nothing'], 'k': 5})
    assert r.status_code == 200
    assert r.json()['ids'] == [] and r.json()['scores'] == []


@pytest.mark.parametrize('key', [None, 'wrong'])
def test_auth(client, key):
    assert search(client, {'words': ['pop'], 'k': 1}, key=key).status_code == 401


@pytest.mark.parametrize('body', [
    {'words': ['pop'], 'k': 0},
    {'words': ['pop'], 'k': 100_000},
    {'words': ['pop'], 'k': 1, 'typo': 1},
    {'words': 'pop', 'k': 1},
    {'k': 1},
])
def test_validation(client, body):
    assert search(client, body).status_code == 422


def test_health_without_key(client):
    r = client.get('/health')
    assert r.status_code == 200
    data = r.json()
    assert data['status'] == 'ok' and data['default_index'] == 'genres'
    assert set(data['indexes']) == {'genres', 'tags'}
    assert data['indexes']['genres']['n_items'] == 6
