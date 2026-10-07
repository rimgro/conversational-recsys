import pytest
from fastapi.testclient import TestClient

from hnsw_index import save_hnsw, train_hnsw

KEY = 'secret'


@pytest.fixture
def client(items, tmp_path, monkeypatch):
    save_hnsw(train_hnsw(items), tmp_path / 'audio')
    (tmp_path / 'not-an-index').mkdir()
    monkeypatch.setenv('HNSW_INDEXES_DIR', str(tmp_path))
    monkeypatch.setenv('HNSW_API_KEY', KEY)
    from service import app
    with TestClient(app) as c:
        yield c


def post(client, body, key=KEY):
    return client.post('/hnsw/search', json=body, headers={'X-API-Key': key} if key else {})


def test_search(client):
    r = post(client, {'track_ids': ['a', 'zzz'], 'k': 2, 'exclude_ids': ['a']})
    assert r.status_code == 200
    data = r.json()
    assert data['ids'] == ['b', 'c'] and data['index'] == 'audio' and data['n_query_tracks'] == 1
    assert len(data['index_version']) == 12


def test_vector_query(client):
    assert post(client, {'vector': [0, 0, 1], 'k': 1}).json()['ids'] == ['c']


def test_errors(client):
    assert post(client, {'track_ids': ['a'], 'k': 1}, key='wrong').status_code == 401
    assert post(client, {'track_ids': ['a'], 'k': 1, 'index': 'nope'}).status_code == 404
    assert post(client, {'k': 1}).status_code == 422                                    # нет запроса
    assert post(client, {'vector': [1, 0, 0], 'track_ids': ['a'], 'k': 1}).status_code == 422
    assert post(client, {'vector': [1, 0], 'k': 1}).status_code == 422                  # не та размерность
    assert post(client, {'track_ids': ['a'], 'weights': [1, 2], 'k': 1}).status_code == 422


def test_health(client):
    h = client.get('/hnsw/health').json()
    assert h['status'] == 'ok' and h['indexes']['audio']['n_items'] == 3 and h['indexes']['audio']['dim'] == 3
