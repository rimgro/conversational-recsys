import pandas as pd
import pytest
from fastapi.testclient import TestClient


KEY = 'secret'


@pytest.fixture
def client(items, tmp_path, monkeypatch):
    import bm25s
    import json
    import numpy as np
    path = tmp_path / 'cards'
    bm = bm25s.BM25(method='lucene')
    bm.index(bm25s.tokenize(['pop dance pop', 'rock hard rock', '', '',
                           'r&b k-pop', 'alternative rock indie pop'],
                          stopwords='en', show_progress=False), show_progress=False)
    bm.save(str(path), show_progress=False)
    pd.DataFrame({'item_col': np.arange(6),
                  'm4a_id': list('abcdef')}).to_parquet(path / 'items.parquet')
    (path / 'meta.json').write_text(json.dumps({'format': 'bm25s-cards-v1',
        'library_version': '0.3.13', 'index_version': 'testversion1'}))
    (path / 'tokenization.json').write_text(json.dumps(
        {'stopwords': 'en', 'stemming': False}))
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
    assert data['index'] == 'cards'


def test_select_index(client):
    body = {'words': ['rock'], 'k': 5}
    default = search(client, body).json()
    assert default['index'] == 'cards'
    for name in ['cards', 'genres', 'tags']:
        data = search(client, dict(body, index=name)).json()
        assert data['ids'] == default['ids']
        assert data['scores'] == default['scores']
        assert data['index'] == name
    assert search(client, dict(body, index='nope')).status_code == 404


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
    assert data['status'] == 'ok' and data['default_index'] == 'cards'
    assert set(data['indexes']) == {'cards'}
    assert data['indexes']['cards']['n_items'] == 6

@pytest.mark.parametrize('words', [[], ['the', 'and'], ['zzzzzzunknown']])
def test_no_matching_tokens(client, words):
    data = search(client, {'words': words, 'k': 200}).json()
    assert data['ids'] == data['scores'] == []


def test_all_tracks_excluded(client):
    data = search(client, {'words': ['rock'], 'k': 200,
                          'exclude_ids': list('abcdef')}).json()
    assert data['ids'] == data['scores'] == []


def test_scores_match_bm25s(client):
    import bm25s
    import numpy as np
    model = client.app.state.models['cards']
    words = ['rock', 'rock', 'indie']
    tokens = bm25s.tokenize([' '.join(words)], stopwords='en',
                           return_ids=False, show_progress=False)[0]
    scores = model['bm'].get_scores(tokens)
    positions = np.flatnonzero(scores > 0)
    positions = positions[np.lexsort((positions, -scores[positions]))]
    actual = search(client, {'words': words, 'k': 200}).json()
    assert actual['ids'] == model['item_ids'][positions].tolist()
    np.testing.assert_array_equal(actual['scores'], scores[positions])
