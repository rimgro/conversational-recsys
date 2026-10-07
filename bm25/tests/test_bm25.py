import json

import pandas as pd
import pytest

from bm25 import infer_bm25, load_bm25, save_bm25, train_bm25


def test_word_tokenization(items):
    model = train_bm25(items)
    assert set(infer_bm25(model, 'rock')) == {'b', 'f'}
    assert list(infer_bm25(model, ['K-POP', 'r&b'])) == ['e']


def test_empty_docs_never_match(items):
    model = train_bm25(items)
    assert {'c', 'd'} <= set(model['item_ids'])
    found = infer_bm25(model, ['pop', 'rock', 'indie', 'dance'], top_k=100)
    assert not {'c', 'd'} & set(found)


def test_scores_descending_and_repeated_terms_count_once(items):
    model = train_bm25(items)
    once = infer_bm25(model, ['pop', 'indie'])
    assert list(once.values()) == sorted(once.values(), reverse=True)
    assert infer_bm25(model, ['pop pop', 'indie', 'POP']) == once


def test_exclude_before_cut(items):
    model = train_bm25(items)
    top = list(infer_bm25(model, 'pop', top_k=1))
    rest = infer_bm25(model, 'pop', top_k=1, exclude_ids=top)
    assert len(rest) == 1 and top[0] not in rest
    for empty in ([], set(), None):
        assert list(infer_bm25(model, 'pop', exclude_ids=empty))[0] == top[0]
    assert top[0] not in infer_bm25(model, 'pop', exclude_ids=top[0])


def test_empty_results(items):
    model = train_bm25(items)
    assert infer_bm25(model, 'unknownword') == {}
    assert infer_bm25(model, []) == {}
    assert infer_bm25(model, 'pop', top_k=0) == {}
    with pytest.raises(ValueError):
        infer_bm25(model, 'pop', top_k=-1)


def test_ties_broken_by_catalog_order():
    items = pd.DataFrame({'spotify_id': ['z', 'y', 'x'],
                          'genres': ['jazz', 'jazz', 'jazz']})
    assert list(infer_bm25(train_bm25(items), 'jazz')) == ['z', 'y', 'x']


def test_id_col():
    items = pd.DataFrame({'track_id': [10, 20], 'genres': ['pop', 'rock']})
    model = train_bm25(items, id_col='track_id')
    assert list(infer_bm25(model, 'rock')) == ['20']
    with pytest.raises(ValueError, match='spotify_id'):
        train_bm25(items)


@pytest.mark.parametrize('bad', ['[pop', "['pop', 1]", '[[]]'])
def test_invalid_cells(bad):
    with pytest.raises(ValueError, match='row 1'):
        train_bm25(pd.DataFrame({'spotify_id': ['a', 'b'], 'genres': ['pop', bad]}))


def test_save_load_roundtrip(items, tmp_path):
    model = train_bm25(items, k1=1.5, b=0.5)
    version = save_bm25(model, tmp_path, info={'n_items': 6})
    loaded = load_bm25(tmp_path)
    assert loaded['index_version'] == version and len(version) == 12
    assert loaded['info'] == {'n_items': 6}
    assert (loaded['k1'], loaded['b'], loaded['id_col']) == (1.5, 0.5, 'spotify_id')
    for q in ['pop', ['rock', 'indie'], 'k-pop']:
        assert infer_bm25(loaded, q, top_k=10) == infer_bm25(model, q, top_k=10)
    # Same content -> same version; different parameters -> different version.
    assert save_bm25(model, tmp_path / 'again') == version
    assert save_bm25(train_bm25(items), tmp_path / 'other') != version


def test_load_old_meta_without_new_keys(items, tmp_path):
    save_bm25(train_bm25(items), tmp_path)
    meta_path = tmp_path / 'meta.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    for key in ('id_col', 'index_version', 'info'):
        del meta[key]
    meta_path.write_text(json.dumps(meta), encoding='utf-8')
    loaded = load_bm25(tmp_path)
    assert loaded['id_col'] == 'spotify_id'
    assert loaded['index_version'] == 'unknown' and loaded['info'] == {}


def test_columns_summed_in_sorted_order(items):
    model = train_bm25(items)
    seen = []

    class Spy:
        def __getitem__(self, key):
            seen.append(key[1])
            return model['index'][key]

    infer_bm25(dict(model, index=Spy()), ['rock', 'pop', 'indie', 'alternative'])
    assert len(seen[0]) == 4 and seen[0] == sorted(seen[0])
