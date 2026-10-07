import pandas as pd
import pytest

import build_index
from bm25 import infer_bm25, load_bm25
from build_index import items_from_genre_matrix, items_from_tag_dicts

GENRE_MATRIX = pd.DataFrame({'id': ['x', 'y', 'z'],
                             'hard rock': [1.5, 0.0, 0.0],
                             'k-pop': [0.0, 2.0, 0.0],
                             'pop': [0.3, 0.7, 0.0]})

TAGS = pd.DataFrame({'id': ['x', 'y', 'y', 'z', None],
                     '(tag, weight)': ["{'chill': 100, 'seen live': 2}",
                                       "{'sad': 50}", "{'sad': 50}", '{}',
                                       "{'pop': 10}"]})


def test_genre_matrix():
    items, stats = items_from_genre_matrix(GENRE_MATRIX)
    assert dict(zip(items['id'], items['genres'])) == {
        'x': ['hard rock', 'pop'], 'y': ['k-pop', 'pop'], 'z': []}
    assert stats['n_empty_docs'] == 1 and stats['n_merged_duplicates'] == 0


def test_tag_dicts():
    items, stats = items_from_tag_dicts(TAGS)
    assert dict(zip(items['id'], items['tags'])) == {
        'x': ['chill', 'seen live'], 'y': ['sad'], 'z': []}
    assert stats == {'min_weight': 0, 'n_dropped_null_ids': 1,
                     'n_merged_duplicates': 1, 'n_items': 3, 'n_empty_docs': 1}
    items, _ = items_from_tag_dicts(TAGS, min_weight=10)
    assert items['tags'][0] == ['chill']


def test_tag_dicts_invalid():
    with pytest.raises(ValueError, match='row 0'):
        items_from_tag_dicts(pd.DataFrame({'id': ['x'], 't': ['{oops']}))


def test_cli_onion_bz2(tmp_path, capsys):
    GENRE_MATRIX.to_csv(tmp_path / 'g.tsv.bz2', sep='\t', index=False)
    TAGS.to_csv(tmp_path / 't.tsv.bz2', sep='\t', index=False)
    build_index.main(['--source', 'onion-genres', '--input',
                      str(tmp_path / 'g.tsv.bz2'), '--out', str(tmp_path / 'genres')])
    build_index.main(['--source', 'onion-tags', '--input',
                      str(tmp_path / 't.tsv.bz2'), '--out', str(tmp_path / 'tags'),
                      '--min-weight', '10'])
    genres, tags = load_bm25(tmp_path / 'genres'), load_bm25(tmp_path / 'tags')
    assert list(infer_bm25(genres, 'rock')) == ['x']
    assert list(infer_bm25(tags, 'chill')) == ['x']
    assert infer_bm25(tags, 'live') == {}
    assert (genres['id_col'], tags['feature_col']) == ('id', 'tags')
    assert tags['info']['source'] == 'onion-tags' and tags['info']['n_items'] == 3
    assert tags['index_version'] in capsys.readouterr().out
