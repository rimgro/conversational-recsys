import json

import pytest

import build_index
from bm25 import infer_bm25, load_bm25

TRACKS = [
    {'m4a_id': 'x', 'm4a_artist': 'Oddisee', 'm4a_song': 'After Thoughts',
     'm4a_genres_full': 'hard rock,rock', 'm4a_tags_full': 'chill,sad'},
    {'m4a_id': 'y', 'm4a_artist': 'Nickelback', 'm4a_song': 'Burn It',
     'm4a_genres_full': 'pop', 'm4a_tags_full': None},
    {'m4a_id': 'y', 'm4a_artist': 'dup', 'm4a_song': 'dup',
     'm4a_genres_full': 'jazz', 'm4a_tags_full': 'jazz'},
    {'m4a_id': 'z', 'm4a_artist': 'Rhapsody', 'm4a_song': 'Flames',
     'm4a_genres_full': '', 'm4a_tags_full': 'sad'},
    {'m4a_id': None, 'm4a_artist': 'ghost', 'm4a_song': 'ghost',
     'm4a_genres_full': 'rock', 'm4a_tags_full': 'sad'},
]


@pytest.fixture
def meta(tmp_path):
    path = tmp_path / 'tracks_meta.jsonl'
    path.write_text('\n'.join(json.dumps(t, ensure_ascii=False) for t in TRACKS) + '\n',
                    encoding='utf-8')
    return path


def test_load_items(meta):
    items, stats = build_index.load_items(meta, ['m4a_genres_full'])
    assert dict(zip(items['m4a_id'], items['terms'])) == {
        'x': ['hard rock', 'rock'], 'y': ['pop'], 'z': ['']}
    assert stats == {'n_rows': 5, 'n_dropped_null_ids': 1,
                     'n_dropped_duplicate_ids': 1, 'n_items': 3,
                     'n_empty_docs': 1}


def test_missing_column(meta):
    with pytest.raises(ValueError, match='nope'):
        build_index.load_items(meta, ['nope'])


def test_cli(meta, tmp_path, capsys):
    out = tmp_path / 'idx'
    build_index.main(['--input', str(meta), '--field', 'm4a_artist', 'm4a_song',
                      '--out', str(out)])
    model = load_bm25(out)
    assert list(infer_bm25(model, 'oddisee thoughts')) == ['x']
    assert (model['id_col'], model['feature_col']) == ('m4a_id', 'm4a_artist+m4a_song')
    assert model['info']['n_items'] == 3
    assert model['index_version'] in capsys.readouterr().out


def test_parquet_shards(tmp_path):
    import pandas as pd
    shards = tmp_path / 'shards'
    shards.mkdir()
    df = pd.DataFrame(TRACKS)
    df.iloc[:2].to_parquet(shards / 'tracks_meta-00000-of-00002.parquet')
    df.iloc[2:].to_parquet(shards / 'tracks_meta-00001-of-00002.parquet')
    items, stats = build_index.load_items(shards, ['m4a_genres_full'])
    assert dict(zip(items['m4a_id'], items['terms'])) == {
        'x': ['hard rock', 'rock'], 'y': ['pop'], 'z': ['']}
    assert stats['n_dropped_duplicate_ids'] == 1
