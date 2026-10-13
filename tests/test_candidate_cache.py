import concurrent.futures
import sqlite3

import pytest
from recsys.serving.cache import CacheWriter,CandidateCache


def test_cache_indexed_lookup_matches_original_top200_and_order(tmp_path):
    path=tmp_path/'train.sqlite'
    rows=[{'query_id':'q'+str(i),'top200':['track'+str(j) for j in range(200)],
        'scores':[1-j/200 for j in range(200)]} for i in range(5)]
    writer=CacheWriter(path,{'split':'train','release_id':'test'})
    writer.add(rows[:2]);writer.add(rows[2:]);writer.finish(5)
    with CandidateCache(path,expected_release_id='test',expected_split='train') as cache:
        assert cache.get('q0')==rows[0]
        assert cache.get_many(['q4','q0','q4'])==[rows[4],rows[0],rows[4]]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            assert list(pool.map(cache.get,['q0','q1','q2','q3']))==rows[:4]
        with pytest.raises(KeyError):cache.get('unknown')
        with pytest.raises(sqlite3.OperationalError):cache.db.execute('DELETE FROM candidates')
    with pytest.raises(ValueError,match='release'):CandidateCache(path,expected_release_id='wrong')
    with pytest.raises(ValueError,match='split'):CandidateCache(path,expected_split='test_public')


def test_cache_not_published_until_complete_and_rejects_duplicate_ids(tmp_path):
    path=tmp_path/'cache.sqlite';writer=CacheWriter(path,{})
    assert not path.exists()
    writer.add([{'query_id':'one','top200':['track'],'scores':[.5]}])
    with pytest.raises(ValueError,match='coverage'):writer.finish(2)
    with pytest.raises(sqlite3.IntegrityError):writer.add([{'query_id':'one','top200':['track'],'scores':[.5]}])
    writer.close()
    with pytest.raises(FileNotFoundError):CandidateCache(path)
