import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from serving_fixture import make_export,write_manifest
from recsys.serving.artifacts import split_bundle
from recsys.serving.index import CandidateIndex
from recsys.serving.offline import OfflineCandidates
from recsys.two_tower_qwen.config import TrainConfig


def tool(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).resolve().parents[1]/'tools'/name)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_batch_cpu_candidates_match_api_full_history_and_reference(tmp_path):
    source=make_export(tmp_path/'source');split_bundle(source,tmp_path/'split')
    index=CandidateIndex(tmp_path/'split/right');batch=OfflineCandidates(index)
    histories=[index.ids[:900],[]]
    group_list=[sorted({batch.groups[index.keys[index.lookup[t]]] for t in h}) for h in histories]
    profiles={'listened_groups':np.array(group_list[0],np.int32),'listened_offsets':np.array([0,len(group_list[0]),len(group_list[0])])}
    rows=pd.DataFrame([{'query_id':'q'+str(i),'user_id':i,'split':'train','query_type':kind,'training_subset':'fit',
        'profile_index':i,'exclude_m4a_id':'0','exclude_artist':'ARTIST0'} for i,kind in enumerate(('exact','similar_to'))])
    vector=np.array([1.]+[0.]*255,np.float32)
    output=batch.generate(np.stack([vector,vector]),rows,profiles)
    for i,row in enumerate(rows.to_dict('records')):
        for discovery in (False,True):
            reference=index.search(vector,history=histories[i],discovery=discovery,
                query_type=row['query_type'],exclude_m4a_id=row['exclude_m4a_id'],exclude_artist=row['exclude_artist'])
            prefix='discovery_' if discovery else ''
            assert output[i][prefix+'top200']==reference['ids']
            np.testing.assert_allclose(output[i][prefix+'scores'],reference['scores'],atol=1e-6)
    assert min(int(t)//2 for t in output[0]['discovery_top200'])>=450


def test_preparation_excludes_target_labels_and_rejects_future_history(tmp_path):
    prep=tool('prepare_split_inference.py')
    source=make_export(tmp_path/'source',n=12)
    (source/'config.json').write_text(json.dumps({'training':TrainConfig().to_dict()}));write_manifest(source)
    split_bundle(source,tmp_path/'split')
    data=tmp_path/'data';data.mkdir()
    queries=pd.DataFrame([{'query_id':'q1','user_id':1,'split':'train','query':'some query','source':'positive','query_type':'exact'}])
    queries.to_parquet(data/'train_queries.parquet')
    pd.DataFrame([{'query_id':'q1','exclude_m4a_id':'','exclude_artist':'','target_m4a_id':'8'}]).to_parquet(data/'train_qrels.parquet')
    users=pd.DataFrame([{'user_id':1,'split':'train','history':['0','unknown','1'],'history_ts':[1,2,3],'history_len':3}])
    users.to_parquet(data/'train.parquet')
    result=prep.prepare(data,tmp_path/'split/left',tmp_path/'out','train')
    frame=pd.read_parquet(tmp_path/'out/queries.parquet')
    assert result['queries']==1 and not any(c.startswith('target_') for c in frame.columns)
    with np.load(tmp_path/'out/profiles.npz') as p:
        assert p['histories'][0][p['masks'][0]].tolist()==[0,1]
    users.at[0,'history_ts']=[1,2,1579564800];users.to_parquet(data/'train.parquet')
    with pytest.raises(ValueError,match='future'):
        prep.prepare(data,tmp_path/'split/left',tmp_path/'bad','train')
