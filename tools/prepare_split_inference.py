"""Prepare full train/test_public inputs with split-correct, pre-window histories."""
from pathlib import Path
import argparse
import datetime
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys.serving.artifacts import load_subset, sha256
from recsys.two_tower_qwen.features import padded_histories, recent_history
from recsys.two_tower_qwen.metrics import CatalogScorer
from recsys.track_tower import text_value

CUTOFFS = {'train': '2020-01-21', 'test_public': '2020-02-20'}


def prepare(data, left, output, split, selection_file=None):
    serving, _ = load_subset(left, 'left')
    config = json.loads((left / 'config.json').read_text())['training']
    catalog = json.loads((left / 'catalog.json').read_text())
    queries_path, qrels_path = data / (split+'_queries.parquet'), data / (split+'_qrels.parquet')
    queries = pd.read_parquet(queries_path)
    references = pd.read_parquet(qrels_path, columns=['query_id','exclude_m4a_id','exclude_artist'])
    if not queries.query_id.is_unique or not references.query_id.is_unique:
        raise ValueError('Duplicate query_id')
    if set(queries.query_id) != set(references.query_id) or set(queries['split']) != {split}:
        raise ValueError('Query/reference split mismatch')
    queries = queries.merge(references, on='query_id', how='left', validate='one_to_one', sort=False)
    queries['query'] = queries['query'].map(text_value)
    if queries['query'].eq('').any(): raise ValueError('Empty query text')
    scorer = CatalogScorer(catalog, set(queries.query_type))
    needed = set(queries.user_id)
    cutoff = datetime.datetime.fromisoformat(CUTOFFS[split]).replace(tzinfo=datetime.timezone.utc).timestamp()
    rows, flat, offsets = [], [], [0]
    for batch in pq.ParquetFile(data/(split+'.parquet')).iter_batches(batch_size=64,
            columns=['user_id','split','history','history_ts','history_len']):
        for row in batch.to_pylist():
            if row['user_id'] not in needed: continue
            h, ts = row['history'], row['history_ts']
            if row['split'] != split or len(h) != len(ts) or len(h) != row['history_len']:
                raise ValueError('Misaligned split/history')
            if ts and (max(ts) >= cutoff or any(a>b for a,b in zip(ts,ts[1:]))):
                raise ValueError('History contains future/unordered events')
            rows.append((row['user_id'], recent_history(h, scorer.lookup, config['history_events'])))
            flat.extend(sorted({int(scorer.group_ids[scorer.lookup[str(t)]]) for t in h if str(t) in scorer.lookup}))
            offsets.append(len(flat))
    if len(rows) != len(needed) or {u for u,_ in rows} != needed:
        raise ValueError('Missing/duplicate user history')
    histories, masks = padded_histories([h for _,h in rows])
    lookup = {u:i for i,(u,_) in enumerate(rows)}
    queries['profile_index'] = queries.user_id.map(lookup).astype(np.int64)
    if split == 'train':
        users = np.unique(queries.user_id)
        heldout = set(np.random.default_rng(config['seed']).choice(users,
            max(1,int(len(users)*config['validation_user_fraction'])),replace=False))
        queries['training_subset'] = np.where(queries.user_id.isin(heldout), 'additional_holdout', 'fit')
        if selection_file:
            selected = set(pd.read_parquet(selection_file, columns=['query_id']).query_id)
            if len(selected) != 2600 or not selected.issubset(set(queries.query_id)):
                raise ValueError('Invalid checkpoint-selection IDs')
            queries.loc[queries.query_id.isin(selected),'training_subset'] = 'checkpoint_selection'
    else:
        queries['training_subset'] = 'official_test_public'
    output.mkdir(parents=True, exist_ok=True)
    queries.to_parquet(output/'queries.parquet',index=False)
    np.savez_compressed(output/'profiles.npz', histories=histories, masks=masks,
        listened_groups=np.asarray(flat,np.int32),listened_offsets=np.asarray(offsets,np.int64))
    manifest = {'split':split,'queries':len(queries),'users':len(rows),'query_types':sorted(set(queries.query_type)),
        'history_cutoff_utc':CUTOFFS[split],'bundle_id':serving['bundle_id'],'release_id':serving['release_id'],
        'source_sha256':{p.name:sha256(p) for p in (queries_path,qrels_path,data/(split+'.parquet'))},
        'sha256':{name:sha256(output/name) for name in ('queries.parquet','profiles.npz')},
        'training_subsets':queries.training_subset.value_counts().to_dict(),
        'target_labels_excluded_from_encoder_inputs':True,'history_future_leakage_checked':True}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--left',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--selection-file',type=Path)
    args=p.parse_args()
    for split in ('test_public','train'):
        print(json.dumps(prepare(args.data,args.left,args.output/split,split,args.selection_file)),flush=True)


if __name__=='__main__':main()
