"""Frozen GPU query embeddings; short benchmark gates full inference by remaining time."""
from pathlib import Path
import argparse
import contextlib
import gc
import json
import os
import sys
import time

os.environ['USE_TF']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from recsys.serving.artifacts import sha256
from recsys.serving.query import QwenQueryEncoder


def atomic(path, value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


@torch.inference_mode()
def encode_batch(model, rows, profiles):
    profile_indices=rows.profile_index.to_numpy()
    h=torch.tensor(profiles['histories'][profile_indices],device=model.device)
    text=model.tokenizer(rows['query'].tolist(),padding=True,truncation=True,
        max_length=model.config.query_tokens,add_special_tokens=False,return_tensors='pt')
    inputs={'hash_indices':model.features['indices'][h],'hash_weights':model.features['weights'][h],
        'audio':model.features['audio'][h],'numbers':model.features['numbers'][h],
        'history_mask':torch.tensor(profiles['masks'][profile_indices],device=model.device),
        'query_ids':text['input_ids'].to(model.device),'query_mask':text['attention_mask'].to(model.device)}
    with (torch.autocast('cuda',dtype=model.dtype) if model.device.type=='cuda' else contextlib.nullcontext()):
        result=model.tower(**inputs).float()
    result=result.cpu().numpy()
    if result.shape!=(len(rows),256) or not np.isfinite(result).all() or not np.allclose(
            np.linalg.norm(result,axis=1),1,atol=1e-5,rtol=0):
        raise ValueError('Invalid query embeddings')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--credentials',type=Path,required=True)
    p.add_argument('--folder-id',required=True)
    p.add_argument('--quota-project',required=True)
    args=p.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jobs/colab_continue'))
    from drive_store import DriveStore
    store=DriveStore(args.credentials,args.folder_id,args.quota_project)
    args.output.mkdir(parents=True,exist_ok=True)
    inputs={}
    for split in ('test_public','train'):
        root=args.inputs/split;manifest=json.loads((root/'manifest.json').read_text())
        for name,digest in manifest['sha256'].items():
            if sha256(root/name)!=digest:raise ValueError('Input checksum mismatch')
        frame=pd.read_parquet(root/'queries.parquet')
        with np.load(root/'profiles.npz',allow_pickle=False) as z:
            profiles={key:z[key] for key in ('histories','masks')}
        inputs[split]=(frame,profiles,manifest)
    model=QwenQueryEncoder(args.left,device='cuda',cache_dir=args.output.parent/'hf_cache')
    if model.dtype!=torch.bfloat16:raise RuntimeError('This job requires BF16 on the authorized G4')
    identity={'release_id':model.serving_manifest['release_id'],'bundle_id':model.manifest['bundle_id'],
        'inputs':{s:v[2]['sha256'] for s,v in inputs.items()},'dtype':str(model.dtype),
        'encoding_recipe':'frozen_qwen_history_v1','model_weights_frozen':True}
    report={'status':'benchmarking','identity':identity,'queries_total':sum(len(x[0]) for x in inputs.values()),
        'gpu':torch.cuda.get_device_name(),'training_updates':0}
    def publish():
        atomic(args.output/'status.json',report);store.put(args.output/'status.json','gpu/status.json')
    publish()
    progress_path=args.output/'progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {
        'identity':identity,'completed':{'test_public':0,'train':0},'chunks':[]}
    if progress['identity']!=identity:raise RuntimeError('Incompatible resumed encoding job')
    rates=[]
    rng=np.random.default_rng(42)
    samples={s:rng.choice(len(v[0]),min(1024,len(v[0])),replace=False) for s,v in inputs.items()}
    for batch in (32,64,128,256):
        if time.time()>args.deadline_unix-240:break
        try:
            frame,profiles,_=inputs['test_public']
            encode_batch(model,frame.iloc[samples['test_public'][:batch]],profiles)
            measured={}
            for split,(frame,profiles,_) in inputs.items():
                torch.cuda.synchronize();started=time.perf_counter()
                selected=samples[split]
                for start in range(0,len(selected),batch):
                    encode_batch(model,frame.iloc[selected[start:start+batch]],profiles)
                torch.cuda.synchronize()
                measured[split]=len(selected)/(time.perf_counter()-started)
            total_seconds=sum(len(v[0])/measured[s] for s,v in inputs.items())
            rates.append({'batch':batch,'qps':measured,'encoding_seconds_estimate':total_seconds})
            print(json.dumps({'benchmark':rates[-1]}),flush=True)
        except torch.cuda.OutOfMemoryError:
            gc.collect();torch.cuda.empty_cache()
            print(json.dumps({'batch':batch,'status':'oom'}),flush=True)
    if not rates:raise RuntimeError('No successful GPU benchmark')
    best=min(rates,key=lambda r:r['encoding_seconds_estimate'])
    remaining_queries=sum(len(v[0])-progress['completed'][s] for s,v in inputs.items())
    # Allows both slower batches and durable-upload overhead; minimum upload estimate
    # is 3 seconds per 4096 rows plus two minutes for final reports and release.
    needed=best['encoding_seconds_estimate']*(remaining_queries/report['queries_total'])*1.25+remaining_queries/4096*3+120
    report.update(benchmark=rates,batch=best['batch'],estimated_remaining_seconds=needed,
        available_seconds=args.deadline_unix-time.time())
    if needed>args.deadline_unix-time.time():
        report['status']='needs_budget';publish();return
    # Compare normalized cosine embeddings between the original batch16 and chosen
    # batch shape; no target labels, optimizer or weight updates are involved.
    frame,profiles,_=inputs['test_public'];comparison=frame.iloc[samples['test_public'][:best['batch']]]
    reference=np.concatenate([encode_batch(model,comparison.iloc[i:i+16],profiles) for i in (0,16)])
    current=encode_batch(model,comparison,profiles)[:32]
    cosine=np.sum(reference*current,axis=1)
    if cosine.min()<.999:raise RuntimeError('Batch inference disagrees with frozen reference')
    report['batch_reference_min_cosine']=float(cosine.min())
    report['status']='running';publish()
    atomic(progress_path,progress);store.put(progress_path,'gpu/progress.json')
    for split,(frame,profiles,_) in inputs.items():
        while progress['completed'][split]<len(frame):
            if time.time()>args.deadline_unix-150:
                report['status']='partial_time_budget';publish();return
            first=progress['completed'][split];end=min(len(frame),first+4096)
            started=time.perf_counter()
            parts=[encode_batch(model,frame.iloc[i:min(i+best['batch'],end)],profiles)
                for i in range(first,end,best['batch'])]
            vector=np.concatenate(parts)
            name=f'vectors/{split}/{first:07d}_{end:07d}.npy'
            target=args.output/name;target.parent.mkdir(parents=True,exist_ok=True)
            np.save(target,vector,allow_pickle=False)
            store.put(target,name)
            progress['chunks'].append({'name':name,'split':split,'first':first,'end':end,
                'sha256':sha256(target),'size':target.stat().st_size})
            progress['completed'][split]=end
            atomic(progress_path,progress);store.put(progress_path,'gpu/progress.json')
            elapsed=time.perf_counter()-started
            report.update(completed=progress['completed'],last_chunk_qps=(end-first)/elapsed,
                estimated_remaining_seconds=sum(len(v[0])-progress['completed'][s] for s,v in inputs.items())/((end-first)/elapsed))
            publish();print(json.dumps({'split':split,'done':end,'total':len(frame),'qps':(end-first)/elapsed}),flush=True)
    report['status']='completed';publish()


if __name__=='__main__':main()
