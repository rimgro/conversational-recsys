"""Batched CPU candidate generation, equivalent to the immutable vector API."""
import numpy as np
from .index import text


class OfflineCandidates:
    def __init__(self,index,threads=4):
        self.index=index;self.threads=threads
        self.groups={key:i for i,key in enumerate(sorted(set(index.keys)))}
        self.group_ids=np.asarray([self.groups[key] for key in index.keys],np.int32)
        self.artists=np.asarray(index.artists,dtype=object)

    def _clean(self,labels,distances,row,profiles):
        groups=self.group_ids[labels]
        # The first alias is marked seen before exclusions, matching the frozen scorer.
        first=np.sort(np.unique(groups,return_index=True)[1]);labels=labels[first];distances=distances[first]
        groups=self.group_ids[labels]
        keep=np.ones(len(labels),bool)
        if row['query_type']=='similar_to':
            reference=text(row.get('exclude_m4a_id'))
            if reference in self.index.lookup:
                keep&=groups!=self.group_ids[self.index.lookup[reference]]
            artist=text(row.get('exclude_artist')).casefold()
            if artist:keep&=self.artists[labels]!=artist
        p=int(row['profile_index']);offsets=profiles['listened_offsets']
        listened=profiles['listened_groups'][offsets[p]:offsets[p+1]]
        if len(listened):
            position=np.searchsorted(listened,groups)
            seen=(position<len(listened))&(listened[np.minimum(position,len(listened)-1)]==groups)
        else:seen=np.zeros(len(groups),bool)
        normal=np.flatnonzero(keep)[:200];novel=np.flatnonzero(keep&~seen)[:200]
        return [(labels[s],1-distances[s]) for s in (normal,novel)]

    def generate(self,vectors,rows,profiles):
        vectors=np.asarray(vectors,dtype=np.float32)
        if vectors.shape!=(len(rows),256) or not np.isfinite(vectors).all():
            raise ValueError('Invalid batch of query vectors')
        norms=np.linalg.norm(vectors,axis=1)
        if not np.allclose(norms,1,atol=1e-5,rtol=0):raise ValueError('Non-normalized vectors')
        vectors=vectors/norms[:,None]
        n=len(self.index.ids);raw=min(n,1024)
        labels,distances=self.index.index.knn_query(vectors,k=raw,num_threads=self.threads)
        output=[]
        for i,row in enumerate(rows.to_dict('records')):
            cleaned=self._clean(labels[i],distances[i],row,profiles)
            count=raw
            while (len(cleaned[0][0])<200 or len(cleaned[1][0])<200) and count<n:
                count=min(n,count*2)
                extra,scores=self.index.index.knn_query(vectors[i:i+1],k=count,num_threads=1)
                cleaned=self._clean(extra[0],scores[0],row,profiles)
            result={k:row[k] for k in ('query_id','user_id','split','query_type','training_subset')}
            for prefix,(selected,scores) in zip(('', 'discovery_'),cleaned):
                result[prefix+'top200']=[self.index.ids[int(label)] for label in selected]
                result[prefix+'scores']=scores.astype(np.float32).tolist()
            result['top20']=result['top200'][:20]
            result['response']=''  # Candidate generator does not generate an LLM reply.
            output.append(result)
        return output
