"""Self-contained indexed top-200 files for pipelines; no encoder or GPU imports."""
from pathlib import Path
import json
import math
import os
import sqlite3
import threading
from urllib.parse import quote
import zlib


def validate(top200,scores):
    if (not isinstance(top200,list) or not isinstance(scores,list) or len(top200)!=len(scores)
            or len(top200)>200 or len(set(top200))!=len(top200)
            or any(not isinstance(t,str) or not t or len(t)>128 for t in top200)
            or any(isinstance(s,bool) or not isinstance(s,(int,float)) or not math.isfinite(s) for s in scores)):
        raise ValueError('Invalid cached candidates')


class CacheWriter:
    """Publish a complete SQLite cache atomically; incomplete builds remain temporary."""
    def __init__(self,path,metadata):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.temporary=self.path.with_suffix(self.path.suffix+'.building')
        if self.path.exists():raise ValueError('Cache already exists')
        if self.temporary.exists():self.temporary.unlink()
        self.db=sqlite3.connect(self.temporary)
        self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.execute('PRAGMA synchronous=NORMAL')
        self.db.execute('CREATE TABLE candidates (query_id TEXT PRIMARY KEY, payload BLOB NOT NULL) WITHOUT ROWID')
        self.db.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID')
        self.metadata={**metadata,'schema_version':1};self.count=0

    def add(self,rows):
        def packed():
            for row in rows:
                qid=row['query_id'];top,scores=row['top200'],row['scores']
                if not isinstance(qid,str) or not qid or len(qid)>128:raise ValueError('Invalid query_id')
                validate(top,scores)
                payload=json.dumps({'top200':top,'scores':scores},allow_nan=False,separators=(',',':')).encode()
                yield qid,zlib.compress(payload,level=1)
        self.db.executemany('INSERT INTO candidates VALUES (?,?)',packed())
        self.db.commit();self.count+=len(rows)

    def finish(self,expected_queries):
        if self.count!=expected_queries:raise ValueError('Cache coverage mismatch')
        metadata={**self.metadata,'queries':self.count}
        self.db.executemany('INSERT INTO metadata VALUES (?,?)',[(k,json.dumps(v)) for k,v in metadata.items()])
        self.db.commit();self.db.close()
        os.replace(self.temporary,self.path)
        return self.path

    def close(self):
        self.db.close()


class CandidateCache:
    """Indexed read-only lookup by original query_id; unknown examples fail explicitly."""
    def __init__(self,path,expected_release_id=None,expected_split=None):
        path=Path(path).resolve()
        if not path.is_file():raise FileNotFoundError(path)
        uri='file:'+quote(str(path),safe='/')+'?mode=ro&immutable=1'
        self.db=sqlite3.connect(uri,uri=True,check_same_thread=False)
        self.lock=threading.RLock()
        try:
            self.metadata={k:json.loads(v) for k,v in self.db.execute('SELECT key,value FROM metadata')}
            if self.metadata.get('schema_version')!=1:raise ValueError('Unknown cache schema')
            if expected_release_id and self.metadata.get('release_id')!=expected_release_id:
                raise ValueError('Wrong cached model release')
            if expected_split and self.metadata.get('split')!=expected_split:raise ValueError('Wrong cached split')
        except BaseException:
            self.db.close();raise

    @staticmethod
    def _decode(qid,payload):
        decoder=zlib.decompressobj();raw=decoder.decompress(payload,65537)
        if len(raw)>65536 or not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
            raise ValueError('Invalid cache payload')
        result=json.loads(raw);validate(result['top200'],result['scores'])
        return {'query_id':qid,'top200':result['top200'],'scores':result['scores']}

    def get(self,query_id):
        if not isinstance(query_id,str):raise TypeError('query_id must be a string')
        with self.lock:
            row=self.db.execute('SELECT payload FROM candidates WHERE query_id=?',(query_id,)).fetchone()
        if row is None:raise KeyError(query_id)
        return self._decode(query_id,row[0])

    def get_many(self,query_ids):
        ids=list(query_ids)
        if not all(isinstance(q,str) for q in ids):raise TypeError('query_id must be a string')
        found={}
        with self.lock:
            for start in range(0,len(ids),500):
                block=ids[start:start+500]
                for q,payload in self.db.execute('SELECT query_id,payload FROM candidates WHERE query_id IN ('+
                        ','.join('?' for _ in block)+')',block):
                    found[q]=payload
        return [self._decode(q,found[q]) if q in found else self.get(q) for q in ids]

    def close(self):
        self.db.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
