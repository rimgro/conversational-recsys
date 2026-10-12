"""Canonical top-200 candidate generation from a matched exported bundle."""
from pathlib import Path
import argparse
import json
import os
import sys

os.environ['USE_TF'] = '0'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--query', required=True)
    p.add_argument('--history', type=Path, help='JSON array of chronological m4a IDs')
    p.add_argument('--top-k', type=int, default=200)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--exact', action='store_true')
    p.add_argument('--discovery', action='store_true')
    p.add_argument('--query-type', default='exact')
    p.add_argument('--exclude-m4a-id', default='')
    p.add_argument('--exclude-artist', default='')
    args = p.parse_args()
    if not 1 <= args.top_k <= 200:
        p.error('top-k must be between 1 and 200')
    history = json.loads(args.history.read_text()) if args.history else []
    if not isinstance(history, list) or not all(isinstance(t, str) for t in history):
        raise ValueError('History must be a JSON array of string m4a IDs')
    import numpy as np
    from recsys.two_tower_qwen.bundle import QwenRetriever
    from recsys.two_tower_qwen.metrics import CatalogScorer
    model = QwenRetriever(args.bundle, device=args.device, cache_dir=args.cache_dir)
    scorer = CatalogScorer(model.catalog, [args.query_type])
    vector = model.encode_query(args.query, history)
    row = {'query_type': args.query_type, 'exclude_m4a_id': args.exclude_m4a_id,
        'exclude_artist': args.exclude_artist}
    listened = scorer.history_keys(history)
    if args.exact:
        vectors = np.load(args.bundle / 'track_vectors.npy', mmap_mode='r')
        similarities = vectors @ vector
        ranked = np.argsort(-similarities, kind='stable')
        candidates = [model.ids[int(i)] for i in ranked]
        scores = {model.ids[int(i)]: float(similarities[i]) for i in ranked}
        clean = scorer.clean_ranking(candidates, row, listened, args.discovery, args.top_k)
    else:
        model.index.set_ef(min(len(model.ids), max(1024, 5 * args.top_k)))
        count = min(len(model.ids), max(1024, 5 * args.top_k))
        while True:
            labels, distances = model.index.knn_query(vector[None, :], k=count)
            candidates = [model.ids[int(i)] for i in labels[0]]
            clean = scorer.clean_ranking(candidates, row, listened, args.discovery, args.top_k)
            scores = {model.ids[int(i)]: float(1-d) for i,d in zip(labels[0],distances[0])}
            if len(clean) >= args.top_k or count == len(model.ids):
                break
            count = min(len(model.ids), count*2)
    print(json.dumps({'bundle_id': model.manifest['bundle_id'], 'mode': 'exact' if args.exact else 'hnsw',
        'returned': len(clean), 'candidates': [{'m4a_id': t, 'score': scores[t]} for t in clean]}, ensure_ascii=False))

if __name__ == '__main__':
    main()
