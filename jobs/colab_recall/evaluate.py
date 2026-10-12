"""Top-200 exact/HNSW evaluation of the frozen best bundle; no optimizer."""
from pathlib import Path
import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import time

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

KS = (1, 5, 10, 20, 50, 100, 200)


def score_candidates(scorer, raw_ids, row, listened):
    normal = scorer.clean_ranking(raw_ids, row, k=200)
    key = scorer.keys[scorer.lookup[str(row["target_m4a_id"])]]
    def position(items):
        return next((i for i, tid in enumerate(items, 1)
                     if scorer.keys[scorer.lookup[tid]] == key), None)
    rank = position(normal)
    novel = scorer.clean_ranking(raw_ids, row, listened, discovery=True, k=200)
    new = key not in listened
    new_rank = position(novel) if new else None
    out = {"query_id": row["query_id"], "user_id": row["user_id"], "query_type": row["query_type"],
        "selection_validation": row["selection_validation"], "target_m4a_id": str(row["target_m4a_id"]),
        "target_rank_top200": rank, "is_new_canonical": new, "top200": normal,
        "discovery_top200": novel if new else [], "returned": len(normal),
        "mrr@200": 1 / rank if rank else 0.}
    for k in KS:
        out[f"recall@{k}"] = float(rank is not None and rank <= k)
        out[f"discovery_recall@{k}"] = float(new_rank is not None and new_rank <= k) if new else float("nan")
    for k in (20, 100, 200):
        out[f"ndcg@{k}"] = 1 / math.log2(rank + 1) if rank is not None and rank <= k else 0.
    out["discovery_ndcg@20"] = (1 / math.log2(new_rank + 1)
        if new_rank is not None and new_rank <= 20 else 0.) if new else float("nan")
    return out


def summarize(frame):
    columns = [c for c in frame if c.startswith(("recall@", "discovery_recall@", "ndcg@"))]
    columns += ["mrr@200", "discovery_ndcg@20"]
    by_type = frame.groupby("query_type")[columns].mean()
    return {"queries": len(frame), "micro": {k: float(frame[k].mean()) if math.isfinite(frame[k].mean()) else None for k in columns},
        "macro_over_query_types": {k: float(by_type[k].mean()) if by_type[k].notna().all() else None
                                   for k in columns},
        "by_type": {str(t): {k: float(v) if math.isfinite(v) else None for k, v in row.items()}
                    for t, row in by_type.to_dict("index").items()},
        "queries_by_type": frame.query_type.value_counts().to_dict()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--credentials", type=Path, required=True)
    p.add_argument("--folder-id", required=True)
    p.add_argument("--quota-project", required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--max-hours", type=float, default=3.)
    p.add_argument("--training-complete-epochs", type=int)
    p.add_argument("--best-checkpoint-epoch-fraction", type=float)
    args = p.parse_args()
    import numpy as np
    import pandas as pd
    import torch
    from recsys.two_tower_qwen.bundle import QwenRetriever
    from recsys.two_tower_qwen.metrics import CatalogScorer
    from recsys.two_tower_qwen.data import atomic_json
    from drive_store import DriveStore

    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "chunks").mkdir(exist_ok=True)
    manifest = json.loads((args.inputs / "manifest.json").read_text())
    for name, checksum in manifest["sha256"].items():
        if hashlib.sha256((args.inputs / name).read_bytes()).hexdigest() != checksum:
            raise ValueError("Evaluation input checksum mismatch")
    queries = pd.read_parquet(args.inputs / "queries.parquet")
    with np.load(args.inputs / "profiles.npz") as compressed:
        profiles = {key: compressed[key] for key in compressed.files}
    store = DriveStore(args.credentials, args.folder_id, args.quota_project)
    model = QwenRetriever(args.bundle, device="cuda", cache_dir=args.output.parent / "hf_cache")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    scorer = CatalogScorer(model.catalog, set(queries.query_type))
    group_keys = sorted(set(scorer.keys))
    listened = []
    offsets, groups = profiles["listened_offsets"], profiles["listened_groups"]
    for i in range(len(offsets) - 1):
        listened.append(frozenset(group_keys[int(g)] for g in groups[offsets[i]:offsets[i+1]]))
    item_vectors = torch.tensor(np.load(args.bundle / "track_vectors.npy"), device=model.device)
    identity = {"bundle_id": model.manifest["bundle_id"], "inputs_sha256": manifest["sha256"],
        "cutoffs": list(KS), "retrieval_modes": ["exact", "hnsw"], "hnsw_ef_min": 1024}
    progress_path = args.output / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        "identity": identity, "completed_queries": 0, "chunks": [], "cumulative_seconds": 0.}
    if progress["identity"] != identity:
        raise RuntimeError("Refusing to resume with different weights/data/scoring")
    previous_seconds = progress["cumulative_seconds"]
    report = {"status": "running", "identity": identity, "gpu": torch.cuda.get_device_name(),
        "dtype": str(model.dtype), "queries_total": len(queries), "input_split": manifest,
        "weights_are_frozen": True, "single_positive_canonical_recall_equals_hit_rate": True,
        "training_complete_epochs": args.training_complete_epochs,
        "best_checkpoint_epoch_fraction": args.best_checkpoint_epoch_fraction}

    def publish():
        report.update(completed_queries=progress["completed_queries"],
            session_seconds=time.monotonic()-started,
            cumulative_seconds=previous_seconds + time.monotonic()-started)
        atomic_json(args.output / "status.json", report)
        store.put(args.output / "status.json", "status.json")

    def hnsw_rows(vector, row, history):
        raw = 1024
        while True:
            model.index.set_ef(max(1024, raw))
            labels, _ = model.index.knn_query(vector[None], k=min(raw, len(model.ids)), num_threads=2)
            ids = [model.ids[int(x)] for x in labels[0]]
            scored = score_candidates(scorer, ids, row, history)
            if (scored["returned"] == 200 and (not scored["is_new_canonical"]
                    or len(scored["discovery_top200"]) == 200)) or raw >= len(model.ids):
                return scored
            raw = min(raw * 2, len(model.ids))

    publish()
    while progress["completed_queries"] < len(queries):
        if previous_seconds + time.monotonic() - started > args.max_hours * 3600 - 120:
            report["status"] = "partial_time_budget"
            break
        first = progress["completed_queries"]
        end = min(len(queries), first + args.chunk)
        exact_rows, ann_rows = [], []
        chunk_started = time.monotonic()
        for begin in range(first, end, args.batch):
            rows = queries.iloc[begin:min(end, begin+args.batch)]
            profile_indices = rows.profile_index.to_numpy()
            h = torch.tensor(profiles["histories"][profile_indices], device=model.device)
            text = model.tokenizer(rows["query"].tolist(), padding=True, truncation=True,
                max_length=model.config.query_tokens, add_special_tokens=False, return_tensors="pt")
            inputs = {"hash_indices": model.features["indices"][h],
                "hash_weights": model.features["weights"][h], "audio": model.features["audio"][h],
                "numbers": model.features["numbers"][h],
                "history_mask": torch.tensor(profiles["masks"][profile_indices], device=model.device),
                "query_ids": text["input_ids"].to(model.device), "query_mask": text["attention_mask"].to(model.device)}
            with torch.inference_mode(), torch.autocast("cuda", dtype=model.dtype):
                vectors = model.tower(**inputs).float()
            with torch.inference_mode():
                scores = vectors @ item_vectors.T
                if not torch.isfinite(scores).all():
                    raise FloatingPointError("Non-finite retrieval scores")
                labels = torch.argsort(scores, dim=1, descending=True, stable=True).cpu().numpy()
            encoded = vectors.cpu().numpy()
            for i, row in enumerate(rows.to_dict("records")):
                history = listened[int(row["profile_index"])]
                # Lazy iteration avoids materializing 64k strings for every query.
                # Two cleaning passes need the same complete ranking, so pass a re-iterable view.
                class Ranking:
                    def __iter__(self):
                        return (model.ids[int(x)] for x in labels[i])
                exact = score_candidates(scorer, Ranking(), row, history)
                ann = hnsw_rows(encoded[i], row, history)
                group = lambda rec: {scorer.keys[scorer.lookup[t]] for t in rec}
                ann["index_overlap_recall@200"] = len(group(exact["top200"]) & group(ann["top200"])) / 200
                exact_rows.append(exact)
                ann_rows.append(ann)
        # Chunk files are durably uploaded BEFORE advancing the resume cursor.
        names = []
        for mode, result in [("exact", exact_rows), ("hnsw", ann_rows)]:
            name = f"chunks/{mode}_{first:06d}_{end:06d}.parquet"
            pd.DataFrame(result).to_parquet(args.output / name, index=False)
            store.put(args.output / name, name)
            names.append(name)
        progress["chunks"].extend(names)
        progress["completed_queries"] = end
        progress["cumulative_seconds"] = previous_seconds + time.monotonic() - started
        atomic_json(progress_path, progress)
        store.put(progress_path, "progress.json")
        rate = (end-first) / (time.monotonic()-chunk_started)
        report.update(last_chunk_queries_per_second=rate,
            remaining_hours_estimate=(len(queries)-end) / rate / 3600,
            full_pass_hours_estimate=len(queries) / rate / 3600)
        publish()
        print(json.dumps({"done": end, "total": len(queries), "qps": rate,
            "remaining_hours": report["remaining_hours_estimate"]}), flush=True)
    summaries = {}
    for mode in ("exact", "hnsw"):
        paths = [args.output / name for name in progress["chunks"] if Path(name).name.startswith(mode + "_")]
        if not paths:
            continue
        frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        if frame.query_id.duplicated().any():
            raise RuntimeError("Duplicate evaluation queries")
        summaries[mode] = {"all_holdout": summarize(frame),
            "checkpoint_selection_validation": summarize(frame[frame.selection_validation]),
            "additional_holdout": summarize(frame[~frame.selection_validation]) if (~frame.selection_validation).any() else None}
        if mode == "hnsw":
            summaries[mode]["mean_index_overlap_recall@200"] = float(frame["index_overlap_recall@200"].mean())
    report["status"] = "completed" if progress["completed_queries"] == len(queries) else report["status"]
    report["metrics"] = summaries
    report["note"] = "Additional holdout excludes query IDs used for checkpoint selection; users may overlap that selection subset."
    atomic_json(args.output / "metrics.json", summaries)
    store.put(args.output / "metrics.json", "metrics.json")
    publish()
    print(json.dumps({"status": report["status"], "queries": progress["completed_queries"]}), flush=True)


if __name__ == "__main__":
    main()
