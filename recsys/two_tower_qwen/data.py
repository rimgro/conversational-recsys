"""Bounded preparation, resumable fixed text features, and past-only histories."""
from dataclasses import dataclass
from pathlib import Path
import gc
import hashlib
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm

from recsys.track_tower import (NUMERIC_FIELDS, TrackPreprocessor, card_text,
                               lyric_chunks, pack_features, pool_lyric_chunks, text_value)
from .features import HistoryHasher, padded_histories, recent_history
from .metrics import CatalogScorer

REQUIRED_FILES = ("tracks_meta.parquet", "train.parquet", "train_queries.parquet", "train_qrels.parquet")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def locate_data(root, supplied=None):
    candidates = [Path(supplied)] if supplied else [Path(root) / "music4all_crs",
        Path(root) / "datasets" / "music4all_crs", Path(root) / "new_dataset" / "music4all_crs"]
    for candidate in candidates:
        if all((candidate / name).is_file() for name in REQUIRED_FILES):
            return candidate.resolve()
    raise FileNotFoundError("Set DATA_DIR to the directory containing " + ", ".join(REQUIRED_FILES))


class ExperimentBudget:
    """Persist active invocation time, including preparation before run creation.

    This does not meter provider billing or idle time between notebook executions.
    """
    def __init__(self, artifact_dir, data_dir, config, started_at=None):
        source_stats = {p.name: [p.stat().st_size, p.stat().st_mtime_ns]
                        for p in Path(data_dir).glob("*.parquet")}
        package = Path(__file__).parent
        identity = digest({"data_stats": source_stats, "config": config.identity(),
                           "code": {p.name: file_hash(p) for p in package.glob("*.py")}})
        self.path = Path(artifact_dir) / "budgets" / (identity[:20] + ".json")
        self.started_at = time.monotonic() if started_at is None else started_at
        self.previous = json.loads(self.path.read_text())["elapsed_seconds"] if self.path.is_file() else 0.
        self.max_seconds = config.max_hours * 3600
        self.record()

    def elapsed(self):
        return self.previous + time.monotonic() - self.started_at

    def record(self):
        atomic_json(self.path, {"elapsed_seconds": self.elapsed(), "max_seconds": self.max_seconds})

    def check(self):
        self.record()
        if self.elapsed() >= self.max_seconds:
            raise TimeoutError("Experiment active-time budget reached")


def check_deadline(deadline):
    if isinstance(deadline, ExperimentBudget):
        deadline.check()
    elif deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Time budget reached during preparation; completed cache blocks are retained")


def cached_encode(encoder, texts, path, batch_size, deadline=None):
    path = Path(path)
    status_path = path.with_suffix(".json")
    dimension = encoder.get_sentence_embedding_dimension()
    shape = (len(texts), dimension)
    if path.is_file() and status_path.is_file():
        matrix = np.load(path, mmap_mode="r+")
        status = json.loads(status_path.read_text())
        if matrix.shape != shape:
            raise ValueError("Text cache shape mismatch")
        completed = status["completed"]
    else:
        matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
        completed = 0
        atomic_json(status_path, {"completed": completed})
    for start in tqdm(range(completed, len(texts), 2048), desc=path.stem):
        check_deadline(deadline)
        stop = min(start + 2048, len(texts))
        value = encoder.encode(texts[start:stop], batch_size=batch_size,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        if value.shape != (stop - start, dimension) or not np.isfinite(value).all():
            raise ValueError("Invalid fixed text embeddings")
        matrix[start:stop] = value
        matrix.flush()
        atomic_json(status_path, {"completed": stop})
    return np.load(path, mmap_mode="r")


@dataclass
class PreparedData:
    catalog: list
    pairs: pd.DataFrame
    fit_rows: np.ndarray
    validation_rows: np.ndarray
    full_validation_rows: np.ndarray
    histories: np.ndarray
    history_mask: np.ndarray
    listened: dict
    query_ids: np.ndarray
    query_mask: np.ndarray
    item_features: np.ndarray
    hash_indices: np.ndarray
    hash_weights: np.ndarray
    audio: np.ndarray
    history_numbers: np.ndarray
    preprocessor: TrackPreprocessor
    scorer: CatalogScorer
    fingerprint: str
    cache_dir: Path


def balanced_validation(pairs, rows, per_type, seed):
    groups = []
    selected = pairs.iloc[rows]
    for _, group in selected.groupby("query_type", sort=True):
        # Include both discovery-eligible and familiar targets where available.
        strata = [part for _, part in group.groupby("is_new", dropna=False)]
        quota = per_type // max(1, len(strata))
        chosen = []
        for j, part in enumerate(strata):
            chosen.extend(part.sample(n=min(quota, len(part)), random_state=seed + j).index)
        remaining = group.drop(index=chosen)
        chosen.extend(remaining.sample(n=min(per_type - len(chosen), len(remaining)),
                                      random_state=seed + 10).index)
        groups.extend(chosen)
    return np.array(sorted(groups), dtype=np.int64)


def prepare_data(data_dir, artifact_dir, config, tokenizer, device, deadline=None):
    config.validate()
    data_dir, artifact_dir = Path(data_dir), Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    muq_path = data_dir / "track_embeddings.parquet"
    if not muq_path.is_file():
        muq_path = data_dir / "tracks_meta.parquet"
    source_paths = sorted(set([data_dir / name for name in REQUIRED_FILES] + [muq_path]))
    sources = {}
    for path in source_paths:
        check_deadline(deadline)
        sources[path.name] = file_hash(path)
    code_sources = [Path(__file__), Path(__file__).with_name("features.py"),
                    Path(__file__).parents[1] / "track_tower.py"]
    feature_config = {k: config.to_dict()[k] for k in (
        "seed", "hash_buckets", "history_events", "query_tokens", "qwen_model", "qwen_revision",
        "text_model", "text_revision", "text_tokens", "lyric_words", "lyric_overlap", "lyric_max_chunks",
        "max_categories", "max_users", "max_queries_per_user", "validation_user_fraction", "validation_per_type")}
    signature = digest({"sources": sources, "features": feature_config,
                        "code": {p.name: file_hash(p) for p in code_sources}})
    cache = artifact_dir / "cache" / signature[:20]
    cache.mkdir(parents=True, exist_ok=True)
    atomic_json(cache / "sources.json", {"sha256": sources, "feature_config": feature_config})
    names = set(pq.read_schema(data_dir / "tracks_meta.parquet").names)
    desired = ["m4a_id", "spotify_id", "m4a_artist", "m4a_song", "artist", "title",
        "m4a_genres_full", "m4a_tags_full", "lang", "key", "mode", "is_instrumental",
        "pseudo_caption", "lastfm_tag_weights", "lyrics", *NUMERIC_FIELDS]
    catalog_frame = pq.read_table(data_dir / "tracks_meta.parquet",
                                  columns=[n for n in desired if n in names]).to_pandas()
    records = catalog_frame.to_dict("records")
    ids = [str(r["m4a_id"]) for r in records]
    lookup = {t: i for i, t in enumerate(ids)}
    if len(ids) < 2 or len(lookup) != len(ids):
        raise ValueError("Catalog requires unique IDs and at least two tracks")
    queries = pq.read_table(data_dir / "train_queries.parquet").to_pandas()
    labels = pq.read_table(data_dir / "train_qrels.parquet").to_pandas()
    pairs = queries.merge(labels, on="query_id", how="left", validate="one_to_one")
    pairs["query"] = pairs["query"].map(text_value)
    if pairs["target_m4a_id"].isna().any() or pairs["query"].eq("").any():
        raise ValueError("Missing query text/target")
    pairs["target_index"] = pairs["target_m4a_id"].astype(str).map(lookup)
    if pairs["target_index"].isna().any():
        raise ValueError("Target outside catalog")
    pairs["target_index"] = pairs["target_index"].astype(np.int64)
    if config.max_users is not None:
        uids = np.unique(pairs.user_id)
        selected = np.random.default_rng(config.seed).choice(uids, min(len(uids), config.max_users), replace=False)
        pairs = pairs[pairs.user_id.isin(selected)]
    if config.max_queries_per_user is not None:
        pairs = pairs.sample(frac=1, random_state=config.seed).groupby("user_id", sort=True).head(config.max_queries_per_user)
    pairs = pairs.reset_index(drop=True)
    unique_users = np.unique(pairs.user_id)
    if len(unique_users) < 2:
        raise ValueError("Need at least two users for user holdout")
    rng = np.random.default_rng(config.seed)
    heldout = set(rng.choice(unique_users, max(1, int(len(unique_users) * config.validation_user_fraction)), replace=False))
    validation_all = np.flatnonzero(pairs.user_id.isin(heldout).to_numpy())
    fit_rows = np.flatnonzero(~pairs.user_id.isin(heldout).to_numpy())
    validation_rows = balanced_validation(pairs, validation_all, config.validation_per_type, config.seed)
    scorer = CatalogScorer(records, set(pairs.query_type))
    if set(pairs.iloc[validation_rows].query_type) != set(scorer.query_types):
        raise ValueError("User holdout lacks query types; increase the smoke-test user/query limits")
    preprocessor = TrackPreprocessor(config.max_categories).fit(records)
    categories, numbers, structured = preprocessor.transform(records)
    preprocessor.save(cache / "preprocessing.json")
    muq = pq.read_table(muq_path, columns=["m4a_id", "muq_embedding"]).to_pandas().set_index("m4a_id")
    audio = np.zeros((len(ids), 128), dtype=np.float32)
    for i, tid in enumerate(ids):
        if tid not in muq.index:
            raise ValueError(f"Missing MuQ for {tid}")
        value = np.asarray(muq.at[tid, "muq_embedding"], dtype=np.float32)
        if value.shape != (128,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid MuQ for {tid}")
        audio[i] = value / max(1e-8, float(np.linalg.norm(value)))
    audio_present = np.linalg.norm(audio, axis=1) > 0
    hash_indices, hash_weights = HistoryHasher(config.hash_buckets, config.seed).transform(records)
    history_numbers = np.column_stack((numbers, audio_present, np.any(hash_weights != 0, axis=1))).astype(np.float32)
    user_cache = cache / "users.parquet"
    if user_cache.is_file():
        users = pd.read_parquet(user_cache)
    else:
        wanted_users = set(unique_users)
        users = []
        for batch in tqdm(pq.ParquetFile(data_dir / "train.parquet").iter_batches(
                batch_size=8, columns=["user_id", "history", "history_ts", "history_len"]), desc="Past histories"):
            check_deadline(deadline)
            for row in batch.to_pylist():
                uid = row["user_id"]
                if uid not in wanted_users:
                    continue
                history, timestamps = row["history"], row["history_ts"]
                if len(history) != len(timestamps) or len(history) != row["history_len"]:
                    raise ValueError("Misaligned listening events")
                if timestamps and (max(timestamps) >= 1579564800 or any(
                        a > b for a, b in zip(timestamps, timestamps[1:]))):
                    raise ValueError("Train history contains future or unordered events")
                users.append({"user_id": uid, "recent": recent_history(history, lookup, config.history_events),
                    "listened": sorted(scorer.history_keys(history)) if uid in heldout else []})
        users = pd.DataFrame(users).sort_values("user_id").reset_index(drop=True)
        if not users.user_id.is_unique or set(users.user_id) != wanted_users:
            raise ValueError("Missing/duplicate user histories")
        temporary = user_cache.with_suffix(".tmp.parquet")
        users.to_parquet(temporary, index=False)
        temporary.replace(user_cache)
    histories, masks = padded_histories(users.recent.tolist())
    user_lookup = {u: i for i, u in enumerate(users.user_id)}
    pairs["user_index"] = pairs.user_id.map(user_lookup).astype(np.int64)
    listened = {int(r.user_id): frozenset(r.listened) for r in users.itertuples() if r.user_id in heldout}
    qt_path, qm_path = cache / "query_ids.npy", cache / "query_mask.npy"
    token_status = cache / "tokenization.json"
    if qt_path.is_file() and qm_path.is_file() and token_status.is_file():
        qt = np.load(qt_path, mmap_mode="r+")
        qm = np.load(qm_path, mmap_mode="r+")
        completed = json.loads(token_status.read_text())["completed"]
    else:
        qt = np.lib.format.open_memmap(qt_path, mode="w+", dtype=np.int32, shape=(len(pairs), config.query_tokens))
        qm = np.lib.format.open_memmap(qm_path, mode="w+", dtype=bool, shape=qt.shape)
        completed = 0
    for start in tqdm(range(completed, len(pairs), 4096), desc="Qwen query tokens"):
        check_deadline(deadline)
        stop = min(start + 4096, len(pairs))
        encoded = tokenizer(pairs.iloc[start:stop]["query"].tolist(), add_special_tokens=False,
            padding="max_length", truncation=True, max_length=config.query_tokens, return_tensors="np")
        qt[start:stop], qm[start:stop] = encoded["input_ids"], encoded["attention_mask"]
        qt.flush(); qm.flush()
        atomic_json(token_status, {"completed": stop})
    item_path = cache / "item_features.npy"
    if not item_path.is_file():
        from sentence_transformers import SentenceTransformer
        check_deadline(deadline)
        encoder = SentenceTransformer(config.text_model, revision=config.text_revision,
            cache_folder=str(artifact_dir / "hf_cache"), device=str(device), trust_remote_code=False)
        encoder.max_seq_length = config.text_tokens
        encoder.eval()
        cards = cached_encode(encoder, [card_text(r) for r in records], cache / "cards.npy", config.encoding_batch, deadline)
        chunks, owners = [], []
        for i, row in enumerate(records):
            fragment = lyric_chunks(row.get("lyrics"), config.lyric_words, config.lyric_overlap, config.lyric_max_chunks)
            chunks.extend(fragment)
            owners.extend([i] * len(fragment))
        if chunks:
            chunk_vectors = cached_encode(encoder, chunks, cache / "lyrics.npy", config.encoding_batch, deadline)
            lyrics = pool_lyric_chunks(np.asarray(chunk_vectors), np.array(owners), len(records))
        else:
            lyrics = np.zeros_like(cards)
        lyric_present = np.bincount(np.asarray(owners, dtype=np.int64), minlength=len(records)) > 0
        card_present = np.array([any(text_value(r.get(k)) for k in ("m4a_song", "title", "m4a_artist",
            "artist", "m4a_genres_full", "m4a_tags_full", "pseudo_caption")) for r in records])
        present = np.column_stack((audio_present, card_present, lyric_present, structured))
        item_features = pack_features(audio, cards, lyrics, categories, numbers, present)
        temporary = item_path.with_suffix(".tmp.npy")
        np.save(temporary, item_features); temporary.replace(item_path)
        del encoder, chunks, owners
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    item_features = np.load(item_path, mmap_mode="r")
    np.savez_compressed(cache / "history_features.npz", indices=hash_indices, weights=hash_weights,
                        audio=audio, numbers=history_numbers)
    text_dim = (item_features.shape[1] - 128 - config.max_categories - 12 - 5) // 2
    atomic_json(cache / "item_args.json", {"text_dim": text_dim,
        "num_categories": len(preprocessor.vocabulary), "max_categories": config.max_categories,
        "token_dim": config.item_token_dim, "embedding_dim": config.embedding_dim,
        "layers": config.item_layers, "heads": config.item_heads,
        "feedforward_dim": config.item_feedforward, "dropout": config.dropout})
    print(f"Catalog: {len(ids)}; fit: {len(fit_rows)}; held-out: {len(validation_all)}; regular validation: {len(validation_rows)}")
    return PreparedData(records, pairs, fit_rows, validation_rows, validation_all, histories, masks,
        listened, np.load(qt_path, mmap_mode="r"), np.load(qm_path, mmap_mode="r"), item_features,
        hash_indices, hash_weights, audio, history_numbers, preprocessor, scorer, signature, cache)
