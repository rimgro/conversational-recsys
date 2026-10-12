"""Deterministic, signed feature hashing with per-field mean weights.

Only categorical values are hashed. Audio and fitted numeric features stay dense.
Neither track IDs nor titles, query text, timestamps or future labels are hashed.
"""
import hashlib
import unicodedata

import numpy as np

from recsys.track_tower import category_tokens, text_value


def normalized(value):
    return unicodedata.normalize("NFKC", text_value(value)).casefold()


def categorical_fields(row):
    fields = {}
    artist = normalized(row.get("m4a_artist")) or normalized(row.get("artist"))
    if artist:
        fields["artist"] = [artist]
    # Reuse canonical handling of key/mode/instrumental; lists below have no cap.
    for token in category_tokens(row):
        name, value = token.split("=", 1)
        if name in ("lang", "key", "mode", "is_instrumental"):
            fields[name] = [normalized(value)]
    for source, name in (("m4a_genres_full", "genre"), ("m4a_tags_full", "tag")):
        raw = row.get(source)
        parts = raw if isinstance(raw, (list, tuple, np.ndarray)) else text_value(raw).split(",")
        values = sorted({normalized(v) for v in parts if normalized(v)})
        if values:
            fields[name] = values
    return fields


class HistoryHasher:
    def __init__(self, buckets=8192, seed=42):
        if buckets <= 0:
            raise ValueError("Hash buckets must be positive")
        self.buckets, self.seed = int(buckets), int(seed)

    def bucket_sign(self, token):
        # Built-in hash() is process-randomized; use a portable specified digest.
        digest = hashlib.blake2b((str(self.seed) + "\0" + token).encode("utf8"),
                                 digest_size=8, person=b"recsys-hist-v1").digest()
        bucket = int.from_bytes(digest[:4], "little") % self.buckets
        sign = 1. if digest[4] & 1 else -1.
        return bucket, sign

    def row(self, record):
        result = {}
        for field, values in categorical_fields(record).items():
            for value in values:
                bucket, sign = self.bucket_sign(field + "=" + value)
                result[bucket] = result.get(bucket, 0.) + sign / len(values)
        return sorted((k, v) for k, v in result.items() if abs(v) > 1e-8)

    def transform(self, records):
        rows = [self.row(r) for r in records]
        width = max(1, max((len(r) for r in rows), default=0))
        indices = np.zeros((len(rows), width), dtype=np.int64)
        weights = np.zeros((len(rows), width), dtype=np.float32)
        for i, row in enumerate(rows):
            if row:
                indices[i, :len(row)], weights[i, :len(row)] = zip(*row)
        return indices, weights

    def to_dict(self):
        return {"schema_version": 1, "algorithm": "blake2b-8-recsys-hist-v1",
                "buckets": self.buckets, "seed": self.seed, "field_weight": "mean"}


def recent_history(history, id_to_index, max_events=256):
    """Select the last events BEFORE skipping unknown IDs. Preserve order/repeats."""
    return [id_to_index[str(t)] for t in list(history)[-max_events:]
            if str(t) in id_to_index]


def padded_histories(histories):
    width = max(1, max((len(h) for h in histories), default=0))
    indices = np.zeros((len(histories), width), dtype=np.int64)
    mask = np.zeros_like(indices, dtype=bool)
    for i, history in enumerate(histories):
        indices[i, :len(history)] = history
        mask[i, :len(history)] = True
    return indices, mask
