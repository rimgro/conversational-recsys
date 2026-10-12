"""Five-field Transformer candidate tower; no user/query-dependent item features.

Text/audio encoders run before this module. Categories and feature projections
are learned jointly with the query tower. Fitted preprocessing is JSON portable.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

NUMERIC_FIELDS = ("danceability", "energy", "valence", "tempo", "duration_ms", "release_year")
FIELD_NAMES = ("audio", "card", "lyrics", "categories", "numbers")
EMPTY_TEXT = {"", "nan", "none", "null", "unknown", "n/a"}


def text_value(value):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    if isinstance(value, (list, tuple, np.ndarray)):
        return ", ".join(text_value(v) for v in value if text_value(v))
    text = " ".join(str(value).split())
    return "" if text.casefold() in EMPTY_TEXT else text


def category_tokens(row, max_categories=48):
    """Keep field identities; normalize and deduplicate deterministic tokens."""
    result = []
    for field in ("lang", "key", "mode", "is_instrumental"):
        value = text_value(row.get(field)).casefold()
        if value and field in ("key", "mode"):
            try:
                number = float(value)
                maximum = 11 if field == "key" else 1
                value = str(int(number)) if number.is_integer() and 0 <= number <= maximum else ""
            except ValueError:
                value = ""
        elif field == "is_instrumental":
            value = {"true": "true", "1": "true", "1.0": "true",
                     "false": "false", "0": "false", "0.0": "false"}.get(value, "")
        if value:
            result.append(field + "=" + value)
    for field, limit in (("m4a_genres_full", 8), ("m4a_tags_full", 32)):
        raw = row.get(field)
        parts = raw if isinstance(raw, (list, tuple, np.ndarray)) else text_value(raw).split(",")
        values = sorted({text_value(v).casefold() for v in parts if text_value(v)})
        result.extend(field + "=" + v for v in values[:limit])
    return result[:max_categories]


def numeric_value(field, value):
    raw = text_value(value)
    if not raw:
        return float("nan")
    try:
        if field == "release_year":
            match = re.search(r"\b(18\d{2}|19\d{2}|20\d{2}|21\d{2})\b", raw)
            return float(match[0]) if match else float("nan")
        number = float(raw)
    except (ValueError, TypeError):
        return float("nan")
    if not math.isfinite(number):
        return float("nan")
    if field in ("danceability", "energy", "valence") and not 0 <= number <= 1:
        return float("nan")
    if field in ("tempo", "duration_ms") and number <= 0:
        return float("nan")
    return math.log1p(number / 1000) if field == "duration_ms" else number


def lyric_chunks(lyrics, words=128, overlap=32, max_chunks=8):
    if words <= 0 or not 0 <= overlap < words or max_chunks < 1:
        raise ValueError("Invalid lyric chunk settings")
    tokens = text_value(lyrics).split()
    if not tokens:
        return []
    chunks = []
    for start in range(0, len(tokens), words - overlap):
        chunks.append(" ".join(tokens[start:start + words]))
        if start + words >= len(tokens):
            break
    if len(chunks) > max_chunks:
        indices = np.linspace(0, len(chunks) - 1, max_chunks).round().astype(int)
        chunks = [chunks[i] for i in indices]
    return chunks


def card_text(row):
    artist = text_value(row.get("m4a_artist")) or text_value(row.get("artist"))
    title = text_value(row.get("m4a_song")) or text_value(row.get("title"))
    tags = text_value(row.get("m4a_tags_full"))
    raw = row.get("lastfm_tag_weights")
    if text_value(raw):
        try:
            weights = raw if isinstance(raw, dict) else json.loads(raw)
            if isinstance(weights, dict):
                keys = sorted(weights, key=lambda k: float(weights[k]), reverse=True)[:15]
                tags += ", " + ", ".join(keys)
        except (ValueError, TypeError):
            pass
    return (f"Title: {title}. Artist: {artist}. "
            f"Genres: {text_value(row.get('m4a_genres_full'))}. Tags: {tags}. "
            f"Description: {text_value(row.get('pseudo_caption'))}")[:1600]


def normalize_rows(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def pool_lyric_chunks(chunk_vectors, row_indices, row_count):
    vectors = np.zeros((row_count, chunk_vectors.shape[1]), dtype=np.float32)
    counts = np.zeros(row_count, dtype=np.float32)
    if len(row_indices):
        np.add.at(vectors, row_indices, chunk_vectors)
        np.add.at(counts, row_indices, 1)
    vectors /= np.maximum(counts[:, None], 1)
    return normalize_rows(vectors)


class TrackPreprocessor:
    """Fit on known catalog metadata only; never uses queries or target labels."""
    def __init__(self, max_categories=48):
        self.max_categories = int(max_categories)
        self.vocabulary = {"<PAD>": 0, "<UNK>": 1}
        self.center = np.zeros(len(NUMERIC_FIELDS), dtype=np.float32)
        self.scale = np.ones(len(NUMERIC_FIELDS), dtype=np.float32)

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError("Empty catalog")
        tokens = sorted({t for r in records for t in category_tokens(r, self.max_categories)})
        self.vocabulary = {"<PAD>": 0, "<UNK>": 1, **{t: i + 2 for i, t in enumerate(tokens)}}
        for i, field in enumerate(NUMERIC_FIELDS):
            values = np.array([numeric_value(field, r.get(field)) for r in records])
            finite = values[np.isfinite(values)]
            if len(finite):
                self.center[i] = np.median(finite)
                q1, q3 = np.quantile(finite, [.25, .75])
                self.scale[i] = max(float(q3 - q1), 1e-3)
        return self

    def transform(self, records):
        records = list(records)
        ids = np.zeros((len(records), self.max_categories), dtype=np.int64)
        values = np.zeros((len(records), len(NUMERIC_FIELDS)), dtype=np.float32)
        observed = np.zeros_like(values)
        for i, row in enumerate(records):
            tokens = category_tokens(row, self.max_categories)
            ids[i, :len(tokens)] = [self.vocabulary.get(t, 1) for t in tokens]
            for j, field in enumerate(NUMERIC_FIELDS):
                number = numeric_value(field, row.get(field))
                if math.isfinite(number):
                    values[i, j] = np.clip((number - self.center[j]) / self.scale[j], -5, 5)
                    observed[i, j] = 1
        numbers = np.concatenate([values, observed], axis=1)
        present = np.stack([np.any(ids != 0, axis=1), np.any(observed != 0, axis=1)], axis=1)
        return ids, numbers, present

    def to_dict(self):
        return {"schema_version": 1, "max_categories": self.max_categories,
                "numeric_fields": list(NUMERIC_FIELDS), "vocabulary": self.vocabulary,
                "center": self.center.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, data):
        if data["schema_version"] != 1 or data["numeric_fields"] != list(NUMERIC_FIELDS):
            raise ValueError("Preprocessing schema mismatch")
        obj = cls(data["max_categories"])
        obj.vocabulary = data["vocabulary"]
        obj.center = np.array(data["center"], dtype=np.float32)
        obj.scale = np.array(data["scale"], dtype=np.float32)
        if not np.isfinite(obj.center).all() or not np.all(obj.scale > 0):
            raise ValueError("Invalid preprocessing statistics")
        return obj

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf8")


def feature_slices(text_dim, max_categories):
    lengths = (128, text_dim, text_dim, max_categories, 2 * len(NUMERIC_FIELDS), 5)
    boundaries = np.cumsum((0,) + lengths).tolist()
    return {name: slice(a, b) for name, a, b in zip(
        FIELD_NAMES + ("present",), boundaries[:-1], boundaries[1:])}


def pack_features(audio, cards, lyrics, categories, numbers, present):
    arrays = [np.asarray(v, dtype=np.float32) for v in (audio, cards, lyrics, categories, numbers, present)]
    if any(v.ndim != 2 for v in arrays) or len({len(v) for v in arrays}) != 1:
        raise ValueError("Expected aligned two-dimensional arrays")
    if arrays[0].shape[1] != 128 or arrays[1].shape[1] != arrays[2].shape[1]:
        raise ValueError("Wrong audio/text dimensions")
    if arrays[4].shape[1] != 2 * len(NUMERIC_FIELDS) or arrays[5].shape[1] != 5:
        raise ValueError("Wrong numeric/presence dimensions")
    if not all(np.isfinite(v).all() for v in arrays):
        raise ValueError("Nonfinite features")
    if not np.all(arrays[3] == np.floor(arrays[3])) or np.any(arrays[3] < 0):
        raise ValueError("Category IDs must be nonnegative integers")
    if not np.all((arrays[5] == 0) | (arrays[5] == 1)):
        raise ValueError("Presence flags must be boolean")
    return np.concatenate(arrays, axis=1)


class TrackTransformerTower(nn.Module):
    def __init__(self, text_dim, num_categories, max_categories=48, token_dim=128,
                 embedding_dim=256, layers=2, heads=4, feedforward_dim=512, dropout=.1):
        super().__init__()
        if token_dim % heads:
            raise ValueError("token_dim must be divisible by heads")
        self.layout = feature_slices(text_dim, max_categories)
        self.input_dim = self.layout["present"].stop
        self.category_embedding = nn.Embedding(num_categories, token_dim, padding_idx=0)
        # An unseen category has no learned semantic information; start UNK neutral.
        with torch.no_grad():
            self.category_embedding.weight[1].zero_()
        def projection(size):
            return nn.Sequential(nn.LayerNorm(size), nn.Linear(size, token_dim))
        self.audio = projection(128)
        self.card = projection(text_dim)
        self.lyrics = projection(text_dim)
        self.numbers = projection(2 * len(NUMERIC_FIELDS))
        self.field_embedding = nn.Parameter(torch.empty(5, token_dim))
        nn.init.normal_(self.field_embedding, std=.02)
        self.token_norm = nn.LayerNorm(token_dim)
        layer = nn.TransformerEncoderLayer(token_dim, heads, feedforward_dim, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        # TransformerEncoder clones identical initial layers: initialize each separately.
        for module in self.transformer.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.zeros_(module.in_proj_bias)
        self.output = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, embedding_dim))

    def forward(self, features):
        category_ids = features[:, self.layout["categories"]].long()
        cat_mask = category_ids.ne(0)
        cat = (self.category_embedding(category_ids) * cat_mask.unsqueeze(-1)).sum(1)
        cat = cat / cat_mask.sum(1, keepdim=True).clamp_min(1)
        tokens = torch.stack((self.audio(features[:, self.layout["audio"]]),
                              self.card(features[:, self.layout["card"]]),
                              self.lyrics(features[:, self.layout["lyrics"]]), cat,
                              self.numbers(features[:, self.layout["numbers"]])), dim=1)
        present = features[:, self.layout["present"]].gt(.5)
        # Replace absent blocks before attention: arbitrary missing values cannot leak.
        tokens = torch.where(present.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        tokens = self.token_norm(tokens + self.field_embedding.unsqueeze(0))
        # A dummy audio slot prevents NaNs when an entirely empty item is supplied.
        fallback = ~present.any(dim=1, keepdim=True)
        first = torch.arange(5, device=features.device).eq(0).unsqueeze(0)
        active = present | (fallback & first)
        contextual = self.transformer(tokens, src_key_padding_mask=~active)
        pooled = (contextual * active.unsqueeze(-1)).sum(1) / active.sum(1, keepdim=True)
        return F.normalize(self.output(pooled), dim=-1)


class TransformerTwoTower(nn.Module):
    def __init__(self, text_dim, user_dim, item_dim, hidden_dim, embedding_dim, dropout,
                 num_categories, max_categories=48, token_dim=128, layers=2, heads=4,
                 feedforward_dim=512):
        super().__init__()
        self.query_net = nn.Sequential(nn.LayerNorm(text_dim + user_dim),
                                       nn.Linear(text_dim + user_dim, hidden_dim), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(hidden_dim, embedding_dim))
        self.item_net = TrackTransformerTower(text_dim, num_categories, max_categories, token_dim,
                                             embedding_dim, layers, heads, feedforward_dim, dropout)
        if item_dim != self.item_net.input_dim:
            raise ValueError("Packed feature schema mismatch")

    def query(self, text, user):
        return F.normalize(self.query_net(torch.cat([text, user], dim=1)), dim=1)

    def item(self, features):
        return self.item_net(features)


def export_item_tower(tower, example):
    """Trace the eval tower without the native MHA inference fast path.

    Older CPU-only PyTorch has no MHA switch and changes graphs between trace
    and its automatic no-grad check. There we validate outputs explicitly;
    callers should also check the reloaded artifact with other batch/mask shapes.
    """
    tower.eval()
    backend = getattr(torch.backends, "mha", None)
    previous = backend.get_fastpath_enabled() if backend is not None else None
    if backend is not None:
        backend.set_fastpath_enabled(False)
    try:
        script = torch.jit.trace(tower, example, check_trace=backend is not None)
        with torch.inference_mode():
            torch.testing.assert_close(script(example), tower(example), atol=2e-5, rtol=2e-5)
        return script
    finally:
        if backend is not None:
            backend.set_fastpath_enabled(previous)
