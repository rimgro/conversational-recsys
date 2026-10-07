"""Music4All-CRS -> EmbeddingGemma-2 -> LanceDB indexing + retrieval evaluation.

This is the current ("final data") pipeline. It indexes the 64,016-track corpus
from the delivered CRS dataset and evaluates retrieval against the synthetic
queries in the ``train`` split.

Architecture (per current decision)
-----------------------------------
* One **combined** branch for now: track text = metadata + lyrics
  (only-meta and only-lyrics branches are deferred).
* Encoder: ``google/embeddinggemma-2`` (768-d, mean pooling, task prefixes).
  - documents: ``title: {title} | text: {content}``
  - queries:   ``task: search result | query: {query}``
* Vectors are L2-normalised; LanceDB is built with cosine + IVF_HNSW_SQ.
* Evaluation: for every train positive, embed its query (+ optional user
  profile) and rank top-K corpus tracks in LanceDB; report Recall@K / nDCG@K /
  MRR overall and per ``query_family``.

Track text fields (must-have): title, artist, description, lyrics, tags,
album, language — plus genres, year and audio attributes.

Data layout (delivered dataset)::

    <dataset>/tracks_meta-00000-of-00015.parquet   # 64,016 tracks
    <dataset>/track_embeddings-*.parquet
    <dataset>/train-00000-of-00082.parquet         # users + positives with queries
    <dataset>/test_public-*.parquet
    <dataset>/test_private-*.parquet

Typical local smoke (no GPU, hashing embedder)::

    python gemma_crs_pipeline.py build-texts --dataset "<dir>" --work-dir data/crs
    python gemma_crs_pipeline.py embed   --work-dir data/crs --embedder hashing
    python gemma_crs_pipeline.py index   --work-dir data/crs
    python gemma_crs_pipeline.py evaluate --work-dir data/crs --dataset "<dir>" --max-users 50

On Kaggle / GPU use ``--embedder gemma`` (see ``kaggle/``).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import multiprocessing
import re
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

MODEL_NAME = "google/embeddinggemma-2"
EMBED_DIM = 768  # EmbeddingGemma-2 native output dimension
DOC_TEMPLATE = "title: {title} | text: {content}"
DOC_TEMPLATE_NO_TITLE = "title: none | text: {content}"
QUERY_TEMPLATE = "task: search result | query: {query}"

MAX_LYRICS_CHARS = 8000
MAX_TAGS = 20
MAX_TAG_CHARS = 40
MIN_TAG_CHARS = 2
MAX_TOTAL_TAG_CHARS = 300
MAX_DESCRIPTION_CHARS = 1200

_JUNK_SUBSTRINGS = (
    "http", "www.", "last.fm", "lastfm", "spotify", "youtube",
    "soundcloud", "bandcamp", "musicbrainz", "discogs",
)
_JUNK_PATTERNS = (
    re.compile(r"\b\d+\s*(?:of|/)\s*\d+\s*stars?\b", re.IGNORECASE),
    re.compile(r"^\d+$"),
    re.compile(r"^(?=.*\d)[0-9a-f]{16,}$", re.IGNORECASE),
    re.compile(r"^(?=[a-z0-9]{5,}$)[a-z]*\d[a-z0-9]*\d[a-z0-9]*$", re.IGNORECASE),
)

TRACK_COLUMNS = [
    "m4a_id", "spotify_id", "title", "m4a_song", "artist", "m4a_artist",
    "release", "m4a_album", "album_name", "release_year", "lang", "is_instrumental",
    "m4a_genres_full", "m4a_genres", "artist_genres", "album_genres", "spotify_genres",
    "m4a_tags_full", "m4a_tags", "tags", "album_tags", "lastfm_tag_weights",
    "lyrics", "lyrics_processed", "pseudo_caption", "artist_description",
    "album_description", "spotify_popularity", "onion_listens", "tempo", "energy",
    "valence", "danceability", "key", "mode", "duration_ms",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def log(msg: str) -> None:
    print(f"[crs] {msg}", flush=True)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def first_nonempty(*values: Any) -> str:
    for v in values:
        if v is None:
            continue
        if isinstance(v, str):
            if v.strip() and v.strip().lower() not in ("none", "null", "nan"):
                return v.strip()
        elif isinstance(v, (list, tuple)):
            if v:
                return ", ".join(str(x) for x in v if x)
        else:
            return str(v)
    return ""


def as_list(value: Any, sep: str = ",") -> list[str]:
    """Coerce str (comma separated) / list / None into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if x is not None and str(x).strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() in ("none", "null", "nan", "[]"):
            return []
        if s.startswith("{"):  # JSON dict of tag -> weight
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    return [str(k) for k in obj.keys()]
            except json.JSONDecodeError:
                pass
        return [t.strip() for t in s.split(sep) if t.strip()]
    return [str(value)]


# --------------------------------------------------------------------------- #
# Tag / text cleaning
# --------------------------------------------------------------------------- #


def normalize_tag(tag: str) -> str:
    tag = str(tag or "").strip().lower()
    tag = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", tag)
    tag = tag.replace("_", " ").replace("|", " ")
    return re.sub(r"\s+", " ", tag).strip(" ,;:-")


def is_junk_tag(tag: str) -> bool:
    if len(tag) < MIN_TAG_CHARS or len(tag) > MAX_TAG_CHARS:
        return True
    if any(s in tag for s in _JUNK_SUBSTRINGS):
        return True
    if any(p.search(tag) for p in _JUNK_PATTERNS):
        return True
    if not re.search(r"[a-z]", tag):
        return True
    letters = sum(c.isalpha() for c in tag)
    digits = sum(c.isdigit() for c in tag)
    return digits > letters


def clean_list(
    values: Sequence[str],
    *,
    max_items: int = MAX_TAGS,
    max_total_chars: int = MAX_TOTAL_TAG_CHARS,
    junk_filter: bool = True,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    total = 0
    for raw in values:
        item = normalize_tag(raw)
        if not item or item in seen:
            continue
        if junk_filter and is_junk_tag(item):
            continue
        seen.add(item)
        if len(out) >= max_items:
            break
        if total + len(item) + 2 > max_total_chars and out:
            break
        out.append(item)
        total += len(item) + 2
    return out


def _truncate(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + " ..."


def _num(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


_KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def key_name(key: Any, mode: Any = None) -> str:
    try:
        idx = int(round(float(key))) % 12
    except (TypeError, ValueError):
        return ""
    name = _KEYS[idx]
    try:
        name += " major" if int(float(mode)) == 1 else " minor"
    except (TypeError, ValueError):
        pass
    return name


# --------------------------------------------------------------------------- #
# Track record + text building
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class TrackRecord:
    id: str
    spotify_id: str = ""
    title: str = ""
    artist: str = ""
    album: str = ""
    year: str = ""
    lang: str = ""
    genres: list[str] = dataclasses.field(default_factory=list)
    tags: list[str] = dataclasses.field(default_factory=list)
    lyrics: str = ""
    description: str = ""
    artist_description: str = ""
    album_description: str = ""
    extra: dict[str, str] = dataclasses.field(default_factory=dict)
    document: str = ""

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def record_from_row(row: dict[str, Any]) -> TrackRecord:
    genres = clean_list(
        as_list(row.get("artist_genres"))
        + as_list(row.get("album_genres"))
        + as_list(row.get("m4a_genres_full"))
        + as_list(row.get("m4a_genres"))
        + as_list(row.get("spotify_genres")),
        junk_filter=False,
    )
    tags = clean_list(
        as_list(row.get("tags"))
        + as_list(row.get("m4a_tags_full"))
        + as_list(row.get("m4a_tags"))
        + as_list(row.get("album_tags"))
        + as_list(row.get("lastfm_tag_weights"))
    )
    lyrics = first_nonempty(row.get("lyrics"), row.get("lyrics_processed"))
    lang = first_nonempty(row.get("lang"))
    if lang.upper().startswith("INTRUMENT"):
        lang = "instrumental"
    rec = TrackRecord(
        id=str(row.get("m4a_id", "")),
        spotify_id=first_nonempty(row.get("spotify_id")),
        title=first_nonempty(row.get("title"), row.get("m4a_song")),
        artist=first_nonempty(row.get("artist"), row.get("m4a_artist")),
        album=first_nonempty(row.get("release"), row.get("m4a_album"), row.get("album_name")),
        year=first_nonempty(row.get("release_year")),
        lang=lang,
        genres=genres,
        tags=tags,
        lyrics=lyrics,
        description=first_nonempty(row.get("pseudo_caption")),
        artist_description=first_nonempty(row.get("artist_description")),
        album_description=first_nonempty(row.get("album_description")),
    )
    rec.extra = {
        "instrumental": "yes" if row.get("is_instrumental") else "no",
        "popularity": _num(row.get("spotify_popularity"), 1),
        "listeners": first_nonempty(row.get("onion_listens")),
        "tempo": _num(row.get("tempo"), 1),
        "energy": _num(row.get("energy"), 3),
        "valence": _num(row.get("valence"), 3),
        "danceability": _num(row.get("danceability"), 3),
        "key": key_name(row.get("key"), row.get("mode")),
        "duration": _duration_ms(row.get("duration_ms")),
    }
    rec.document = build_document(rec)
    return rec


def _duration_ms(value: Any) -> str:
    try:
        total_s = int(float(value) / 1000)
    except (TypeError, ValueError):
        return ""
    if total_s <= 0:
        return ""
    return f"{total_s // 60}:{total_s % 60:02d}"


def build_content(rec: TrackRecord) -> str:
    """Metadata + lyrics content (without the ``title:`` wrapper)."""
    lines: list[str] = []
    if rec.artist:
        lines.append(f"Artist: {rec.artist}")
    if rec.album:
        year = f" ({rec.year})" if rec.year else ""
        lines.append(f"Album: {rec.album}{year}")
    if rec.genres:
        lines.append("Genres: " + ", ".join(rec.genres))
    if rec.tags:
        lines.append("Tags: " + ", ".join(rec.tags))
    if rec.lang:
        lines.append(f"Language: {rec.lang}")
    if rec.description:
        lines.append("Description: " + _truncate(rec.description, MAX_DESCRIPTION_CHARS))
    if rec.artist_description:
        lines.append("Artist bio: " + _truncate(rec.artist_description, MAX_DESCRIPTION_CHARS))
    if rec.album_description:
        lines.append("Album info: " + _truncate(rec.album_description, MAX_DESCRIPTION_CHARS))

    audio = []
    if rec.extra.get("tempo"):
        audio.append(f"tempo {rec.extra['tempo']} BPM")
    if rec.extra.get("key"):
        audio.append(f"key {rec.extra['key']}")
    if rec.extra.get("energy"):
        audio.append(f"energy {rec.extra['energy']}")
    if rec.extra.get("valence"):
        audio.append(f"valence {rec.extra['valence']}")
    if rec.extra.get("danceability"):
        audio.append(f"danceability {rec.extra['danceability']}")
    if rec.extra.get("duration"):
        audio.append(f"duration {rec.extra['duration']}")
    audio.append(f"instrumental {rec.extra.get('instrumental', 'no')}")
    lines.append("Audio: " + "; ".join(audio))

    pop = []
    if rec.extra.get("popularity"):
        pop.append(f"popularity {rec.extra['popularity']}")
    if rec.extra.get("listeners"):
        pop.append(f"listeners {rec.extra['listeners']}")
    if pop:
        lines.append("Listeners: " + "; ".join(pop))

    if rec.lyrics:
        lines.append("Lyrics: " + _truncate(rec.lyrics, MAX_LYRICS_CHARS))
    return "\n".join(lines)


def build_document(rec: TrackRecord) -> str:
    content = build_content(rec)
    if rec.title:
        return DOC_TEMPLATE.format(title=rec.title, content=content)
    return DOC_TEMPLATE_NO_TITLE.format(content=content)


def build_query_text(query: str, user_profile: str = "") -> str:
    """Query side (SearchQuery prompt). Profile, if given, is prepended as context."""
    query = re.sub(r"\s+", " ", str(query or "")).strip()
    parts = []
    if user_profile:
        parts.append(re.sub(r"\s+", " ", user_profile).strip())
    parts.append(query)
    return QUERY_TEMPLATE.format(query="\n".join(parts))


# --------------------------------------------------------------------------- #
# Dataset readers
# --------------------------------------------------------------------------- #


def _parquet_files(dataset_dir: str | Path, prefix: str) -> list[Path]:
    return sorted(Path(dataset_dir).glob(f"{prefix}-*.parquet"))


def iter_track_rows(dataset_dir: str | Path, *, batch_size: int = 2048) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    files = _parquet_files(dataset_dir, "tracks_meta")
    if not files:
        raise FileNotFoundError(f"no tracks_meta-*.parquet in {dataset_dir}")
    for path in files:
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        cols = [c for c in TRACK_COLUMNS if c in available]
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
            yield from batch.to_pylist()


def iter_train_users(
    dataset_dir: str | Path, *, split: str = "train", batch_size: int = 512
) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    files = _parquet_files(dataset_dir, split)
    if not files:
        raise FileNotFoundError(f"no {split}-*.parquet in {dataset_dir}")
    cols = ["user_id", "history", "user_profile", "positives"]
    for path in files:
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        use = [c for c in cols if c in available]
        for batch in pf.iter_batches(batch_size=batch_size, columns=use):
            yield from batch.to_pylist()


# --------------------------------------------------------------------------- #
# Embedders
# --------------------------------------------------------------------------- #


class BaseEmbedder:
    dim: int = EMBED_DIM

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> "Any":
        raise NotImplementedError


class GemmaEmbedder(BaseEmbedder):
    """google/embeddinggemma-2 via sentence-transformers (mean pooling + L2)."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        *,
        device: str | None = None,
        dtype: str | None = None,
        max_seq_length: int = 2048,
        truncate_dim: int | None = None,
        normalize: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, Any] = {}
        if dtype:
            try:
                import torch

                mapping = {
                    "float16": torch.float16, "half": torch.float16,
                    "bfloat16": torch.bfloat16, "float32": torch.float32,
                }
                kwargs["model_kwargs"] = {"torch_dtype": mapping.get(str(dtype).lower(), dtype)}
            except Exception:  # pragma: no cover
                kwargs["model_kwargs"] = {"torch_dtype": dtype}
        log(f"loading {model_name} (device={device})")
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True, **kwargs)
        self.model.max_seq_length = max_seq_length
        self.normalize = normalize
        try:
            full = int(self.model.get_sentence_embedding_dimension())
        except Exception:  # pragma: no cover
            full = EMBED_DIM
        self.truncate_dim = truncate_dim
        self.dim = truncate_dim or full

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> "Any":
        import numpy as np

        vecs = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=False,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 512,
        )
        vecs = np.asarray(vecs, dtype="float32")
        nonfinite = ~np.isfinite(vecs).all(axis=1)
        if nonfinite.any():
            log(f"warning: {int(nonfinite.sum())}/{len(vecs)} embeddings had non-finite values -> zeroed")
            vecs = np.nan_to_num(vecs, nan=0.0, posinf=0.0, neginf=0.0)
        if self.truncate_dim:
            vecs = vecs[:, : self.truncate_dim]
        if self.normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        return vecs


class HashingEmbedder(BaseEmbedder):
    """Deterministic hashing embedder for offline tests / smoke (non-semantic)."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> "Any":
        import numpy as np

        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype("float32")


def get_embedder(name: str, **kwargs: Any) -> BaseEmbedder:
    if name == "gemma":
        kwargs.pop("hash_dim", None)
        return GemmaEmbedder(**kwargs)
    if name == "hashing":
        return HashingEmbedder(dim=int(kwargs.get("hash_dim", 256)))
    raise ValueError(f"unknown embedder {name!r}")


# --------------------------------------------------------------------------- #
# LanceDB
# --------------------------------------------------------------------------- #

VECTOR_COLUMN = "vector_combined"
META_COLUMNS = [
    "id", "spotify_id", "title", "artist", "album", "year", "lang",
    "genres", "tags", "document",
]


def _arrow_table(records: Sequence[TrackRecord], vectors: "Any", dim: int, *, store_text: bool) -> "Any":
    import numpy as np
    import pyarrow as pa

    cols: dict[str, Any] = {
        "id": [r.id for r in records],
        "spotify_id": [r.spotify_id for r in records],
        "title": [r.title for r in records],
        "artist": [r.artist for r in records],
        "album": [r.album for r in records],
        "year": [r.year for r in records],
        "lang": [r.lang for r in records],
        "genres": [", ".join(r.genres) for r in records],
        "tags": [", ".join(r.tags) for r in records],
    }
    if store_text:
        cols["document"] = [r.document for r in records]
    flat = np.asarray(vectors, dtype="float32").reshape(-1)
    cols[VECTOR_COLUMN] = pa.FixedSizeListArray.from_arrays(pa.array(flat), dim)
    return pa.table(cols)


def build_lancedb(
    db_path: str | Path,
    records: Sequence[TrackRecord],
    vectors: "Any",
    *,
    table_name: str = "tracks",
    metric: str = "cosine",
    index_type: str = "IVF_HNSW_SQ",
    store_text: bool = True,
    m: int = 20,
    ef_construction: int = 300,
    batch_rows: int = 4096,
    overwrite: bool = True,
) -> Any:
    import lancedb
    import numpy as np

    vectors = np.asarray(vectors, dtype="float32")
    nonfinite = ~np.isfinite(vectors).all(axis=1)
    if nonfinite.any():
        log(f"warning: {int(nonfinite.sum())} vectors had non-finite values -> zeroed before indexing")
        vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)

    dim = int(vectors.shape[1])
    db = lancedb.connect(str(db_path))
    if hasattr(db, "list_tables"):
        resp = db.list_tables()
        names = set(getattr(resp, "tables", resp))
    else:  # pragma: no cover
        names = set(db.table_names())
    if table_name in names:
        if not overwrite:
            return db.open_table(table_name)
        db.drop_table(table_name)

    n = len(records)
    rows = min(batch_rows, n)
    table = db.create_table(table_name, data=_arrow_table(records[:rows], vectors[:rows], dim, store_text=store_text))
    for start in range(batch_rows, n, batch_rows):
        table.add(_arrow_table(records[start : start + batch_rows], vectors[start : start + batch_rows], dim, store_text=store_text))
    log(f"LanceDB table {table_name!r}: {table.count_rows()} rows, dim={dim}")

    if index_type != "none":
        import warnings

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                table.create_index(
                    metric=metric,
                    index_type=index_type,
                    vector_column_name=VECTOR_COLUMN,
                    m=m,
                    ef_construction=ef_construction,
                    replace=True,
                )
            log(f"ANN index: {index_type} metric={metric}")
        except Exception as exc:  # pragma: no cover
            log(f"ANN index failed ({exc}); brute force only")
    return table


def open_table(db_path: str | Path, table_name: str = "tracks") -> Any:
    import lancedb

    db = lancedb.connect(str(db_path))
    return db.open_table(table_name)


def search_vectors(
    table: Any,
    query_vec: "Any",
    k: int = 100,
    *,
    nprobes: int = 200,
    columns: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    import numpy as np

    q = np.asarray(query_vec, dtype="float32").reshape(-1)
    builder = table.search(q, vector_column_name=VECTOR_COLUMN).metric("cosine")
    try:
        builder = builder.nprobes(nprobes)
    except Exception:
        pass
    if columns:
        builder = builder.select(list(columns))
    rows = builder.limit(k).to_list()
    for row in rows:
        row["score"] = 1.0 - float(row.pop("_distance", 0.0))
    return rows


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def _paths(work_dir: str | Path) -> dict[str, Path]:
    root = ensure_dir(work_dir)
    return {
        "root": root,
        "texts": root / "tracks_texts.jsonl",
        "vectors": ensure_dir(root / "vectors"),
        "lancedb": root / "lancedb",
        "manifest": root / "manifest.json",
    }


def cmd_build_texts(args: argparse.Namespace) -> None:
    p = _paths(args.work_dir)
    n = 0
    with open(p["texts"], "w", encoding="utf-8") as fh:
        for i, row in enumerate(iter_track_rows(args.dataset)):
            if args.limit and i >= args.limit:
                break
            rec = record_from_row(row)
            if not rec.id:
                continue
            fh.write(json.dumps(rec.to_json(), ensure_ascii=False) + "\n")
            n += 1
    log(f"wrote {n} track texts -> {p['texts']}")


def _iter_text_records(path: Path, limit: int | None = None) -> Iterator[TrackRecord]:
    fields = {f.name for f in dataclasses.fields(TrackRecord)}
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                break
            obj = json.loads(line)
            yield TrackRecord(**{k: v for k, v in obj.items() if k in fields})


def cmd_embed(args: argparse.Namespace) -> None:
    import numpy as np

    p = _paths(args.work_dir)
    if not p["texts"].exists():
        raise SystemExit(f"{p['texts']} not found; run build-texts first")
    embedder = get_embedder(
        args.embedder,
        device=args.device,
        dtype=args.dtype,
        max_seq_length=args.max_seq_length,
        truncate_dim=args.truncate_dim,
        hash_dim=args.hash_dim,
    )
    recs = list(_iter_text_records(p["texts"], limit=args.limit))
    log(f"embedding {len(recs)} documents with {args.embedder} (dim={embedder.dim})")
    ids: list[str] = []
    chunks: list[Any] = []
    for start in range(0, len(recs), args.shard_size):
        chunk = recs[start : start + args.shard_size]
        vecs = embedder.encode([r.document for r in chunk], batch_size=args.batch_size)
        np.savez(p["vectors"] / f"shard_{start:07d}.npz", ids=np.array([r.id for r in chunk]), vectors=vecs)
        ids.extend(r.id for r in chunk)
        chunks.append(vecs)
        log(f"  shard {start + len(chunk)}/{len(recs)}")
    all_vecs = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, embedder.dim), dtype="float32")
    np.savez(p["vectors"] / "all.npz", ids=np.array(ids), vectors=all_vecs)
    p["manifest"].write_text(
        json.dumps({"model": MODEL_NAME if args.embedder == "gemma" else "hashing",
                    "embedder": args.embedder, "dim": int(embedder.dim), "n": len(ids)}),
        encoding="utf-8",
    )
    log(f"done: {len(ids)} vectors (dim={embedder.dim})")


def _load_vectors(p: dict[str, Path]) -> tuple[list[str], "Any"]:
    import numpy as np

    merged = p["vectors"] / "all.npz"
    if merged.exists():
        data = np.load(merged, allow_pickle=True)
        return [str(x) for x in data["ids"]], data["vectors"]
    shards = sorted(p["vectors"].glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"no vectors found in {p['vectors']}; run embed first")
    ids: list[str] = []
    vecs: list[Any] = []
    for shard in shards:
        d = np.load(shard, allow_pickle=True)
        ids.extend(str(x) for x in d["ids"])
        vecs.append(d["vectors"])
    return ids, np.concatenate(vecs, axis=0)


def cmd_index(args: argparse.Namespace) -> None:
    p = _paths(args.work_dir)
    ids, vectors = _load_vectors(p)
    by_id = {r.id: r for r in _iter_text_records(p["texts"])}
    records = [by_id[i] for i in ids if i in by_id]
    if len(records) != len(ids):
        log(f"warning: {len(ids) - len(records)} ids missing from texts")
    build_lancedb(
        p["lancedb"], records, vectors,
        index_type=args.index_type, store_text=not args.no_store_text,
        m=args.m, ef_construction=args.ef_construction, overwrite=not args.no_overwrite,
    )


# ---- evaluation ---------------------------------------------------------- #


def _discount(rank: int) -> float:
    """nDCG discount for a single relevant item (rank is 0-based)."""
    import math

    return 1.0 / math.log2(rank + 2)


def evaluate(
    work_dir: str | Path,
    dataset_dir: str | Path,
    embedder: BaseEmbedder,
    *,
    k: int = 100,
    max_users: int | None = None,
    max_queries: int | None = None,
    with_profile: bool = False,
    batch_size: int = 64,
    nprobes: int = 200,
    ks: Sequence[int] = (1, 5, 10, 20, 100),
) -> dict[str, Any]:
    p = _paths(work_dir)
    table = open_table(p["lancedb"])

    # Collect (query_text, gt_id, family).
    queries: list[str] = []
    gts: list[str] = []
    families: list[str] = []
    n_users = 0
    for user in iter_train_users(dataset_dir, split="train"):
        if max_users is not None and n_users >= max_users:
            break
        n_users += 1
        profile = str(user.get("user_profile") or "") if with_profile else ""
        for pos in user.get("positives") or []:
            q = pos.get("query")
            gt = pos.get("m4a_id")
            if not q or not gt:
                continue
            queries.append(build_query_text(q, profile))
            gts.append(str(gt))
            families.append(pos.get("query_family") or "unknown")
            if max_queries is not None and len(queries) >= max_queries:
                break
        if max_queries is not None and len(queries) >= max_queries:
            break

    log(f"evaluating {len(queries)} queries over {n_users} users (with_profile={with_profile})")
    hits: dict[int, list[float]] = {kk: [] for kk in ks}
    ndcg: dict[int, list[float]] = {kk: [] for kk in ks}
    mrr: list[float] = []
    fam_stats: dict[str, dict[str, float]] = {}
    topk = max([k, *ks])

    for start in range(0, len(queries), batch_size):
        batch_q = queries[start : start + batch_size]
        qvecs = embedder.encode(batch_q, batch_size=batch_size)
        for j, qvec in enumerate(qvecs):
            idx = start + j
            gt = gts[idx]
            fam = families[idx]
            rows = search_vectors(table, qvec, k=topk, nprobes=nprobes, columns=["id"])
            ranked = [r["id"] for r in rows]
            rank = ranked.index(gt) if gt in ranked else None
            for kk in ks:
                hits[kk].append(1.0 if (rank is not None and rank < kk) else 0.0)
                ndcg[kk].append(_discount(rank) if (rank is not None and rank < kk) else 0.0)
            mrr.append(1.0 / (rank + 1) if rank is not None else 0.0)
            st = fam_stats.setdefault(fam, {"n": 0.0, "r20": 0.0, "r100": 0.0, "mrr": 0.0})
            st["n"] += 1
            st["r20"] += 1.0 if (rank is not None and rank < 20) else 0.0
            st["r100"] += 1.0 if rank is not None else 0.0
            st["mrr"] += 1.0 / (rank + 1) if rank is not None else 0.0

    n = max(1, len(queries))
    result: dict[str, Any] = {
        "n_queries": len(queries),
        "n_users": n_users,
        "with_profile": with_profile,
        "recall": {f"@{kk}": sum(hits[kk]) / n for kk in ks},
        "ndcg": {f"@{kk}": sum(ndcg[kk]) / n for kk in ks},
        "mrr": sum(mrr) / n,
        "by_family": {
            fam: {
                "n": int(st["n"]),
                "recall@20": st["r20"] / max(1, st["n"]),
                "recall@100": st["r100"] / max(1, st["n"]),
                "mrr": st["mrr"] / max(1, st["n"]),
            }
            for fam, st in sorted(fam_stats.items())
        },
    }
    return result


def cmd_evaluate(args: argparse.Namespace) -> None:
    p = _paths(args.work_dir)
    embedder = get_embedder(
        args.embedder,
        device=args.device,
        dtype=args.dtype,
        max_seq_length=args.max_seq_length,
        truncate_dim=args.truncate_dim,
        hash_dim=args.hash_dim,
    )
    result = evaluate(
        p["root"], args.dataset, embedder,
        k=args.k, max_users=args.max_users, max_queries=args.max_queries,
        with_profile=args.with_profile, batch_size=args.batch_size,
    )
    out = p["root"] / f"eval_{'profile' if args.with_profile else 'query'}_{args.embedder}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"wrote {out}")
    print(json.dumps({k: result[k] for k in ("n_queries", "recall", "ndcg", "mrr")}, ensure_ascii=False, indent=2))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Music4All-CRS -> EmbeddingGemma-2 -> LanceDB")
    parser.add_argument("--work-dir", default="data/crs")
    parser.add_argument("--dataset", default="hnsw/CRS dataset-20261007T113227Z-1-001/CRS dataset")
    sub = parser.add_subparsers(dest="command", required=True)

    p_bt = sub.add_parser("build-texts")
    p_bt.add_argument("--limit", type=int, default=None)

    p_emb = sub.add_parser("embed")
    p_emb.add_argument("--embedder", choices=["gemma", "hashing"], default="gemma")
    p_emb.add_argument("--device", default=None)
    p_emb.add_argument("--dtype", default=None)
    p_emb.add_argument("--max-seq-length", type=int, default=2048)
    p_emb.add_argument("--truncate-dim", type=int, default=None)
    p_emb.add_argument("--hash-dim", type=int, default=256)
    p_emb.add_argument("--batch-size", type=int, default=64)
    p_emb.add_argument("--shard-size", type=int, default=10000)
    p_emb.add_argument("--limit", type=int, default=None)

    p_ix = sub.add_parser("index")
    p_ix.add_argument("--index-type", default="IVF_HNSW_SQ")
    p_ix.add_argument("--m", type=int, default=20)
    p_ix.add_argument("--ef-construction", type=int, default=300)
    p_ix.add_argument("--no-store-text", action="store_true")
    p_ix.add_argument("--no-overwrite", action="store_true")

    p_ev = sub.add_parser("evaluate")
    p_ev.add_argument("--embedder", choices=["gemma", "hashing"], default="gemma")
    p_ev.add_argument("--device", default=None)
    p_ev.add_argument("--dtype", default=None)
    p_ev.add_argument("--max-seq-length", type=int, default=2048)
    p_ev.add_argument("--truncate-dim", type=int, default=None)
    p_ev.add_argument("--hash-dim", type=int, default=256)
    p_ev.add_argument("--batch-size", type=int, default=64)
    p_ev.add_argument("--k", type=int, default=100)
    p_ev.add_argument("--max-users", type=int, default=None)
    p_ev.add_argument("--max-queries", type=int, default=None)
    p_ev.add_argument("--with-profile", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = {
        "build-texts": cmd_build_texts,
        "embed": cmd_embed,
        "index": cmd_index,
        "evaluate": cmd_evaluate,
    }[args.command]
    handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
