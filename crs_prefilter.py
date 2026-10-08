"""LanceDB pre-filtering with diffusion-LM extracted tags.

Architecture (JEV-like "structured read" on a small masked-diffusion LM):

1. Offline: classify every track against a 63-tag taxonomy.
   - LM tags (mood / theme / vocals): one forward pass per (track, tag) with the
     answer slot set to ``<|mask|>``; read P(Yes) vs P(No) from the logits
     (single denoising step, no text generation). Keep tags with P(Yes) > 0.8.
   - Metadata tags (genre, language, era, instruments, tempo/energy): derived
     deterministically from ``tracks_meta``.
   - Persist as ``|tag1|tag2|`` in the LanceDB column ``extracted_tags``.
2. Online: extract tags from the user query with the same classifier, build a
   SQL pre-filter (``extracted_tags LIKE '%|tag|%'``), then run the ANN search
   only over matching rows, or rank by tag overlap (tags-only), or plain ANN.

The diffusion classifier defaults to ``dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1``
(masked diffusion, Qwen3-0.6B backbone). It is a drop-in stand-in for
DiffusionGemma-26B, which needs an H100/Ampere+ vLLM build and cannot run on
Kaggle T4x2 (Turing; no bf16/fp8, no diffusion vLLM image). On capable hardware
swap ``--tag-model`` and the vLLM structured-read endpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------------- #
# Tag taxonomy (63 tags)
# --------------------------------------------------------------------------- #

MOOD_TAGS = {
    "sad": ["sad", "sorrow", "tears", "cry", "lonely", "miss you", "goodbye"],
    "happy": ["happy", "joy", "smile", "sunshine", "celebration", "good time"],
    "energetic": ["energetic", "energy", "pump", "adrenaline", "workout"],
    "calm": ["calm", "peaceful", "quiet", "gentle", "relax", "serene"],
    "aggressive": ["aggressive", "anger", "fight", "rage", "heavy", "brutal"],
    "melancholic": ["melancholy", "melancholic", "wistful", "bittersweet"],
    "dark": ["dark", "shadow", "night", "sinister", "haunting", "gloom"],
    "uplifting": ["uplifting", "inspir", "rise up", "triumph", "hopeful"],
    "romantic": ["romantic", "romance", "love you", "kiss", "heart"],
    "angry": ["angry", "furious", "hate", "riot"],
    "dreamy": ["dream", "dreamy", "ethereal", "floating", "hazy"],
    "nostalgic": ["nostalg", "memories", "remember", "yesterday", "past"],
    "hopeful": ["hope", "hopeful", "brighter", "tomorrow"],
    "tense": ["tense", "tension", "anxious", "paranoi", "nervous"],
}
THEME_TAGS = {
    "love": ["love", "lover", "beloved"],
    "heartbreak": ["heartbreak", "broken heart", "breakup", "left me"],
    "party": ["party", "dance floor", "club", "celebrate"],
    "nature": ["nature", "ocean", "river", "mountain", "sky", "forest", "rain"],
    "protest": ["protest", "revolution", "resist", "injustice", "fight the power"],
    "introspection": ["myself", "inside", "my mind", "reflection", "who am i"],
    "freedom": ["free", "freedom", "escape", "run away"],
    "loneliness": ["alone", "lonely", "loneliness", "isolat"],
}
VOCAL_TAGS = {
    "male vocals": ["he ", "his ", "him ", "man", "boy"],
    "female vocals": ["she ", "her ", "woman", "girl"],
    "instrumental": [],
}
LM_TAGS = list(MOOD_TAGS) + list(THEME_TAGS) + list(VOCAL_TAGS)

GENRE_KEYWORDS = {
    "rock": ["rock"],
    "pop": ["pop"],
    "electronic": ["electronic", "electro", "edm", "techno", "house", "synth"],
    "hip hop": ["hip hop", "hip-hop", "rap", "trap"],
    "jazz": ["jazz", "swing", "bebop"],
    "metal": ["metal", "metalcore", "doom", "thrash"],
    "classical": ["classical", "orchestra", "symphon", "baroque", "opera"],
    "folk": ["folk", "acoustic", "singer-songwriter"],
    "country": ["country", "americana", "bluegrass"],
    "r&b": ["r&b", "rnb", "soul", "funk"],
    "reggae": ["reggae", "ska", "dub"],
    "punk": ["punk", "hardcore"],
    "indie": ["indie", "alternative"],
    "blues": ["blues"],
}
LANGUAGE_MAP = {
    "en": "english", "eng": "english",
    "ru": "russian", "rus": "russian",
    "es": "spanish", "spa": "spanish",
    "fr": "french", "fra": "french",
    "de": "german", "ger": "german", "deu": "german",
}
INSTRUMENT_KEYWORDS = {
    "guitar": ["guitar", "riff", "acoustic guitar"],
    "piano": ["piano", "keys", "keyboard"],
    "drums": ["drum", "percussion", "beat"],
    "synth": ["synth", "synthesizer", "moog"],
    "saxophone": ["sax", "saxophone"],
    "strings": ["violin", "cello", "strings", "orchestra"],
    "bass": ["bass"],
    "electronic beats": ["electronic beat", "drum machine", "808", "beat"],
}
ERA_TAGS = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"]
ENERGY_TAGS = ["fast tempo", "slow tempo", "high energy", "low energy"]

ALL_TAGS = (
    LM_TAGS
    + list(GENRE_KEYWORDS)
    + list(LANGUAGE_MAP.values())
    + ["other language"]
    + ERA_TAGS
    + list(INSTRUMENT_KEYWORDS)
    + ENERGY_TAGS
)
# de-duplicate while preserving order
ALL_TAGS = list(dict.fromkeys(ALL_TAGS))

TAG_GROUP = {}
for _t in MOOD_TAGS:
    TAG_GROUP[_t] = "mood"
for _t in THEME_TAGS:
    TAG_GROUP[_t] = "theme"
for _t in VOCAL_TAGS:
    TAG_GROUP[_t] = "vocal"
for _t in GENRE_KEYWORDS:
    TAG_GROUP[_t] = "genre"
for _t in set(LANGUAGE_MAP.values()) | {"other language"}:
    TAG_GROUP[_t] = "language"
for _t in ERA_TAGS:
    TAG_GROUP[_t] = "era"
for _t in INSTRUMENT_KEYWORDS:
    TAG_GROUP[_t] = "instrument"
for _t in ENERGY_TAGS:
    TAG_GROUP[_t] = "energy"

QUESTION_TEMPLATES = {
    "mood": 'Is the overall mood of this track best described as "{tag}"?',
    "theme": 'Are the lyrics of this track primarily about "{tag}"?',
    "vocal": 'Does this track have "{tag}"?',
}


def log(msg: str) -> None:
    print(f"[prefilter] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Metadata-derived tags
# --------------------------------------------------------------------------- #


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower())


def derived_tags(record: dict[str, Any]) -> list[str]:
    """Deterministic tags from tracks_meta (genre/language/era/instruments/energy)."""
    blob = " ".join(
        _norm(record.get(k))
        for k in (
            "m4a_genres_full", "m4a_genres", "m4a_tags_full", "m4a_tags",
            "artist_genres", "album_genres", "spotify_genres", "tags",
            "pseudo_caption",
        )
    )
    tags: list[str] = []

    for tag, kws in GENRE_KEYWORDS.items():
        if any(k in blob for k in kws):
            tags.append(tag)

    lang = _norm(record.get("lang")) or _norm(record.get("spotify2_language"))
    lang_tag = LANGUAGE_MAP.get(lang)
    if lang.startswith("intrument") or record.get("is_instrumental"):
        lang_tag = None
    tags.append(lang_tag or ("other language" if lang else "other language"))

    try:
        year = int(float(record.get("release_year")))
        decade = (year // 10) * 10
        era = f"{decade}s"
        if era in ERA_TAGS:
            tags.append(era)
    except (TypeError, ValueError):
        pass

    for tag, kws in INSTRUMENT_KEYWORDS.items():
        if any(k in blob for k in kws):
            tags.append(tag)

    try:
        tempo = float(record.get("tempo"))
        if tempo >= 130:
            tags.append("fast tempo")
        elif 0 < tempo <= 85:
            tags.append("slow tempo")
    except (TypeError, ValueError):
        pass
    try:
        energy = float(record.get("energy"))
        if energy >= 0.7:
            tags.append("high energy")
        elif energy <= 0.35:
            tags.append("low energy")
    except (TypeError, ValueError):
        pass

    # de-dup, keep taxonomy order
    order = {t: i for i, t in enumerate(ALL_TAGS)}
    tags = sorted(set(tags), key=lambda t: order.get(t, 10**6))
    return tags


def track_lm_text(record: dict[str, Any], max_chars: int = 1200) -> str:
    """Short track description used by the diffusion tag classifier."""
    parts = []
    title = record.get("title") or record.get("m4a_song") or ""
    artist = record.get("artist") or record.get("m4a_artist") or ""
    if title:
        parts.append(f"Title: {title}")
    if artist:
        parts.append(f"Artist: {artist}")
    for label, key in (
        ("Album", "release"), ("Year", "release_year"),
        ("Genres", "m4a_genres_full"), ("Tags", "m4a_tags_full"),
        ("Caption", "pseudo_caption"),
    ):
        val = record.get(key)
        if val:
            parts.append(f"{label}: {val}")
    lyrics = record.get("lyrics") or ""
    if lyrics:
        parts.append("Lyrics: " + _norm(lyrics)[: max_chars // 2])
    return "\n".join(parts)[:max_chars]


# --------------------------------------------------------------------------- #
# Classifiers
# --------------------------------------------------------------------------- #


class BaseTagger:
    def classify(self, texts: Sequence[str], tags: Sequence[str]) -> "Any":
        """Return a [len(texts) x len(tags)] matrix of P(tag | text)."""
        raise NotImplementedError


class LexiconTagger(BaseTagger):
    """Keyword tagger (no LM). Used for tests and as an offline fallback."""

    def __init__(self) -> None:
        self._kw: dict[str, list[str]] = {}
        self._kw.update(MOOD_TAGS)
        self._kw.update(THEME_TAGS)
        self._kw.update({t: [t] for t in VOCAL_TAGS})

    def classify(self, texts: Sequence[str], tags: Sequence[str]) -> "Any":
        import numpy as np

        out = np.zeros((len(texts), len(tags)), dtype="float32")
        for i, text in enumerate(texts):
            blob = _norm(text)
            for j, tag in enumerate(tags):
                kws = self._kw.get(tag, [tag])
                if kws and any(k in blob for k in kws):
                    out[i, j] = 1.0
        return out


class DiffusionTagClassifier(BaseTagger):
    """JEV-like structured read on a masked-diffusion LM.

    One forward pass per (text, tag): the answer slot after ``Answer:`` is the
    ``<|mask|>`` token; P(Yes) / P(No) are read from the logits at that position.
    No text generation, no multi-step denoising.
    """

    def __init__(
        self,
        model_name: str = "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
        *,
        device: str | None = None,
        dtype: str = "float16",
        max_length: int = 512,
        enable_thinking: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.torch = torch
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if device and str(device).startswith("cuda"):
            kwargs["torch_dtype"] = dtype_map.get(dtype, torch.float16)
        log(f"loading tag classifier {model_name} (device={device})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, **kwargs)
        if device:
            self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        self.max_length = max_length
        self.enable_thinking = enable_thinking
        self.mask_id = self.tokenizer.mask_token_id
        if self.mask_id is None:
            self.mask_id = self.tokenizer.convert_tokens_to_ids("<|mask|>")
        if self.mask_id is None or self.mask_id < 0:
            raise RuntimeError("mask token not found in tokenizer")
        self.yes_id = self._first_token_id(" Yes")
        self.no_id = self._first_token_id(" No")

    def _first_token_id(self, text: str) -> int:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return int(ids[0])

    def _prompt(self, text: str, question: str) -> str:
        messages = [
            {"role": "system", "content": "You are a precise music tag classifier. Answer only Yes or No."},
            {"role": "user", "content": f"{text}\n\nQuestion: {question} Answer Yes or No."},
        ]
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt + "Answer:"

    def classify(self, texts: Sequence[str], tags: Sequence[str], *, batch_size: int = 128) -> "Any":
        import numpy as np

        questions = [QUESTION_TEMPLATES[TAG_GROUP[t]].format(tag=t) for t in tags]
        out = np.zeros((len(texts), len(tags)), dtype="float32")
        prompts: list[str] = []
        index: list[tuple[int, int]] = []
        for i, text in enumerate(texts):
            if not str(text).strip():
                continue
            for j, q in enumerate(questions):
                prompts.append(self._prompt(text, q))
                index.append((i, j))
        if not prompts:
            return out
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            probs = self._forward(batch)
            for k, p in enumerate(probs):
                i, j = index[start + k]
                out[i, j] = p
        return out

    def _forward(self, prompts: Sequence[str]) -> "Any":
        import numpy as np

        enc = self.tokenizer(
            list(prompts), return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_length, add_special_tokens=False,
        )
        mask_col = self.torch.full((len(prompts), 1), self.mask_id, dtype=enc["input_ids"].dtype)
        input_ids = self.torch.cat([enc["input_ids"], mask_col], dim=1)
        attention = self.torch.cat(
            [enc["attention_mask"], self.torch.ones((len(prompts), 1), dtype=enc["attention_mask"].dtype)],
            dim=1,
        )
        if self.device:
            input_ids = input_ids.to(self.device)
            attention = attention.to(self.device)
        with self.torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention).logits[:, -1, :]
        yes = logits[:, self.yes_id].float()
        no = logits[:, self.no_id].float()
        probs = self.torch.softmax(self.torch.stack([yes, no], dim=1), dim=1)[:, 0]
        return probs.cpu().numpy()


def get_tagger(name: str, **kwargs: Any) -> BaseTagger:
    if name == "diffusion":
        return DiffusionTagClassifier(**kwargs)
    if name == "lexicon":
        return LexiconTagger()
    raise ValueError(f"unknown tagger {name!r}")


# --------------------------------------------------------------------------- #
# Tag extraction helpers
# --------------------------------------------------------------------------- #


def tags_to_string(tags: Iterable[str]) -> str:
    return "|" + "|".join(sorted(set(tags))) + "|"


def string_to_tags(value: str) -> list[str]:
    return [t for t in str(value or "").split("|") if t]


def extract_track_tags(
    records: Sequence[dict[str, Any]],
    tagger: BaseTagger,
    *,
    lm_tags: Sequence[str] = LM_TAGS,
    threshold: float = 0.8,
    batch_size: int = 8,
) -> dict[str, str]:
    """Offline: LM tags (mood/theme/vocal) + deterministic metadata tags."""
    out: dict[str, str] = {}
    texts = [track_lm_text(r) for r in records]
    probs = None
    if len(records):
        import numpy as np

        try:
            probs = tagger.classify(texts, list(lm_tags), batch_size=batch_size)
        except TypeError:
            probs = tagger.classify(texts, list(lm_tags))
    for i, rec in enumerate(records):
        tags = derived_tags(rec)
        if probs is not None:
            for j, tag in enumerate(lm_tags):
                if probs[i, j] > threshold:
                    tags.append(tag)
        order = {t: k for k, t in enumerate(ALL_TAGS)}
        out[rec["m4a_id"]] = tags_to_string(sorted(set(tags), key=lambda t: order.get(t, 10**6)))
    return out


NEGATION_PATTERNS = [r"\bwithout\b", r"\bno\b", r"\bnot\b", r"\bexcept\b", r"\bavoid\b"]


def extract_query_tags(query: str, tagger: BaseTagger, *, threshold: float = 0.6) -> tuple[list[str], list[str]]:
    """Online: query -> (include tags, exclude tags)."""
    include: list[str] = []
    exclude: list[str] = []
    blob = _norm(query)

    # metadata tags via lexicon on the query text
    for tag, kws in GENRE_KEYWORDS.items():
        if any(k in blob for k in kws):
            include.append(tag)
    for tag, kws in INSTRUMENT_KEYWORDS.items():
        if any(k in blob for k in kws):
            include.append(tag)
    for tag in ERA_TAGS:
        if tag[:-1] in blob or tag in blob:
            include.append(tag)
    if re.search(r"\binstrumental\b", blob):
        include.append("instrumental")
    if re.search(r"\bmale vocal", blob) or "male singer" in blob:
        include.append("male vocals")
    if re.search(r"\bfemale vocal", blob) or "female singer" in blob:
        include.append("female vocals")
    if re.search(r"\b(fast|uptempo|upbeat)\b", blob):
        include.append("fast tempo")
    if re.search(r"\b(slow|downtempo)\b", blob):
        include.append("slow tempo")

    # LM tags
    if isinstance(tagger, DiffusionTagClassifier):
        probs = tagger.classify([query], list(LM_TAGS))[0]
        for j, tag in enumerate(LM_TAGS):
            if probs[j] > threshold:
                include.append(tag)
    else:
        for tag, kws in {**MOOD_TAGS, **THEME_TAGS}.items():
            if any(k in blob for k in kws):
                include.append(tag)

    # naive negation: "without X" / "no X" / "but not X"
    for match in re.finditer(r"(without|no|but not|not|avoid|except)\s+([a-z0-9 &'\-]{2,40})", blob):
        neg_text = match.group(2)
        for tag in list(include):
            words = [w for w in re.split(r"[^a-z0-9]+", tag) if w]
            if words and all(w in neg_text for w in words):
                include.remove(tag)
                exclude.append(tag)

    order = {t: i for i, t in enumerate(ALL_TAGS)}
    include = sorted(set(include), key=lambda t: order.get(t, 10**6))
    exclude = sorted(set(exclude), key=lambda t: order.get(t, 10**6))
    return include, exclude


# --------------------------------------------------------------------------- #
# LanceDB: add extracted_tags column + search
# --------------------------------------------------------------------------- #


def add_tags_column(
    src_db: str | Path,
    dst_db: str | Path,
    tags_by_id: dict[str, str],
    *,
    extras_by_id: dict[str, dict[str, Any]] | None = None,
    table_name: str = "tracks",
    popularity_by_id: dict[str, float] | None = None,
) -> Any:
    """Rebuild the LanceDB table with an ``extracted_tags`` string column."""
    import lancedb
    import numpy as np
    import pyarrow as pa

    src = lancedb.connect(str(src_db)).open_table(table_name)
    data = src.to_arrow()
    ids = data.column("id").to_pylist()
    vec_col = data.column("vector_combined").combine_chunks()
    vectors = np.asarray(vec_col.flatten().to_numpy(zero_copy_only=False), dtype="float32")
    dim = vectors.size // max(1, len(ids))
    vectors = vectors.reshape(len(ids), dim)

    cols: dict[str, Any] = {
        "id": ids,
        "spotify_id": data.column("spotify_id").to_pylist() if "spotify_id" in data.column_names else [""] * len(ids),
        "title": data.column("title").to_pylist() if "title" in data.column_names else [""] * len(ids),
        "artist": data.column("artist").to_pylist() if "artist" in data.column_names else [""] * len(ids),
        "album": data.column("album").to_pylist() if "album" in data.column_names else [""] * len(ids),
        "year": data.column("year").to_pylist() if "year" in data.column_names else [""] * len(ids),
        "lang": data.column("lang").to_pylist() if "lang" in data.column_names else [""] * len(ids),
        "genres": data.column("genres").to_pylist() if "genres" in data.column_names else [""] * len(ids),
        "tags": data.column("tags").to_pylist() if "tags" in data.column_names else [""] * len(ids),
        "extracted_tags": [tags_by_id.get(i, "|") for i in ids],
        "popularity": [float((popularity_by_id or {}).get(i, 0.0)) for i in ids],
        "document": data.column("document").to_pylist() if "document" in data.column_names else [""] * len(ids),
        "vector_combined": pa.FixedSizeListArray.from_arrays(pa.array(vectors.reshape(-1)), dim),
    }
    tbl = pa.table(cols)

    dst = lancedb.connect(str(dst_db))
    names = set(getattr(dst.list_tables(), "tables", []))
    if table_name in names:
        dst.drop_table(table_name)
    out = dst.create_table(table_name, data=tbl)

    import warnings

    for col in ("vector_combined", "extracted_tags"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                if col == "vector_combined":
                    out.create_index(metric="cosine", index_type="IVF_HNSW_SQ",
                                     vector_column_name=col, m=20, ef_construction=300, replace=True)
                else:
                    out.create_index(column=col, index_type="BTREE", replace=True)
        except Exception as exc:  # pragma: no cover
            log(f"index on {col} skipped: {exc}")
    log(f"rebuilt table with tags: {out.count_rows()} rows, {len(tags_by_id)} tagged")
    return out


def tag_filter(include: Sequence[str], exclude: Sequence[str] = ()) -> str | None:
    clauses = [f"extracted_tags LIKE '%|{t}|%'" for t in include]
    clauses += [f"extracted_tags NOT LIKE '%|{t}|%'" for t in exclude]
    return " AND ".join(clauses) if clauses else None


def search_vector(table: Any, query_vec: "Any", k: int, *, nprobes: int = 200) -> list[dict[str, Any]]:
    import numpy as np

    q = np.asarray(query_vec, dtype="float32").reshape(-1)
    rows = (table.search(q, vector_column_name="vector_combined")
            .metric("cosine").limit(k).to_list())
    for r in rows:
        r["score"] = 1.0 - float(r.pop("_distance", 0.0))
    return rows


def search_hybrid(
    table: Any, query_vec: "Any", include: Sequence[str], exclude: Sequence[str], k: int,
    *, fallback: bool = True, nprobes: int = 200,
) -> list[dict[str, Any]]:
    where = tag_filter(include, exclude)
    if not where:
        return search_vector(table, query_vec, k, nprobes=nprobes)
    import numpy as np

    q = np.asarray(query_vec, dtype="float32").reshape(-1)
    rows = (table.search(q, vector_column_name="vector_combined")
            .where(where).metric("cosine").limit(k).to_list())
    if fallback and not rows:
        rows = search_vector(table, query_vec, k, nprobes=nprobes)
    for r in rows:
        if "_distance" in r:
            r["score"] = 1.0 - float(r.pop("_distance"))
    return rows[:k]


def load_metadata(table: Any) -> "Any":
    """Load the table metadata (without vectors) into a pandas DataFrame once."""
    data = table.to_arrow()
    keep = [c for c in data.column_names if c != "vector_combined"]
    return data.select(keep).to_pandas()


def search_tags_only(
    metadata_df: "Any", include: Sequence[str], exclude: Sequence[str], k: int,
) -> list[dict[str, Any]]:
    """Rank by number of matched query tags, then popularity."""
    if not include:
        return []
    mask = metadata_df["extracted_tags"].apply(lambda s: all(f"|{t}|" in s for t in include))
    for t in exclude:
        mask &= ~metadata_df["extracted_tags"].apply(lambda s, _t=t: f"|{_t}|" in s)
    df = metadata_df[mask]
    if df.empty:
        return []
    score = df["extracted_tags"].apply(lambda s: sum(1 for t in include if f"|{t}|" in s))
    df = df.assign(_score=score).sort_values(["_score", "popularity"], ascending=False)
    return df.head(k).to_dict("records")


# --------------------------------------------------------------------------- #
# Evaluation on the English dataset (queries + qrels)
# --------------------------------------------------------------------------- #


def load_queries_qrels(
    queries_path: str | Path,
    qrels_path: str | Path,
    *,
    max_queries: int | None = None,
    query_types: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    import pyarrow.parquet as pq

    qrels = {r["query_id"]: r for r in pq.read_table(qrels_path).to_pylist()}
    rows = pq.read_table(queries_path).to_pylist()
    if query_types:
        rows = [r for r in rows if r["query_type"] in query_types]
    if max_queries:
        rows = rows[:max_queries]
    return [r for r in rows if r["query_id"] in qrels], qrels


def ndcg_at(rank: int | None, k: int = 20) -> float:
    if rank is None or rank >= k:
        return 0.0
    return 1.0 / math.log2(rank + 2)


def _filter_similar_to(rows: list[dict[str, Any]], qrel: dict[str, Any]) -> list[dict[str, Any]]:
    excl_id = qrel.get("exclude_m4a_id")
    excl_artist = (qrel.get("exclude_artist") or "").strip().lower()
    out = []
    for r in rows:
        if excl_id and r.get("id") == excl_id:
            continue
        if excl_artist and str(r.get("artist", "")).strip().lower() == excl_artist:
            continue
        out.append(r)
    return out


def evaluate_approach(
    table: Any,
    queries: list[dict[str, Any]],
    qrels: dict[str, dict[str, Any]],
    *,
    approach: str,
    embedder: Any = None,
    tagger: BaseTagger | None = None,
    metadata_df: Any = None,
    k: int = 20,
    query_batch: int = 32,
    threshold: float = 0.6,
) -> dict[str, Any]:
    """approach in {vector, tags, hybrid}; returns nDCG@20/Recall/MRR + latency."""
    results = []
    latencies = []
    for start in range(0, len(queries), query_batch):
        batch = queries[start : start + query_batch]
        texts = [r["query"] for r in batch]
        vecs = None
        if approach in ("vector", "hybrid"):
            vecs = embedder.encode([f"task: search result | query: {t}" for t in texts], batch_size=query_batch)
        for j, q in enumerate(batch):
            t0 = time.perf_counter()
            qrel = qrels[q["query_id"]]
            gt = qrel["target_m4a_id"]
            if approach == "vector":
                rows = search_vector(table, vecs[j], k * 5)
            elif approach == "hybrid":
                include, exclude = extract_query_tags(texts[j], tagger, threshold=threshold)
                rows = search_hybrid(table, vecs[j], include, exclude, k * 5)
            elif approach == "tags":
                include, exclude = extract_query_tags(texts[j], tagger, threshold=threshold)
                rows = search_tags_only(metadata_df, include, exclude, k * 5)
            else:
                raise ValueError(approach)
            latencies.append(time.perf_counter() - t0)
            rows = _filter_similar_to(rows, qrel)
            ranked = [r["id"] for r in rows]
            rank = ranked.index(gt) if gt in ranked else None
            results.append((q["query_type"], rank))

    n = max(1, len(results))
    metrics: dict[str, Any] = {
        "approach": approach,
        "n": len(results),
        "ndcg@20": sum(ndcg_at(r, 20) for _, r in results) / n,
        "recall@20": sum(1 for _, r in results if r is not None and r < 20) / n,
        "recall@100": sum(1 for _, r in results if r is not None) / n,
        "mrr": sum(1.0 / (r + 1) for _, r in results if r is not None) / n,
        "latency_ms_mean": 1000.0 * sum(latencies) / max(1, len(latencies)),
        "rps": 1.0 / (sum(latencies) / max(1, len(latencies))),
        "by_query_type": {},
    }
    by: dict[str, list[int | None]] = {}
    for qt, r in results:
        by.setdefault(qt, []).append(r)
    for qt, ranks in sorted(by.items()):
        m = max(1, len(ranks))
        metrics["by_query_type"][qt] = {
            "n": len(ranks),
            "ndcg@20": sum(ndcg_at(r, 20) for r in ranks) / m,
            "recall@20": sum(1 for r in ranks if r is not None and r < 20) / m,
            "recall@100": sum(1 for r in ranks if r is not None) / m,
        }
    return metrics


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def cmd_extract_tags(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    records = pq.read_table(args.tracks_meta).to_pylist()
    if args.limit:
        records = records[: args.limit]
    tagger = get_tagger(args.tagger, device=args.device, dtype=args.dtype)
    log(f"classifying {len(records)} tracks with {args.tagger} tagger")
    tags = extract_track_tags(records, tagger, threshold=args.threshold)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps({"m4a_id": k, "extracted_tags": v}) for k, v in tags.items()), encoding="utf-8")
    log(f"wrote {out}")


def cmd_add_tags(args: argparse.Namespace) -> None:
    tags_by_id = {}
    for line in Path(args.tags).read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            tags_by_id[obj["m4a_id"]] = obj["extracted_tags"]
    add_tags_column(args.src_db, args.dst_db, tags_by_id)


def measure_rps(
    table: Any,
    queries: list[dict[str, Any]],
    *,
    approach: str,
    embedder: Any = None,
    tagger: BaseTagger | None = None,
    metadata_df: Any = None,
    sample: int = 200,
    k: int = 20,
    threshold: float = 0.6,
) -> dict[str, float]:
    """End-to-end single-request latency (batch=1): embed + tags + search."""
    subset = queries[: max(1, min(sample, len(queries)))]
    lat: list[float] = []
    for q in subset:
        text = q["query"]
        t0 = time.perf_counter()
        vec = None
        if approach in ("vector", "hybrid"):
            vec = embedder.encode([f"task: search result | query: {text}"], batch_size=1)[0]
        if approach in ("tags", "hybrid"):
            include, exclude = extract_query_tags(text, tagger, threshold=threshold)
        else:
            include, exclude = [], []
        if approach == "vector":
            search_vector(table, vec, k)
        elif approach == "hybrid":
            search_hybrid(table, vec, include, exclude, k)
        elif approach == "tags":
            search_tags_only(metadata_df, include, exclude, k)
        lat.append(time.perf_counter() - t0)
    mean = sum(lat) / max(1, len(lat))
    return {"n": len(lat), "latency_ms_mean": 1000.0 * mean, "rps": 1.0 / mean if mean else 0.0}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="LanceDB pre-filtering with diffusion tags")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("extract-tags")
    e.add_argument("--tracks-meta", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--tagger", choices=["diffusion", "lexicon"], default="diffusion")
    e.add_argument("--device", default="cuda")
    e.add_argument("--dtype", default="float16")
    e.add_argument("--threshold", type=float, default=0.8)
    e.add_argument("--limit", type=int, default=None)

    a = sub.add_parser("add-tags")
    a.add_argument("--src-db", required=True)
    a.add_argument("--dst-db", required=True)
    a.add_argument("--tags", required=True)

    sub.add_parser("taxonomy")
    args = p.parse_args(argv)
    if args.command == "extract-tags":
        cmd_extract_tags(args)
    elif args.command == "add-tags":
        cmd_add_tags(args)
    else:
        for group in ("mood", "theme", "vocal", "genre", "language", "era", "instrument", "energy"):
            tags = [t for t in ALL_TAGS if TAG_GROUP.get(t) == group]
            print(f"{group:10s} ({len(tags):2d}): {', '.join(tags)}")
        print(f"TOTAL: {len(ALL_TAGS)} tags")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
