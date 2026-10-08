"""Синтетика в формате Music4All-CRS, чтобы тесты работали без файлов датасета.

make_synthetic() -> (tracks_meta, {"train": tables, "test_public": tables}), где tables — словарь
{"users", "queries", "qrels"} с теми же колонками, что train.parquet / *_queries.parquet / *_qrels.parquet.
Дальше те же функции, что для настоящих файлов (recsys.data.crs). Запросы на английском, по шаблонам из
атрибутов цели; ~90% позитивов из истории, плюс discovery-запросы novelty и similar_to.
Метрики на синтетике ничего не говорят о реальном качестве.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# жанр -> теги жанра
GENRES = {
    "indie rock": ["indie", "alternative", "guitar", "melancholic", "british"],
    "hip hop": ["rap", "underground hip hop", "urban", "party"],
    "electronic": ["dance", "house", "energetic", "party", "synth"],
    "ambient": ["chill", "calm", "atmospheric", "instrumental", "relaxing"],
    "jazz": ["smooth jazz", "saxophone", "instrumental", "relaxing", "night"],
    "classical": ["piano", "instrumental", "orchestral", "calm", "romantic"],
    "metal": ["heavy metal", "aggressive", "guitar", "energetic", "dark"],
    "pop": ["catchy", "dance", "happy", "female vocalists", "party"],
    "folk": ["acoustic", "singer-songwriter", "calm", "melancholic", "guitar"],
    "soul": ["rnb", "romantic", "smooth", "female vocalists"],
    "punk": ["punk rock", "energetic", "fast", "aggressive", "garage"],
}
EXTRA_TAGS = ["sad", "happy", "summer", "female vocalists", "male vocalists", "night", "romantic"]
# теги, которые можно исключить в negative_constraint («... but no rap»)
NEGATABLE = ["sad", "aggressive", "female vocalists", "male vocalists", "dark", "party", "dance", "guitar",
             "piano", "rap", "synth", "fast"]
COUNTRIES = {"US": "american", "GB": "british", "DE": "german", "SE": "swedish", "FR": "french"}
DECADES = [1970, 1980, 1990, 2000, 2010]
SITUATIONS = [("for running", "high"), ("for the gym", "high"), ("for a party", "high"),
              ("for studying", "low"), ("to fall asleep", "low"), ("for quiet work", "low")]
QUERY_TYPES = ["exact", "lyrics_recall", "genre", "mood", "situation", "era_region", "audio_attributes",
               "complex", "negative_constraint"]
_ADJ = ["Velvet", "Electric", "Silent", "Golden", "Neon", "Paper", "Crystal", "Midnight", "Wild", "Lunar",
        "Broken", "Hollow", "Scarlet", "Frozen", "Cosmic", "Northern", "Rusty", "Gentle", "Static", "Violet"]
_NOUN = ["Owls", "Rivers", "Machines", "Hearts", "Wolves", "Lights", "Engines", "Gardens", "Ghosts", "Tides",
         "Foxes", "Mirrors", "Waves", "Pilots", "Saints", "Echoes", "Kites", "Comets", "Bridges", "Crowns"]
_WORDS = ["rain", "fire", "morning", "shadows", "dreams", "city", "ocean", "glass", "summer", "home", "stars",
          "road", "smoke", "gold", "silence", "storm", "paper", "echo", "blue", "heart", "night", "river",
          "window", "train", "mountain", "letter", "mirror", "garden", "winter", "light"]
# словарь текстов песен: ~2000 псевдослов, чтобы строчка из нескольких слов была почти уникальной
_LYRIC_WORDS = [a + b + c for a in ["ba", "lo", "mi", "ne", "ru", "sa", "te", "vo", "ki", "da", "fe", "go"]
                for b in ["ra", "li", "mo", "nu", "se", "ta", "vi", "ko", "de", "pa", "zu", "lo", "re", "ni"]
                for c in ["", "n", "s", "l", "r", "t", "m", "k", "x", "y", "d", "p"]]
TRAIN_START, TEST_START, TEST_END = 1579564800, 1582156800, 1584662400  # 2020-01-21, 2020-02-20, 2020-03-20


def make_synthetic(n_tracks: int = 5000, n_users: int = 300, n_artists: int = 300,
                   seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, Dict[str, pd.DataFrame]]]:
    rng = np.random.default_rng(seed)
    meta = _make_tracks(rng, n_tracks, n_artists)
    by_genre = {g: np.flatnonzero(meta["_genre"].to_numpy() == g) for g in GENRES}
    pop = meta["onion_listens"].to_numpy(dtype=float) + 1.0
    emb = np.stack(meta["muq_embedding"].to_numpy())

    rows: Dict[str, List[dict]] = {"train": [], "test_public": []}
    for u in range(n_users):
        fav = list(rng.choice(list(GENRES), size=int(rng.integers(1, 3)), replace=False))
        pool = np.concatenate([by_genre[g] for g in fav])
        n_unique = int(rng.integers(15, 120))
        own = _sample(rng, pool, pop, int(n_unique * 0.85)) + _sample(rng, np.arange(n_tracks), pop, n_unique // 7)
        history, ts = _listen(rng, list(dict.fromkeys(own)), start=1262304000, end=TRAIN_START,
                              n=int(rng.integers(50, 600)))
        demo = _demographics(rng)
        for split, (start, end) in (("train", (TRAIN_START, TEST_START)), ("test_public", (TEST_START, TEST_END))):
            own = list(dict.fromkeys(history))  # «свои» = реально прослушанные: от них считается is_new
            positives = _positives(rng, meta, own, pool, pop, start, end)
            discovery = _discovery(rng, meta, emb, own, positives)
            rows[split].append(_user_row(u, split, history, ts, demo, positives, discovery, meta))
            # история теста = история train + прослушивания train-окна
            history = history + [p["_idx"] for p in positives]
            ts = ts + [p["ts"] for p in positives]
    return meta.drop(columns=["_genre"]), {split: _tables(r) for split, r in rows.items()}


# ---------------------------------------------------------------- каталог

def _make_tracks(rng, n_tracks: int, n_artists: int) -> pd.DataFrame:
    genres = list(GENRES)
    names = [f"{a} {n}" for a in _ADJ for n in _NOUN]
    rng.shuffle(names)
    artists = [("The " + n) if rng.random() < 0.3 else n for n in names[:n_artists]]
    a_genre = rng.choice(genres, size=len(artists))
    a_country = rng.choice(list(COUNTRIES), size=len(artists))
    a_decade = rng.choice(DECADES, size=len(artists))
    a_pop = rng.lognormal(0, 1.0, size=len(artists))
    centers = {g: _unit(rng.normal(size=128)) for g in genres}

    rows = []
    for i in range(n_tracks):
        a = int(rng.integers(len(artists)))
        g = str(a_genre[a])
        own = list(rng.choice(GENRES[g], size=min(len(GENRES[g]), int(rng.integers(2, 5))), replace=False))
        extra = [t for t in rng.choice(EXTRA_TAGS, size=int(rng.integers(0, 3)), replace=False) if t not in own]
        tags = {g: 100}
        tags.update({t: int(rng.integers(30, 95)) for t in own})
        tags.update({t: int(rng.integers(5, 40)) for t in extra})
        instrumental = "instrumental" in tags
        energetic = any(t in tags for t in ("energetic", "aggressive", "fast", "party", "dance"))
        calm = any(t in tags for t in ("calm", "chill", "relaxing"))
        title = " ".join(w.capitalize() for w in rng.choice(_WORDS, size=int(rng.integers(1, 4)), replace=False))
        lyrics = "" if instrumental else " ".join(rng.choice(_LYRIC_WORDS, size=60))
        rows.append({
            "m4a_id": f"T{i:015d}",
            "spotify_id": f"S{i:021d}",
            "m4a_artist": artists[a],
            "m4a_song": title,
            "m4a_album": f"{title.split()[0]} Album",
            "release_year": int(a_decade[a]) + int(rng.integers(0, 10)),
            "lang": "INTRUMENTAL" if instrumental else "en",
            "is_instrumental": instrumental,
            "artist_country": str(a_country[a]),
            "m4a_genres_full": g,
            "lastfm_tag_weights": json.dumps(tags),
            "onion_listens": int(a_pop[a] * rng.lognormal(3, 1)),
            "energy": float(np.clip(rng.normal(0.8 if energetic else 0.3 if calm else 0.55, 0.1), 0, 1)),
            "valence": float(np.clip(rng.normal(0.7 if "happy" in tags else 0.3 if "sad" in tags else 0.5, 0.1), 0, 1)),
            "tempo": float(rng.normal(150 if energetic else 85 if calm else 115, 10)),
            "danceability": float(np.clip(rng.normal(0.7 if "dance" in tags else 0.45, 0.1), 0, 1)),
            "pseudo_caption": f"A {', '.join(list(tags)[:3])} song.",
            "lyrics": lyrics,
            "muq_embedding": list(_unit(centers[g] + 0.6 * _unit(rng.normal(size=128)))),
            "_genre": g,
        })
    return pd.DataFrame(rows)


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def _sample(rng, idx: np.ndarray, pop: np.ndarray, k: int) -> List[int]:
    k = min(k, len(idx))
    return [int(x) for x in rng.choice(idx, size=k, replace=False, p=pop[idx] / pop[idx].sum())] if k else []


# ---------------------------------------------------------------- пользователи и запросы

def _listen(rng, own: List[int], start: int, end: int, n: int):
    """n прослушиваний своих треков (частые чаще), отсортированных по времени."""
    w = rng.pareto(1.2, size=len(own)) + 0.1
    picks = rng.choice(len(own), size=n, p=w / w.sum())
    ts = np.sort(rng.integers(start, end, size=n))
    return [int(own[i]) for i in picks], [int(t) for t in ts]


def _demographics(rng):
    if rng.random() < 0.4:
        return None
    return {"age": int(rng.integers(16, 55)), "gender": str(rng.choice(["m", "f"])),
            "country": str(rng.choice(["US", "DE", "RU", "BR", "PL"]))}


def _positives(rng, meta, own: List[int], pool: np.ndarray, pop: np.ndarray, start: int, end: int) -> List[dict]:
    out, seen = [], set()
    for _ in range(int(rng.integers(1, 8))):
        idx = int(own[int(rng.integers(len(own)))]) if rng.random() < 0.9 and own else _sample(rng, pool, pop, 1)[0]
        if idx in seen:
            continue
        seen.add(idx)
        query_type, query = _query(rng, meta.iloc[idx])
        out.append({"_idx": idx, "is_new": idx not in own, "ts": int(rng.integers(start, end)),
                    "n_listens": int(rng.integers(1, 6)), "query": query, "query_type": query_type})
    out.sort(key=lambda p: p["ts"])
    return out


def _discovery(rng, meta, emb: np.ndarray, own: List[int], positives: List[dict]) -> List[dict]:
    """novelty — новый трек нового для пользователя артиста; similar_to — новый трек, похожий (MuQ) на трек X
    из истории, другого артиста."""
    own_artists = {meta.at[i, "m4a_artist"] for i in own}
    out = []
    for p in positives:
        if not p["is_new"] or rng.random() < 0.3:
            continue
        t = meta.iloc[p["_idx"]]
        tags = list(json.loads(t["lastfm_tag_weights"]))
        if t["m4a_artist"] not in own_artists and rng.random() < 0.5:
            out.append({**p, "kind": "novelty", "reference": None,
                        "query": f"new artists please, something {tags[-1]} i haven't heard, maybe {tags[0]}"})
            continue
        others = [i for i in own if meta.at[i, "m4a_artist"] != t["m4a_artist"]]
        if not others:
            continue
        x = max(others, key=lambda i: float(emb[i] @ emb[p["_idx"]]))
        out.append({**p, "kind": "similar_to", "reference": x,
                    "query": f"tracks like {meta.at[x, 'm4a_song'].lower()} by {meta.at[x, 'm4a_artist'].lower()} "
                             f"but from other artists, more {tags[-1]}"})
    return out


def _query(rng, t: pd.Series) -> Tuple[str, str]:
    tags = list(json.loads(t["lastfm_tag_weights"]))
    genre, own = tags[0], tags[1:]
    decade = f"{(t['release_year'] // 10) % 10}0s"
    country = COUNTRIES[t["artist_country"]]
    query_type = str(rng.choice(QUERY_TYPES))
    if query_type == "exact":
        return query_type, f"play {t['m4a_song']} by {t['m4a_artist']}".lower()
    if query_type == "lyrics_recall" and t["lyrics"]:
        words = t["lyrics"].split()
        i = int(rng.integers(0, len(words) - 4))
        return query_type, "the song that goes " + " ".join(words[i:i + 4])
    if query_type in ("genre", "lyrics_recall") or not own:
        return "genre", f"{own[0] if own else ''} {genre}".strip()
    if query_type == "mood":
        return query_type, f"{rng.choice(own)} {genre}"
    if query_type == "situation":
        level = "high" if t["energy"] > 0.6 else "low"
        return query_type, f"music {rng.choice([s for s, lv in SITUATIONS if lv == level])}"
    if query_type == "era_region":
        return query_type, f"{country} {genre} from the {decade}"
    if query_type == "audio_attributes":
        speed = "fast" if t["tempo"] > 120 else "slow"
        energy = "energetic" if t["energy"] > 0.6 else "calm"
        return query_type, f"{speed} {energy} track"
    if query_type == "complex":
        return query_type, f"{rng.choice(own)} {country} {genre} from the {decade}"
    absent = [x for x in NEGATABLE if x not in tags]  # negative_constraint: исключаем тег, которого у трека нет
    return "negative_constraint", f"{genre}, but no {rng.choice(absent)}"


def _query_id(split: str, user_id: int, kind: str, m4a_id: str) -> str:
    return hashlib.sha1(f"{split}|{user_id}|{kind}|{m4a_id}".encode()).hexdigest()[:16]


def _user_row(u: int, split: str, history: List[int], ts: List[int], demo, positives: List[dict],
              discovery: List[dict], meta: pd.DataFrame) -> dict:
    ids = meta["m4a_id"].to_numpy()
    user_id = 100000 + u
    artists = pd.Series([meta.at[i, "m4a_artist"] for i in history]).value_counts().index[:10].tolist()
    genres = pd.Series([meta.at[i, "m4a_genres_full"] for i in history]).value_counts().index[:5].tolist()
    profile = (f"Listening since 2010; {len(history)} plays before this period, {len(set(history))} distinct tracks, "
               f"{len(set(meta.at[i, 'm4a_artist'] for i in history))} artists. "
               f"Favourite artists: {', '.join(artists)}. Favourite genres: {', '.join(genres)}.")

    def track(i: int) -> dict:
        return {"m4a_id": str(ids[i]), "spotify_id": meta.at[i, "spotify_id"], "artist": meta.at[i, "m4a_artist"],
                "title": meta.at[i, "m4a_song"]}

    return {
        "user_id": user_id, "split": split, "history": [str(ids[i]) for i in history], "history_ts": ts,
        "history_len": len(history), "user_profile": profile, "user_demographics": demo,
        "n_positives": len(positives),
        "positives": [{"query_id": _query_id(split, user_id, "positive", ids[p["_idx"]]), **track(p["_idx"]),
                       "ts": p["ts"], "n_listens": p["n_listens"], "is_new": p["is_new"], "query": p["query"],
                       "query_family": p["query_type"]} for p in positives],
        "n_discovery": len(discovery),
        "discovery": [{"query_id": _query_id(split, user_id, d["kind"], ids[d["_idx"]]), "kind": d["kind"],
                       **track(d["_idx"]), "ts": d["ts"], "query": d["query"],
                       "reference_m4a_id": None if d["reference"] is None else str(ids[d["reference"]]),
                       "reference_artist": None if d["reference"] is None else meta.at[d["reference"], "m4a_artist"]}
                      for d in discovery],
    }


def _tables(rows: List[dict]) -> Dict[str, pd.DataFrame]:
    """Строки пользователей -> users / queries / qrels, как в файлах датасета."""
    queries, qrels = [], []
    for r in rows:
        for p in r["positives"]:
            queries.append({"query_id": p["query_id"], "user_id": r["user_id"], "split": r["split"],
                            "query": p["query"], "source": "positive", "query_type": p["query_family"]})
            qrels.append({"query_id": p["query_id"], "target_m4a_id": p["m4a_id"], "exclude_m4a_id": None,
                          "exclude_artist": None, "is_new": p["is_new"], "ts": p["ts"]})
        for d in r["discovery"]:
            queries.append({"query_id": d["query_id"], "user_id": r["user_id"], "split": r["split"],
                            "query": d["query"], "source": "discovery", "query_type": d["kind"]})
            qrels.append({"query_id": d["query_id"], "target_m4a_id": d["m4a_id"],
                          "exclude_m4a_id": d["reference_m4a_id"], "exclude_artist": d["reference_artist"],
                          "is_new": True, "ts": d["ts"]})
    return {"users": pd.DataFrame(rows), "queries": pd.DataFrame(queries), "qrels": pd.DataFrame(qrels)}
