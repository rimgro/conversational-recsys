"""Синтетика в формате Music4All-CRS (tracks_meta + сплиты), чтобы всё запускалось без данных.

make_synthetic() -> (tracks_meta, {"train": df, "test_public": df}); дальше те же функции, что для
настоящих файлов (recsys.data.crs). Запросы на русском, по шаблонам из атрибутов цели;
~90% целей из истории (как в датасете). Метрики на синтетике ничего не говорят о реальном качестве.
"""
from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# жанр -> (по-русски, теги жанра)
GENRES: Dict[str, Tuple[str, List[str]]] = {
    "indie rock": ("инди-рок", ["indie", "alternative", "guitar", "melancholic", "british"]),
    "hip hop": ("хип-хоп", ["rap", "underground hip hop", "urban", "party"]),
    "electronic": ("электронная музыка", ["dance", "house", "energetic", "party", "synth"]),
    "ambient": ("эмбиент", ["chill", "calm", "atmospheric", "instrumental", "relaxing"]),
    "jazz": ("джаз", ["smooth jazz", "saxophone", "instrumental", "relaxing", "night"]),
    "classical": ("классическая музыка", ["piano", "instrumental", "orchestral", "calm", "romantic"]),
    "metal": ("метал", ["heavy metal", "aggressive", "guitar", "energetic", "dark"]),
    "pop": ("поп", ["catchy", "dance", "happy", "female vocalists", "party"]),
    "folk": ("фолк", ["acoustic", "singer-songwriter", "calm", "melancholic", "guitar"]),
    "soul": ("соул", ["rnb", "romantic", "smooth", "female vocalists"]),
    "punk": ("панк", ["punk rock", "energetic", "fast", "aggressive", "garage"]),
}
# как тег звучит в русском запросе (прилагательные в форме, которую ловит словарь разбора)
TAG_RU: Dict[str, str] = {
    "indie": "инди", "alternative": "альтернативный", "guitar": "гитарный", "melancholic": "меланхоличный",
    "british": "британский", "rap": "рэп", "underground hip hop": "андеграундный хип-хоп", "urban": "уличный",
    "party": "для вечеринки", "dance": "танцевальный", "house": "хаус", "energetic": "энергичный",
    "synth": "синтезаторный", "chill": "расслабленный", "calm": "спокойный", "atmospheric": "атмосферный",
    "instrumental": "без слов", "relaxing": "расслабляющий", "smooth jazz": "смус-джаз", "saxophone": "с саксофоном",
    "night": "ночной", "piano": "фортепианный", "orchestral": "оркестровый", "romantic": "романтичный",
    "heavy metal": "хеви-метал", "aggressive": "агрессивный", "dark": "мрачный", "catchy": "цепляющий",
    "happy": "весёлый", "female vocalists": "с женским вокалом", "acoustic": "акустический",
    "singer-songwriter": "авторский", "rnb": "ритм-н-блюз", "smooth": "плавный", "punk rock": "панк-рок",
    "fast": "быстрый", "garage": "гаражный", "sad": "грустный", "summer": "летний",
    "male vocalists": "с мужским вокалом",
}
# как исключение «без ...»
TAG_RU_GEN: Dict[str, str] = {
    "sad": "грустных нот", "aggressive": "агрессии", "female vocalists": "женского вокала",
    "male vocalists": "мужского вокала", "dark": "мрачности", "party": "вечеринок", "dance": "танцевальности",
    "guitar": "гитар", "piano": "фортепиано", "rap": "рэпа", "synth": "синтезаторов", "fast": "быстрого темпа",
}
EXTRA_TAGS = ["happy", "sad", "dark", "romantic", "summer", "night", "male vocalists"]
COUNTRIES = {"US": "американский", "GB": "британский", "DE": "немецкий", "SE": "шведский", "FR": "французский"}
DECADE_RU = {1970: "70-х", 1980: "80-х", 1990: "90-х", 2000: "2000-х", 2010: "2010-х"}
SITUATIONS = [("для пробежки", "high"), ("для тренировки", "high"), ("для вечеринки", "high"),
              ("для учёбы", "low"), ("чтобы уснуть", "low"), ("для работы в тишине", "low")]
_ADJ = ["Velvet", "Electric", "Silent", "Golden", "Neon", "Paper", "Crystal", "Midnight", "Wild", "Lunar",
        "Broken", "Hollow", "Scarlet", "Frozen", "Cosmic", "Northern", "Rusty", "Gentle", "Static", "Violet"]
_NOUN = ["Owls", "Rivers", "Machines", "Hearts", "Wolves", "Lights", "Engines", "Gardens", "Ghosts", "Tides",
         "Foxes", "Mirrors", "Waves", "Pilots", "Saints", "Echoes", "Kites", "Comets", "Bridges", "Crowns"]
_WORDS = ["rain", "fire", "morning", "shadows", "dreams", "city", "ocean", "glass", "summer", "home", "stars",
          "road", "smoke", "gold", "silence", "storm", "paper", "echo", "blue", "heart", "night", "river",
          "window", "train", "mountain", "letter", "mirror", "garden", "winter", "light"]
_LAT2CYR = [("sh", "ш"), ("ch", "ч"), ("th", "т"), ("oo", "у"), ("ee", "и"), ("a", "а"), ("b", "б"), ("c", "к"),
            ("d", "д"), ("e", "е"), ("f", "ф"), ("g", "г"), ("h", "х"), ("i", "и"), ("j", "дж"), ("k", "к"),
            ("l", "л"), ("m", "м"), ("n", "н"), ("o", "о"), ("p", "п"), ("q", "к"), ("r", "р"), ("s", "с"),
            ("t", "т"), ("u", "у"), ("v", "в"), ("w", "в"), ("x", "кс"), ("y", "й"), ("z", "з")]
TRAIN_START, TEST_START, TEST_END = 1579564800, 1582156800, 1584662400  # 2020-01-21, 2020-02-20, 2020-03-20


def to_cyrillic(text: str) -> str:
    """Грубая транслитерация латиницы в кириллицу (имитирует запросы «кате буш»)."""
    out, s, i = [], text.lower(), 0
    while i < len(s):
        for lat, cyr in _LAT2CYR:
            if s.startswith(lat, i):
                out.append(cyr)
                i += len(lat)
                break
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def make_synthetic(n_tracks: int = 5000, n_users: int = 300, n_artists: int = 300,
                   seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    rng = np.random.default_rng(seed)
    meta = _make_tracks(rng, n_tracks, n_artists)
    by_genre = {g: np.flatnonzero(meta["_genre"].to_numpy() == g) for g in GENRES}
    pop = meta["onion_listens"].to_numpy(dtype=float) + 1.0

    train_rows, test_rows = [], []
    for u in range(n_users):
        fav = list(rng.choice(list(GENRES), size=int(rng.integers(1, 3)), replace=False))
        pool = np.concatenate([by_genre[g] for g in fav])
        n_unique = int(rng.integers(15, 120))
        own = _sample(rng, pool, pop, int(n_unique * 0.85)) + _sample(rng, np.arange(n_tracks), pop, n_unique // 7)
        own = list(dict.fromkeys(own))
        history, ts = _listen(rng, own, start=1262304000, end=TRAIN_START, n=int(rng.integers(50, 600)))
        own = list(dict.fromkeys(history))  # «свои» = реально прослушанные: от них считается is_new
        profile_info = _demographics(rng)
        train_pos = _positives(rng, meta, own, pool, pop, TRAIN_START, TEST_START)
        train_rows.append(_user_row(u, "train", history, ts, profile_info, train_pos, meta))
        # история теста = история train + прослушивания train-окна
        h2 = history + [p["_idx"] for p in train_pos]
        t2 = ts + [p["ts"] for p in train_pos]
        own2 = list(dict.fromkeys(h2))
        test_pos = _positives(rng, meta, own2, pool, pop, TEST_START, TEST_END)
        test_rows.append(_user_row(u, "test_public", h2, t2, profile_info, test_pos, meta))
    return meta.drop(columns=["_genre"]), {"train": pd.DataFrame(train_rows), "test_public": pd.DataFrame(test_rows)}


# ---------------------------------------------------------------- каталог

def _make_tracks(rng, n_tracks: int, n_artists: int) -> pd.DataFrame:
    genres = list(GENRES)
    names = [f"{a} {n}" for a in _ADJ for n in _NOUN]
    rng.shuffle(names)
    artists = [("The " + n) if rng.random() < 0.3 else n for n in names[:n_artists]]
    a_genre = rng.choice(genres, size=len(artists))
    a_country = rng.choice(list(COUNTRIES), size=len(artists))
    a_decade = rng.choice(list(DECADE_RU), size=len(artists))
    a_pop = rng.lognormal(0, 1.0, size=len(artists))
    centers = {g: _unit(rng.normal(size=128)) for g in genres}

    rows = []
    for i in range(n_tracks):
        a = int(rng.integers(len(artists)))
        g = str(a_genre[a])
        own = list(rng.choice(GENRES[g][1], size=min(len(GENRES[g][1]), int(rng.integers(2, 5))), replace=False))
        extra = [t for t in rng.choice(EXTRA_TAGS, size=int(rng.integers(0, 3)), replace=False) if t not in own]
        tags = {g: 100}
        tags.update({t: int(rng.integers(30, 95)) for t in own})
        tags.update({t: int(rng.integers(5, 40)) for t in extra})
        instrumental = "instrumental" in tags
        energetic = any(t in tags for t in ("energetic", "aggressive", "fast", "party", "dance"))
        calm = any(t in tags for t in ("calm", "chill", "relaxing"))
        title = " ".join(w.capitalize() for w in rng.choice(_WORDS, size=int(rng.integers(1, 4)), replace=False))
        lyrics = "" if instrumental else " ".join(rng.choice(_WORDS, size=60))
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


# ---------------------------------------------------------------- пользователи

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
    n = int(rng.integers(1, 8))
    out, seen = [], set()
    for _ in range(n):
        if rng.random() < 0.9 and own:
            idx = int(own[int(rng.integers(len(own)))])
        else:
            idx = _sample(rng, pool, pop, 1)[0]
        if idx in seen:
            continue
        seen.add(idx)
        family, query = _query(rng, meta.iloc[idx])
        out.append({"_idx": idx, "is_new": idx not in own, "ts": int(rng.integers(start, end)),
                    "n_listens": int(rng.integers(1, 6)), "query": query, "query_family": family})
    out.sort(key=lambda p: p["ts"])
    return out


def _query(rng, t: pd.Series) -> Tuple[str, str]:
    tags = list(json.loads(t["lastfm_tag_weights"]))
    genre = tags[0]
    g_ru = GENRES[genre][0]
    own_ru = [TAG_RU[x] for x in tags[1:] if x in TAG_RU]
    decade = DECADE_RU[(t["release_year"] // 10) * 10]
    family = str(rng.choice(["exact", "lyrics_recall", "genre", "mood", "situation", "era_region",
                             "audio_attributes", "complex", "negative_constraint"]))
    if family == "exact":
        name = f"{t['m4a_artist']} {t['m4a_song']}".lower()
        return family, to_cyrillic(name) if rng.random() < 0.3 else name
    if family == "lyrics_recall" and t["lyrics"]:
        words = t["lyrics"].split()
        i = int(rng.integers(0, len(words) - 4))
        return family, "песня со словами " + " ".join(words[i:i + 4])
    if family == "genre" or (family == "lyrics_recall") or not own_ru:
        return "genre", f"{own_ru[0] if own_ru else ''} {g_ru}".strip()
    if family == "mood":
        return family, f"{rng.choice(own_ru)} {g_ru}"
    if family == "situation":
        level = "high" if t["energy"] > 0.6 else "low"
        place = [s for s, lv in SITUATIONS if lv == level]
        return family, f"музыка {rng.choice(place)}"
    if family == "era_region":
        return family, f"{COUNTRIES[t['artist_country']]} {g_ru} {decade}"
    if family == "audio_attributes":
        speed = "быстрый" if t["tempo"] > 120 else "медленный"
        energy = "энергичный" if t["energy"] > 0.6 else "спокойный"
        return family, f"{speed} {energy} трек"
    if family == "complex":
        return family, f"{rng.choice(own_ru)} {COUNTRIES[t['artist_country']]} {g_ru} {decade}"
    # negative_constraint: исключаем тег, которого у трека нет
    absent = [x for x in TAG_RU_GEN if x not in tags]
    return "negative_constraint", f"{g_ru}, но без {TAG_RU_GEN[str(rng.choice(absent))]}"


def _user_row(u: int, split: str, history: List[int], ts: List[int], demo, positives: List[dict],
              meta: pd.DataFrame) -> dict:
    ids = meta["m4a_id"].to_numpy()
    hist_ids = [str(ids[i]) for i in history]
    artists = pd.Series([meta.at[i, "m4a_artist"] for i in history]).value_counts().index[:10].tolist()
    genres = pd.Series([meta.at[i, "m4a_genres_full"] for i in history]).value_counts().index[:5].tolist()
    profile = (f"Слушает музыку с 2010 года; до периода {len(history)} прослушиваний, {len(set(history))} разных "
               f"треков, {len(set(meta.at[i, 'm4a_artist'] for i in history))} артистов. "
               f"Любимые артисты: {', '.join(artists)}. Любимые жанры: {', '.join(genres)}.")
    return {
        "user_id": 100000 + u,
        "split": split,
        "history": hist_ids,
        "history_ts": ts,
        "history_len": len(hist_ids),
        "user_profile": profile,
        "user_demographics": demo,
        "n_positives": len(positives),
        "positives": [{
            "m4a_id": str(ids[p["_idx"]]), "spotify_id": meta.at[p["_idx"], "spotify_id"],
            "artist": meta.at[p["_idx"], "m4a_artist"], "title": meta.at[p["_idx"], "m4a_song"],
            "ts": p["ts"], "n_listens": p["n_listens"], "is_new": p["is_new"],
            "query": p["query"], "query_family": p["query_family"],
        } for p in positives],
    }
