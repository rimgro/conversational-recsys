"""Синтетические данные в формате нашего датасета: каталог, прослушивания, диалоги.

Нужны, чтобы весь пайплайн запускался до появления реальных данных. Качество на них
ничего не говорит о реальном качестве: запросы строятся из тех же тегов, по которым ищем.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

GENRES: Dict[str, List[str]] = {
    "indie rock": ["indie", "alternative", "guitar", "melancholic", "british"],
    "hip hop": ["rap", "beats", "underground hip hop", "urban", "party"],
    "electronic": ["dance", "house", "energetic", "party", "synth"],
    "ambient": ["chill", "calm", "atmospheric", "instrumental", "relaxing"],
    "jazz": ["smooth jazz", "saxophone", "instrumental", "relaxing", "night"],
    "classical": ["piano", "instrumental", "orchestral", "calm", "romantic"],
    "metal": ["heavy metal", "aggressive", "guitar", "energetic", "dark"],
    "pop": ["catchy", "dance", "happy", "female vocalists", "party"],
    "folk": ["acoustic", "singer-songwriter", "calm", "melancholic", "guitar"],
    "lo-fi": ["chill hop", "beats", "study", "chill", "instrumental"],
    "rnb": ["soul", "romantic", "smooth", "female vocalists", "sexy"],
    "punk": ["punk rock", "energetic", "fast", "aggressive", "garage"],
}
EXTRA_TAGS = ["happy", "sad", "dark", "romantic", "summer", "night", "80s", "90s", "00s", "male vocalists"]
_ADJ = ["Velvet", "Electric", "Silent", "Golden", "Neon", "Paper", "Crystal", "Midnight", "Wild", "Lunar",
        "Broken", "Hollow", "Scarlet", "Frozen", "Cosmic", "Northern", "Rusty", "Gentle", "Static", "Violet"]
_NOUN = ["Owls", "Rivers", "Machines", "Hearts", "Wolves", "Lights", "Engines", "Gardens", "Ghosts", "Tides",
         "Foxes", "Mirrors", "Waves", "Pilots", "Saints", "Echoes", "Kites", "Comets", "Bridges", "Crowns"]
_WORDS = ["Rain", "Fire", "Morning", "Shadows", "Dreams", "City", "Ocean", "Glass", "Summer", "Static",
          "Home", "Stars", "Road", "Smoke", "Gold", "Silence", "Storm", "Paper", "Echo", "Blue"]
COUNTRIES = ["Germany", "Brazil", "USA", "Russia", "Poland", "UK", "Japan", "Spain", "Mexico", "Sweden"]


def make_synthetic(
    n_tracks: int = 5000,
    n_users: int = 500,
    n_dialogs: int = 300,
    n_artists: int = 300,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """-> (tracks, interactions, dialogs) в формате, который читают loaders/dataset."""
    rng = np.random.default_rng(seed)
    genres = list(GENRES)

    # артисты
    names = list(dict.fromkeys(f"{a} {n}" for a in _ADJ for n in _NOUN))
    rng.shuffle(names)
    artists = [("The " + n) if rng.random() < 0.3 else n for n in names[:n_artists]]
    artist_genre = rng.choice(genres, size=len(artists))
    artist_pop = rng.lognormal(0, 1.0, size=len(artists))

    # треки
    rows = []
    for i in range(n_tracks):
        a = int(rng.integers(len(artists)))
        g = artist_genre[a]
        pool = GENRES[g]
        own = list(rng.choice(pool, size=int(rng.integers(2, 5)), replace=False))
        extra = list(rng.choice(EXTRA_TAGS, size=int(rng.integers(0, 3)), replace=False))
        tags = [g] + own + [t for t in extra if t not in own]
        weights = [100.0] + sorted((float(rng.integers(30, 95)) for _ in own), reverse=True) + \
                  [float(rng.integers(5, 40)) for t in extra if t not in own]
        title = " ".join(rng.choice(_WORDS, size=int(rng.integers(1, 3)), replace=False))
        rows.append({
            "track_id": f"t{i:06d}",
            "title": title,
            "artist": artists[a],
            "tags": tags,
            "tag_weights": weights,
            "genres": [g],
            "_pop": float(artist_pop[a] * rng.lognormal(0, 0.7)),
        })
    tracks = pd.DataFrame(rows)
    by_genre = {g: tracks.index[tracks["genres"].map(lambda x, g=g: x[0] == g)].to_numpy() for g in genres}
    pop = tracks["_pop"].to_numpy()

    def sample_tracks(idx: np.ndarray, k: int) -> np.ndarray:
        p = pop[idx] / pop[idx].sum()
        return rng.choice(idx, size=min(k, len(idx)), replace=False, p=p)

    # пользователи и история
    users, inter = [], []
    for u in range(n_users):
        fav = list(rng.choice(genres, size=int(rng.integers(1, 3)), replace=False))
        n_hist = int(rng.integers(5, 31))
        fav_idx = np.concatenate([by_genre[g] for g in fav])
        n_fav = int(round(n_hist * 0.85))
        hist = list(sample_tracks(fav_idx, n_fav)) + list(sample_tracks(tracks.index.to_numpy(), n_hist - n_fav))
        hist = list(dict.fromkeys(int(h) for h in hist))
        counts = rng.geometric(0.3, size=len(hist)).astype(float)
        uid = str(100000 + u)
        users.append({
            "user_id": uid,
            "fav": fav,
            "history": [{"track_id": tracks.at[h, "track_id"], "count": c} for h, c in zip(hist, counts)],
            "user_info": _user_info(rng, fav),
        })
        inter.extend({"user_id": uid, "track_id": tracks.at[h, "track_id"], "count": c} for h, c in zip(hist, counts))
    interactions = pd.DataFrame(inter)

    # популярность = число слушателей (как для Onion) + сглаживание по «скрытой» популярности
    listeners = interactions.groupby("track_id")["user_id"].nunique()
    tracks["popularity"] = tracks["track_id"].map(listeners).fillna(0.0) + tracks["_pop"]
    tracks = tracks.drop(columns=["_pop"])

    dialogs = [_make_dialog(rng, d, users[int(rng.integers(len(users)))], tracks, by_genre, artists, artist_genre)
               for d in range(n_dialogs)]
    return tracks, interactions, dialogs


def _user_info(rng, fav: List[str]) -> str:
    if rng.random() < 0.15:
        return ""
    gender = rng.choice(["male", "female"])
    age = int(rng.integers(16, 55))
    country = rng.choice(COUNTRIES)
    return f"{gender}, {age} years old, from {country}. Favourite genres: {', '.join(fav)}."


def _make_dialog(rng, d: int, user: dict, tracks: pd.DataFrame, by_genre, artists, artist_genre) -> Dict[str, Any]:
    genres = list(GENRES)
    g = user["fav"][0] if rng.random() < 0.6 else str(rng.choice(genres))
    t1, t2 = rng.choice(GENRES[g], size=2, replace=False)
    other = [t for t in EXTRA_TAGS + [x for gg in genres if gg != g for x in GENRES[gg]] if t not in GENRES[g]]
    ex = str(rng.choice(other)) if rng.random() < 0.35 else None
    seed_artist = None
    if rng.random() < 0.3:
        cand = [a for a, ag in zip(artists, artist_genre) if ag == g]
        seed_artist = str(rng.choice(cand)) if cand else None

    first = str(rng.choice([
        f"Hi! I'm looking for some {t1} {g} music.",
        f"Can you recommend something {t1} and {t2}?",
        f"I want to listen to {g}, preferably {t1}.",
        f"Give me some {g} tracks for tonight.",
    ]))
    if seed_artist:
        first += f" Something like {seed_artist}."
    msgs = [{"role": "user", "text": first}]
    if rng.random() < 0.5:
        msgs.append({"role": "assistant", "text": "Sure! Any preferences on the mood or energy?"})
        follow = f"Maybe more {t2}."
        if ex:
            follow += f" And please no {ex}."
        msgs.append({"role": "user", "text": follow})
    elif ex:
        msgs[0]["text"] += f" But without {ex}, please."

    # цели: популярные треки нужного жанра с тегом t1, не из истории и без исключённого тега
    heard = {h["track_id"] for h in user["history"]}
    idx = by_genre[g]
    ok = [i for i in idx
          if t1 in tracks.at[i, "tags"] and tracks.at[i, "track_id"] not in heard
          and (ex is None or ex not in tracks.at[i, "tags"])]
    if seed_artist:
        ok = [i for i in ok if tracks.at[i, "artist"] == seed_artist] or ok
    ok = sorted(ok, key=lambda i: -tracks.at[i, "popularity"] if "popularity" in tracks else 0)[:30]
    targets = [tracks.at[i, "track_id"] for i in rng.choice(ok, size=min(5, len(ok)), replace=False)] if ok else []

    return {
        "dialog_id": f"d{d:05d}",
        "user_id": user["user_id"],
        "user_info": user["user_info"],
        "history": user["history"],
        "messages": msgs,
        "target_track_ids": targets,
    }
