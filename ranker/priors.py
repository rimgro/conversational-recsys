"""CTR артиста = доля таргетов: сколько раз треки артиста были таргетом / сколько у него слушателей.

Считается только по «чужим» пользователям (часть A для обучения, весь train для теста),
иначе признак подсказывает ответ. Сглаживание к общему уровню p0 с силой alpha.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ranker.data import Split


@dataclass
class Priors:
    p0: float
    artist: pd.Series      # artist -> ctr

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.artist.rename("ctr").reset_index().to_parquet(path / "artist.parquet")
        (path / "priors.json").write_text(json.dumps({"p0": self.p0}))

    @classmethod
    def load(cls, path: str | Path) -> "Priors":
        path = Path(path)
        a = pd.read_parquet(path / "artist.parquet")
        return cls(p0=json.loads((path / "priors.json").read_text())["p0"], artist=a.set_index("artist")["ctr"])


def compute_priors(split: Split, tracks: pd.DataFrame, alpha: float = 20.0) -> Priors:
    artist_of = tracks["artist"].to_numpy()
    tgt = pd.Series(artist_of[split.queries["target_idx"].to_numpy()])
    lst = pd.DataFrame({"user_id": split.history["user_id"].to_numpy(),
                        "artist": artist_of[split.history["track_idx"].to_numpy()]}).drop_duplicates()
    c = pd.concat([tgt[tgt != ""].value_counts().rename("t"),
                   lst.loc[lst["artist"] != "", "artist"].value_counts().rename("n")], axis=1).fillna(0.0)
    p0 = float(c["t"].sum() / max(c["n"].sum(), 1.0))
    ctr = ((c["t"] + alpha * p0) / (c["n"] + alpha)).rename("ctr")
    ctr.index.name = "artist"
    return Priors(p0=p0, artist=ctr)


def lookup(priors: Priors, artists: np.ndarray) -> np.ndarray:
    """CTR артиста; неизвестный артист -> p0."""
    art = priors.artist.reindex(artists).to_numpy(dtype=float)
    return np.where(np.isnan(art), priors.p0, art)
