"""Признаки ранкера «на момент запроса» (point-in-time), polars. Новый формат датасета (queries + qrels).

Вариант B: в train признаки пересчитываются на момент каждого запроса — cutoff = начало дня его ts
(всё, что слушали до cutoff: история до окна + позитивы окна раньше cutoff; сам таргет в окне — в момент ts,
поэтому не попадает). В test все запросы видят один срез: cutoff = начало тестового окна.

    python -m ranker.pit --data-dir ../new_ds_v1/music4all_crs --split train \
        --queries ranker_train_queries.csv --candidates candidates.parquet --out features_train.parquet

candidates: query_id, m4a_id (+ любые колонки источников, например rrf_score, n_sources — копируются как есть).
Без --candidates можно --demo-candidates N (таргет + треки истории + случайные) — только для проверки кода.
"""
from __future__ import annotations

import argparse
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl
import pyarrow.parquet as pq

DAY = 86_400
WINDOW_START = {
    "train": int(datetime(2020, 1, 21, tzinfo=timezone.utc).timestamp()),
    "test_public": int(datetime(2020, 2, 20, tzinfo=timezone.utc).timestamp()),
    "test_private": int(datetime(2020, 2, 20, tzinfo=timezone.utc).timestamp()),
}
WINDOW_YEAR = 2020
CTR_DAYS = 30          # ctr = доля слушателей, слушавших трек (артиста) в последние CTR_DAYS дней до cutoff
ALPHA = 20.0           # сглаживание ctr к среднему p0
TOP_GENRES = 40        # категории top_genre: самые частые жанры + other / unknown
AGE_BUCKETS = ["10s", "20s", "30s", "40s", "50+", "unknown"]
GENDERS = ["m", "f", "unknown"]

FEATURES = [
    # пользовательские
    "age_bucket", "gender", "top_genre",
    # айтемные
    "popularity", "ctr_item", "ctr_artist",
    # кросс (пользователь x трек)
    "age_at_release", "year_diff_history", "in_history", "log_history_count", "artist_share",
    "log_album_listens", "lang_share", "genre_share", "profile_tag_score",
    # запрос x трек
    "query_tag_frac",
]
CATEGORICAL = ["age_bucket", "gender", "top_genre"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- каталог

def _csv_list(col: str) -> pl.Expr:
    return (pl.col(col).fill_null("").str.to_lowercase().str.split(",")
            .list.eval(pl.element().str.strip_chars()).list.eval(pl.element().filter(pl.element() != "")))


def load_tracks(data_dir: Path) -> pl.DataFrame:
    m = pl.read_parquet(data_dir / "tracks_meta.parquet",
                        columns=["m4a_id", "m4a_artist", "m4a_album", "release_year", "lang",
                                 "spotify_popularity", "m4a_genres_full", "m4a_tags_full"])
    year = pl.col("release_year").cast(pl.Float64, strict=False)
    t = m.with_row_index("track_idx").select(
        pl.col("track_idx").cast(pl.Int32),
        "m4a_id",
        pl.col("m4a_artist").fill_null("").alias("artist"),
        pl.when(pl.col("m4a_album").fill_null("") != "")
          .then(pl.col("m4a_artist").fill_null("") + "|" + pl.col("m4a_album")).alias("album"),
        pl.when(year > 1900).then(year).alias("year"),
        pl.col("lang").fill_null("").str.to_lowercase().alias("lang"),
        pl.col("spotify_popularity").cast(pl.Float32).alias("popularity"),
        _csv_list("m4a_genres_full").list.unique().alias("genres"),
        pl.concat_list(_csv_list("m4a_genres_full"), _csv_list("m4a_tags_full")).list.unique().alias("tags"),
    )
    # целочисленные коды для джойнов
    return t.with_columns(
        pl.col("artist").rank("dense").cast(pl.Int32).alias("artist_idx"),
        pl.col("album").rank("dense").cast(pl.Int32).alias("album_idx"),
    )


# ---------------------------------------------------------------- пользователи, история, окно

def age_bucket_expr(age: pl.Expr) -> pl.Expr:
    return (pl.when(age.is_null() | (age < 10) | (age > 100)).then(pl.lit("unknown"))
            .when(age >= 50).then(pl.lit("50+"))
            .otherwise((age // 10 * 10).cast(pl.Int32).cast(pl.Utf8) + "s"))


def load_users(data_dir: Path, split: str, tracks: pl.DataFrame, user_ids: Optional[set],
               with_window: bool, batch_rows: int = 400):
    """-> users (user_id, age, gender, age_bucket), hist (user_id, track_idx, cnt, last_ts),
    window (user_id, track_idx, ts) — позитивы окна (только для train-режима)."""
    ids = tracks.select("m4a_id", "track_idx")
    pf = pq.ParquetFile(data_dir / f"{split}.parquet")
    cols = ["user_id", "history", "history_ts", "user_demographics"] + (["positives"] if with_window else [])
    users, hist, window = [], [], []
    for i, batch in enumerate(pf.iter_batches(batch_size=batch_rows, columns=cols)):
        b = pl.from_arrow(batch)
        if user_ids is not None:
            b = b.filter(pl.col("user_id").is_in(list(user_ids)))
        if b.height == 0:
            continue
        users.append(b.select(
            "user_id",
            pl.col("user_demographics").struct.field("age").cast(pl.Float64).alias("age"),
            pl.col("user_demographics").struct.field("gender").alias("gender")))
        hist.append(b.select("user_id", "history", "history_ts").explode("history", "history_ts")
                    .join(ids, left_on="history", right_on="m4a_id", how="inner")
                    .group_by("user_id", "track_idx")
                    .agg(pl.len().cast(pl.Float32).alias("cnt"), pl.col("history_ts").max().alias("last_ts")))
        if with_window:
            window.append(b.select("user_id", "positives").explode("positives").unnest("positives")
                          .select("user_id", "m4a_id", "ts")
                          .join(ids, on="m4a_id", how="inner").select("user_id", "track_idx", "ts"))
        if (i + 1) % 10 == 0:
            log(f"  {split}: {(i + 1) * batch_rows} пользователей")
    users_df = pl.concat(users).with_columns(
        pl.when(pl.col("age") > 0).then(pl.col("age")).alias("age"),
        pl.when(pl.col("gender").is_in(["m", "f"])).then(pl.col("gender")).otherwise(pl.lit("unknown")).alias("gender"),
    ).with_columns(age_bucket_expr(pl.col("age")).alias("age_bucket"))
    window_df = pl.concat(window) if window else pl.DataFrame(
        schema={"user_id": pl.Int64, "track_idx": pl.Int32, "ts": pl.Int64})
    return users_df, pl.concat(hist), window_df


def load_queries(data_dir: Path, split: str, ids: Optional[pl.DataFrame], mode: str) -> pl.DataFrame:
    q = pl.read_parquet(data_dir / f"{split}_queries.parquet",
                        columns=["query_id", "user_id", "query", "query_type"])
    if ids is not None:
        q = q.join(ids.select("query_id"), on="query_id", how="semi")
    r = pl.read_parquet(data_dir / f"{split}_qrels.parquet",
                        columns=["query_id", "target_m4a_id", "exclude_m4a_id", "exclude_artist", "ts"])
    q = q.join(r, on="query_id", how="left")
    if mode == "train":
        cutoff = (pl.col("ts") // DAY) * DAY
    else:                                   # test: ts запроса неизвестен, у всех один срез
        cutoff = pl.lit(WINDOW_START[split], dtype=pl.Int64)
    return q.with_columns(cutoff.alias("cutoff"))


# ---------------------------------------------------------------- агрегаты истории

def _with_meta(df: pl.DataFrame, tracks: pl.DataFrame, cols: List[str]) -> pl.DataFrame:
    return df.join(tracks.select(["track_idx"] + cols), on="track_idx", how="left")


def aggregates(ev: pl.DataFrame, key: str, tracks: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    """ev: key, track_idx, cnt -> счётчики по (key, track / artist / album / lang / genre / tag) и итоги по key."""
    e = _with_meta(ev, tracks, ["artist_idx", "album_idx", "lang", "year", "genres", "tags"])
    out = {
        "track": e.group_by(key, "track_idx").agg(pl.col("cnt").sum()),
        "artist": e.group_by(key, "artist_idx").agg(pl.col("cnt").sum()),
        "album": e.filter(pl.col("album_idx").is_not_null()).group_by(key, "album_idx").agg(pl.col("cnt").sum()),
        "lang": e.filter(pl.col("lang") != "").group_by(key, "lang").agg(pl.col("cnt").sum()),
        "genre": e.select(key, "cnt", "genres").explode("genres").drop_nulls("genres")
                  .group_by(key, "genres").agg(pl.col("cnt").sum()).rename({"genres": "genre"}),
        "tag": e.select(key, "cnt", "tags").explode("tags").drop_nulls("tags")
                .group_by(key, "tags").agg(pl.col("cnt").sum()).rename({"tags": "tag"}),
        "total": e.group_by(key).agg(
            pl.col("cnt").sum().alias("total"),
            pl.col("cnt").filter(pl.col("lang") != "").sum().alias("lang_total"),
            (pl.col("cnt") * pl.col("year")).filter(pl.col("year").is_not_null()).sum().alias("year_sum"),
            pl.col("cnt").filter(pl.col("year").is_not_null()).sum().alias("year_n"),
            (pl.col("cnt") * pl.col("tags").list.len()).sum().alias("tag_total"),
        ),
    }
    return out


def query_deltas(queries: pl.DataFrame, window: pl.DataFrame, tracks: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    """Позитивы окна этого же пользователя до cutoff запроса -> те же агрегаты, но по query_id."""
    ev = (queries.select("query_id", "user_id", "cutoff")
          .join(window, on="user_id", how="inner")
          .filter(pl.col("ts") < pl.col("cutoff"))
          .select("query_id", "track_idx", pl.lit(1.0, dtype=pl.Float32).alias("cnt")))
    return aggregates(ev, "query_id", tracks)


# ---------------------------------------------------------------- ctr на момент cutoff

def ctr_tables(hist: pl.DataFrame, window: pl.DataFrame, tracks: pl.DataFrame, cutoffs: List[int],
               level: str) -> pl.DataFrame:
    """cutoff, <level>, ctr: (слушатели за последние CTR_DAYS дней + ALPHA * p0) / (все слушатели до cutoff + ALPHA).
    level = track_idx | artist_idx. Слушатель = пользователь с хотя бы одним событием до cutoff."""
    key = level
    if key == "track_idx":
        h = hist.select("user_id", key, "last_ts")
        w = window.select("user_id", key, "ts")
    else:
        h = _with_meta(hist, tracks, [key]).group_by("user_id", key).agg(pl.col("last_ts").max())
        w = _with_meta(window, tracks, [key]).select("user_id", key, "ts")
    h_users = h.group_by(key).agg(pl.len().alias("h_users"))
    w_new = w.join(h, on=["user_id", key], how="anti").group_by("user_id", key).agg(pl.col("ts").min())
    lo = min(cutoffs) - CTR_DAYS * DAY
    h_recent = h.filter(pl.col("last_ts") >= lo).rename({"last_ts": "ts"})
    out = []
    for c in sorted(set(cutoffs)):
        u_all = (h_users.join(w_new.filter(pl.col("ts") < c).group_by(key).agg(pl.len().alias("w_users")),
                              on=key, how="full", coalesce=True)
                 .select(key, (pl.col("h_users").fill_null(0) + pl.col("w_users").fill_null(0)).alias("u_all")))
        u_30 = (pl.concat([h_recent, w.filter(pl.col("ts") < c)])
                .group_by("user_id", key).agg(pl.col("ts").max())
                .filter(pl.col("ts") >= c - CTR_DAYS * DAY)
                .group_by(key).agg(pl.len().alias("u_30")))
        t = u_all.join(u_30, on=key, how="left").with_columns(pl.col("u_30").fill_null(0))
        p0 = t["u_30"].sum() / max(t["u_all"].sum(), 1)
        out.append(t.select(pl.lit(c, dtype=pl.Int64).alias("cutoff"), key,
                            ((pl.col("u_30") + ALPHA * p0) / (pl.col("u_all") + ALPHA)).cast(pl.Float32).alias("ctr"),
                            pl.lit(p0, dtype=pl.Float32).alias("p0")))
    return pl.concat(out)


# ---------------------------------------------------------------- признаки

_TOKEN = re.compile(r"[^\w']+")


def query_ngrams(queries: pl.DataFrame, vocab: set) -> pl.DataFrame:
    """query_id, tag: n-граммы запроса (1–3 слова), которые есть среди тегов/жанров каталога."""
    rows_q, rows_t = [], []
    for qid, text in zip(queries["query_id"], queries["query"]):
        toks = [t for t in _TOKEN.split((text or "").lower()) if t]
        grams = {" ".join(toks[i:i + n]) for n in (1, 2, 3) for i in range(len(toks) - n + 1)}
        for g in grams & vocab:
            rows_q.append(qid), rows_t.append(g)
    return pl.DataFrame({"query_id": rows_q, "tag": rows_t}, schema={"query_id": pl.Utf8, "tag": pl.Utf8})


def _add(df: pl.DataFrame, base: pl.DataFrame, delta: pl.DataFrame, keys_base, keys_delta, name: str) -> pl.DataFrame:
    """name = счётчик из истории (по user_id) + счётчик из окна до cutoff (по query_id)."""
    b = base.rename({"cnt": "_b"}) if "cnt" in base.columns else base
    d = delta.rename({"cnt": "_d"}) if "cnt" in delta.columns else delta
    return (df.join(b, on=keys_base, how="left").join(d, on=keys_delta, how="left")
            .with_columns((pl.col("_b").fill_null(0) + pl.col("_d").fill_null(0)).alias(name)).drop("_b", "_d"))


def top_genre_per_query(q: pl.DataFrame, base: Dict, delta: Dict, top: List[str]) -> pl.DataFrame:
    g = (q.select("query_id", "user_id").join(base["genre"], on="user_id", how="inner")
         .select("query_id", "genre", "cnt"))
    g = pl.concat([g, delta["genre"].join(q.select("query_id"), on="query_id", how="semi").select("query_id", "genre", "cnt")])
    best = (g.group_by("query_id", "genre").agg(pl.col("cnt").sum())
            .sort(["query_id", "cnt", "genre"], descending=[False, True, False])
            .group_by("query_id", maintain_order=True).first())
    return best.select("query_id", pl.when(pl.col("genre").is_in(top)).then(pl.col("genre"))
                       .otherwise(pl.lit("other")).alias("top_genre"))


def build_chunk(cands: pl.DataFrame, q: pl.DataFrame, users: pl.DataFrame, tracks: pl.DataFrame,
                base: Dict, delta: Dict, ctr_i: pl.DataFrame, ctr_a: pl.DataFrame,
                qgrams: pl.DataFrame, top: List[str]) -> pl.DataFrame:
    meta = tracks.select("track_idx", "m4a_id", "artist", "artist_idx", "album_idx", "lang", "year", "popularity",
                         "genres", "tags")
    targets = tracks.select(pl.col("m4a_id").alias("target_m4a_id"), pl.col("track_idx").alias("target_idx"))
    qq = q.join(targets, on="target_m4a_id", how="left")
    df = (cands.join(meta.select("m4a_id", "track_idx"), on="m4a_id", how="inner")
          .join(qq.select("query_id", "user_id", "cutoff", "query_type", "target_idx", "exclude_m4a_id",
                          "exclude_artist"), on="query_id", how="inner")
          .join(meta.drop("m4a_id"), on="track_idx", how="left"))
    # similar_to: сам референс X и его артист в оценке выкидываются — выкидываем и из групп
    df = df.filter(~((pl.col("query_type") == "similar_to") &
                     ((pl.col("m4a_id") == pl.col("exclude_m4a_id")) |
                      (pl.col("artist") == pl.col("exclude_artist").fill_null("\x00")))))
    df = df.with_columns((pl.col("track_idx") == pl.col("target_idx")).fill_null(False).cast(pl.Int8).alias("label"))
    df = df.with_row_index("_row")

    df = _add(df, base["track"], delta["track"], ["user_id", "track_idx"], ["query_id", "track_idx"], "h_cnt")
    df = _add(df, base["artist"], delta["artist"], ["user_id", "artist_idx"], ["query_id", "artist_idx"], "a_cnt")
    df = _add(df, base["album"], delta["album"], ["user_id", "album_idx"], ["query_id", "album_idx"], "al_cnt")
    df = _add(df, base["lang"], delta["lang"], ["user_id", "lang"], ["query_id", "lang"], "l_cnt")
    bt = base["total"]
    dt = delta["total"].rename({c: f"d_{c}" for c in delta["total"].columns if c != "query_id"})
    df = df.join(bt, on="user_id", how="left").join(dt, on="query_id", how="left")
    tot = {c: pl.col(c).fill_null(0) + pl.col(f"d_{c}").fill_null(0)
           for c in ("total", "lang_total", "year_sum", "year_n", "tag_total")}

    # жанры и теги трека: развернуть, сложить счётчики истории и окна, свернуть обратно по строке
    def exploded(col: str, name: str, base_t: pl.DataFrame, delta_t: pl.DataFrame) -> pl.DataFrame:
        e = df.select("_row", "user_id", "query_id", pl.col(col).alias(name)).explode(name).drop_nulls(name)
        return _add(e, base_t, delta_t, ["user_id", name], ["query_id", name], "c")

    g = exploded("genres", "genre", base["genre"], delta["genre"]).group_by("_row").agg(pl.col("c").max().alias("g_max"))
    tg = exploded("tags", "tag", base["tag"], delta["tag"]).group_by("_row").agg(pl.col("c").sum().alias("t_sum"))
    qt = (df.select("_row", "query_id", pl.col("tags").alias("tag")).explode("tag").drop_nulls("tag")
          .join(qgrams, on=["query_id", "tag"], how="semi").group_by("_row").agg(pl.len().alias("q_hits")))
    q_n = qgrams.group_by("query_id").agg(pl.len().alias("q_n"))
    df = (df.join(g, on="_row", how="left").join(tg, on="_row", how="left").join(qt, on="_row", how="left")
          .join(q_n, on="query_id", how="left")
          .join(users.select("user_id", "age", "age_bucket", "gender"), on="user_id", how="left")
          .join(top_genre_per_query(q, base, delta, top), on="query_id", how="left")
          .join(ctr_i.rename({"ctr": "ctr_item", "p0": "p0_i"}), on=["cutoff", "track_idx"], how="left")
          .join(ctr_a.rename({"ctr": "ctr_artist", "p0": "p0_a"}), on=["cutoff", "artist_idx"], how="left"))
    p0 = {k: v.group_by("cutoff").agg(pl.col("p0").first().alias(k)) for k, v in (("p0i", ctr_i), ("p0a", ctr_a))}
    df = df.join(p0["p0i"], on="cutoff", how="left").join(p0["p0a"], on="cutoff", how="left")

    total = tot["total"]
    mean_year = tot["year_sum"] / tot["year_n"]
    f = df.with_columns(
        pl.col("age_bucket").fill_null("unknown"),
        pl.col("gender").fill_null("unknown"),
        pl.col("top_genre").fill_null("unknown"),
        pl.col("ctr_item").fill_null(pl.col("p0i")),
        pl.col("ctr_artist").fill_null(pl.col("p0a")),
        (pl.col("age") - (WINDOW_YEAR - pl.col("year"))).alias("age_at_release"),
        (pl.col("year") - mean_year).abs().alias("year_diff_history"),
        (pl.col("h_cnt") > 0).cast(pl.Int8).alias("in_history"),
        pl.col("h_cnt").log1p().alias("log_history_count"),
        pl.when(total > 0).then(pl.col("a_cnt") / total).otherwise(0.0).alias("artist_share"),
        pl.col("al_cnt").log1p().alias("log_album_listens"),
        pl.when(pl.col("lang") == "").then(None)
          .when(tot["lang_total"] > 0).then(pl.col("l_cnt") / tot["lang_total"]).otherwise(0.0).alias("lang_share"),
        pl.when(total > 0).then(pl.col("g_max").fill_null(0) / total).otherwise(0.0).alias("genre_share"),
        pl.when(tot["tag_total"] > 0).then(pl.col("t_sum").fill_null(0) / tot["tag_total"]).otherwise(0.0)
          .alias("profile_tag_score"),
        pl.when(pl.col("q_n") > 0).then(pl.col("q_hits").fill_null(0) / pl.col("q_n")).otherwise(0.0)
          .alias("query_tag_frac"),
    )
    passthrough = [c for c in cands.columns if c not in ("query_id", "m4a_id")]
    out = f.select(["query_id", "m4a_id", "query_type", "label"] + passthrough + FEATURES)
    return out.with_columns([pl.col(c).cast(pl.Int8 if c == "in_history" else pl.Float32)
                             for c in FEATURES if c not in CATEGORICAL])


def user_table(q: pl.DataFrame, users: pl.DataFrame, base: Dict, delta: Dict, top: List[str],
               chunk: int = 20_000) -> pl.DataFrame:
    """Пользовательские признаки на момент каждого запроса: query_id, user_id, cutoff, age, age_bucket, gender, top_genre."""
    tg = pl.concat([top_genre_per_query(q.slice(i, chunk), base, delta, top) for i in range(0, q.height, chunk)])
    return (q.select("query_id", "user_id", "query_type", "cutoff")
            .join(users.select("user_id", "age", "age_bucket", "gender"), on="user_id", how="left")
            .join(tg, on="query_id", how="left")
            .with_columns(pl.col("age_bucket").fill_null("unknown"), pl.col("gender").fill_null("unknown"),
                          pl.col("top_genre").fill_null("unknown")))


def item_table(tracks: pl.DataFrame, ctr_i: pl.DataFrame, ctr_a: pl.DataFrame) -> pl.DataFrame:
    """Айтемные признаки для всех треков на каждый cutoff: cutoff, m4a_id, popularity, ctr_item, ctr_artist."""
    cut = ctr_i.select("cutoff", pl.col("p0").alias("p0i")).unique("cutoff").join(
        ctr_a.select("cutoff", pl.col("p0").alias("p0a")).unique("cutoff"), on="cutoff")
    return (cut.join(tracks.select("track_idx", "artist_idx", "m4a_id", "popularity"), how="cross")
            .join(ctr_i.select("cutoff", "track_idx", pl.col("ctr").alias("ctr_item")), on=["cutoff", "track_idx"], how="left")
            .join(ctr_a.select("cutoff", "artist_idx", pl.col("ctr").alias("ctr_artist")), on=["cutoff", "artist_idx"], how="left")
            .select("cutoff", "m4a_id", "popularity",
                    pl.col("ctr_item").fill_null(pl.col("p0i")), pl.col("ctr_artist").fill_null(pl.col("p0a")))
            .sort("cutoff", "m4a_id"))


def demo_candidates(q: pl.DataFrame, hist: pl.DataFrame, tracks: pl.DataFrame, n: int, seed: int = 0) -> pl.DataFrame:
    """Только для проверки кода: таргет + до n/2 самых слушаемых треков истории + случайные."""
    rng = np.random.default_rng(seed)
    ids = tracks["m4a_id"].to_numpy()
    top_hist = (hist.sort("cnt", descending=True).group_by("user_id", maintain_order=True).head(n // 2)
                .join(tracks.select("track_idx", "m4a_id"), on="track_idx").select("user_id", "m4a_id"))
    rand = pl.DataFrame({"query_id": np.repeat(q["query_id"].to_numpy(), n // 2),
                         "m4a_id": ids[rng.integers(0, len(ids), len(q) * (n // 2))]})
    parts = [q.select("query_id", pl.col("target_m4a_id").alias("m4a_id")),
             q.select("query_id", "user_id").join(top_hist, on="user_id").select("query_id", "m4a_id"), rand]
    return pl.concat(parts).unique(["query_id", "m4a_id"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="train", help="train | test_public")
    ap.add_argument("--mode", choices=["train", "test"], help="по умолчанию: train для train, test для остальных")
    ap.add_argument("--queries", help="csv/parquet с колонкой query_id (например ranker_train_queries.csv)")
    ap.add_argument("--candidates", help="parquet: query_id, m4a_id, [колонки источников]")
    ap.add_argument("--demo-candidates", type=int, default=0)
    ap.add_argument("--chunk-queries", type=int, default=5000)
    ap.add_argument("--tables-only", action="store_true",
                    help="без кандидатов: только user (на запрос) и item (на cutoff x трек) таблицы; --out = папка")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    data_dir = Path(args.data_dir)
    mode = args.mode or ("train" if args.split == "train" else "test")
    if not (args.candidates or args.demo_candidates or args.tables_only):
        ap.error("нужен --candidates или --demo-candidates")

    log("каталог")
    tracks = load_tracks(data_dir)
    ids = None
    if args.queries:
        ids = pl.read_csv(args.queries) if args.queries.endswith(".csv") else pl.read_parquet(args.queries)
    q = load_queries(data_dir, args.split, ids, mode)
    log(f"запросов {q.height}, режим {mode}")

    log("история пользователей")
    # все пользователи сплита: ctr — статистика по всей аудитории, не только по выбранным запросам
    users, hist, window = load_users(data_dir, args.split, tracks, None, with_window=(mode == "train"))
    log(f"пользователей {users.height}, пар (user, track) {hist.height}, позитивов окна {window.height}")
    base = aggregates(hist.select("user_id", "track_idx", "cnt"), "user_id", tracks)
    delta = query_deltas(q, window, tracks)

    log("ctr")
    cutoffs = q["cutoff"].unique().to_list()
    ctr_i = ctr_tables(hist, window, tracks, cutoffs, "track_idx")
    ctr_a = ctr_tables(hist, window, tracks, cutoffs, "artist_idx")
    top = (base["genre"].group_by("genre").agg(pl.col("cnt").sum()).sort("cnt", descending=True)
           .head(TOP_GENRES)["genre"].to_list())

    if args.tables_only:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        ut = user_table(q, users, base, delta, top)
        it = item_table(tracks, ctr_i, ctr_a)
        ut.write_parquet(out_dir / f"user_features_{args.split}.parquet")
        it.write_parquet(out_dir / f"item_features_{args.split}.parquet")
        log(f"готово: user {ut.shape}, item {it.shape} ({it['cutoff'].n_unique()} cutoff) -> {out_dir}")
        return

    vocab = set(tracks.select(pl.col("tags").explode()).drop_nulls()["tags"].to_list())
    qgrams = query_ngrams(q, vocab)
    q_users = q.select("user_id").unique()
    if args.candidates:
        cands = pl.read_parquet(args.candidates).join(q.select("query_id"), on="query_id", how="semi")
    else:
        cands = demo_candidates(q, hist.join(q_users, on="user_id", how="semi"), tracks, args.demo_candidates)
    log(f"кандидатов {cands.height}")

    out_path = Path(args.out)
    parts_dir = out_path.with_suffix(".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    qids = q["query_id"].to_list()
    for i in range(0, len(qids), args.chunk_queries):
        chunk_ids = pl.DataFrame({"query_id": qids[i:i + args.chunk_queries]})
        qc = q.join(chunk_ids, on="query_id", how="semi")
        cc = cands.join(chunk_ids, on="query_id", how="semi")
        build_chunk(cc, qc, users, tracks, base, delta, ctr_i, ctr_a, qgrams, top) \
            .write_parquet(parts_dir / f"part-{i // args.chunk_queries:05d}.parquet")
        log(f"  признаки: {min(i + args.chunk_queries, len(qids))}/{len(qids)} запросов")
    pl.scan_parquet(parts_dir / "*.parquet").sink_parquet(out_path)
    res = pl.scan_parquet(out_path)
    stats = res.select(pl.len().alias("rows"), pl.col("query_id").n_unique().alias("queries"),
                       pl.col("label").sum().alias("positives")).collect()
    log(f"готово: {out_path} {stats.to_dicts()[0]}")


if __name__ == "__main__":
    main()
