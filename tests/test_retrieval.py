import numpy as np

from recsys.data import build_profile
from recsys.dialog import RuleSummarizer
from recsys.retrieval import (AudioRetriever, BM25Retriever, LyricsRetriever, RelistenRetriever, TitleRetriever,
                              build_bm25_index)
from recsys.retrieval.local_index import build_lyrics_index
from recsys.schemas import Context, HistoryItem, Message, Request


def _ctx(catalog, text, history=()):
    req = Request(dialog=[Message("user", text)], history=list(history))
    return Context.build(req, RuleSummarizer(catalog).summarize(req), build_profile(req.history, catalog),
                         exclude_listened=False)


def test_title_finds_exact_request(data):
    row = data.catalog.df.iloc[7]
    query = f"play {row['title'].lower()} by {row['artist'].lower()}"
    found = [c.track_id for c in TitleRetriever(data.catalog).search(_ctx(data.catalog, query))]
    assert row["track_id"] in found[:5]


def test_title_ignores_generic_queries(data):
    assert TitleRetriever(data.catalog).search(_ctx(data.catalog, "calm music for studying")) == []


def test_relisten_ranks_own_tracks_by_query(data):
    cat = data.catalog
    jazz = cat.df.index[cat.df["genres"].map(lambda g: g == ["jazz"])][:3]
    metal = cat.df.index[cat.df["genres"].map(lambda g: g == ["metal"])][:3]
    history = [HistoryItem(cat.df.at[i, "track_id"], count=1, timestamp=100 - n) for n, i in enumerate(list(metal) + list(jazz))]
    out = RelistenRetriever(cat, build_bm25_index(cat)).search(_ctx(cat, "jazz", history))
    assert {c.track_id for c in out} == {h.track_id for h in history}
    assert {c.track_id for c in out[:3]} == {cat.df.at[i, "track_id"] for i in jazz}


def test_bm25_uses_country_and_decade(data):
    cat = data.catalog
    ctx = _ctx(cat, "american rock from the 80s")
    assert {"c:us", "t:80s"} <= set(BM25Retriever(cat, build_bm25_index(cat)).build_query(ctx))


def test_lyrics_and_audio(data):
    cat = data.catalog
    row = cat.df[cat.df["lyrics"].str.len() > 0].iloc[0]
    words = " ".join(row["lyrics"].split()[10:16])
    found = [c.track_id for c in LyricsRetriever(cat, build_lyrics_index(cat)).search(_ctx(cat, f"the song that goes {words}"))]
    assert row["track_id"] in found[:10]
    req = data.requests[0]
    audio = AudioRetriever(cat).search(_ctx(cat, "", req.history))
    assert len(audio) == 200 and audio[0].score >= audio[-1].score


def test_audio_uses_reference_track(data):
    """similar_to: аудио-поиск идёт от трека-образца, а не от всей истории."""
    cat = data.catalog
    row = cat.df.iloc[3]
    req = data.requests[0]
    ctx = _ctx(cat, f"tracks like {row['title'].lower()} by {row['artist'].lower()} but from other artists", req.history)
    audio = AudioRetriever(cat).search(ctx)
    pos = cat.positions([row["track_id"]])[0]
    with np.errstate(all="ignore"):  # numpy + Accelerate (macOS): ложные предупреждения matmul
        sims = cat.embeddings @ cat.embeddings[pos]
    assert {c.track_id for c in audio[:5]} <= set(cat.track_ids[sims.argsort()[::-1][:50]])
