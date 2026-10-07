from recsys.data import build_profile
from recsys.data.synthetic import to_cyrillic
from recsys.dialog import RuleSummarizer
from recsys.retrieval import (AudioRetriever, BM25Retriever, LyricsRetriever, RelistenRetriever, TitleRetriever,
                              build_bm25_index)
from recsys.retrieval.bm25 import build_lyrics_index
from recsys.schemas import Context, HistoryItem, Message, Request


def _ctx(catalog, text, history=()):
    req = Request(dialog=[Message("user", text)], history=list(history))
    return Context.build(req, RuleSummarizer(catalog).summarize(req), build_profile(req.history, catalog),
                         exclude_listened=False)


def test_title_finds_transliterated_name(data):
    row = data.catalog.df.iloc[7]
    query = to_cyrillic(f"{row['artist']} {row['title']}")  # «вельвет оулс ...»
    found = [c.track_id for c in TitleRetriever(data.catalog).search(_ctx(data.catalog, query))]
    assert row["track_id"] in found[:5]


def test_title_ignores_generic_queries(data):
    assert TitleRetriever(data.catalog).search(_ctx(data.catalog, "спокойная музыка для учёбы")) == []


def test_relisten_ranks_own_tracks_by_query(data):
    cat = data.catalog
    jazz = cat.df.index[cat.df["genres"].map(lambda g: g == ["jazz"])][:3]
    metal = cat.df.index[cat.df["genres"].map(lambda g: g == ["metal"])][:3]
    history = [HistoryItem(cat.df.at[i, "track_id"], count=1, timestamp=100 - n) for n, i in enumerate(list(metal) + list(jazz))]
    out = RelistenRetriever(cat, build_bm25_index(cat)).search(_ctx(cat, "джаз", history))
    assert {c.track_id for c in out} == {h.track_id for h in history}
    assert {c.track_id for c in out[:3]} == {cat.df.at[i, "track_id"] for i in jazz}


def test_bm25_uses_country_and_decade(data):
    cat = data.catalog
    ctx = _ctx(cat, "американский рок 80-х")
    assert {"c:us", "t:80s"} <= set(BM25Retriever(cat, build_bm25_index(cat)).build_query(ctx))


def test_lyrics_and_audio(data):
    cat = data.catalog
    row = cat.df[cat.df["lyrics"].str.len() > 0].iloc[0]
    words = " ".join(row["lyrics"].split()[10:16])
    found = [c.track_id for c in LyricsRetriever(cat, build_lyrics_index(cat)).search(_ctx(cat, f"песня со словами {words}"))]
    assert row["track_id"] in found[:10]
    req = data.requests[0]
    audio = AudioRetriever(cat).search(_ctx(cat, "", req.history))
    assert len(audio) == 200 and audio[0].score >= audio[-1].score
