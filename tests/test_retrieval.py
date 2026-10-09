import numpy as np

from recsys.data import build_profile
from recsys.dialog import RuleSummarizer
from recsys.retrieval import AudioRetriever, BM25Retriever, RelistenRetriever, build_card_index
from recsys.retrieval.sources import query_terms
from recsys.schemas import Context, HistoryItem, Message, Request


def _ctx(catalog, text, history=()):
    req = Request(dialog=[Message("user", text)], history=list(history))
    return Context.build(req, RuleSummarizer(catalog).summarize(req), build_profile(req.history, catalog),
                         exclude_listened=False)


def test_relisten_ranks_own_tracks_by_query(data):
    cat = data.catalog
    jazz = cat.df.index[cat.df["genres"].map(lambda g: g == ["jazz"])][:3]
    metal = cat.df.index[cat.df["genres"].map(lambda g: g == ["metal"])][:3]
    history = [HistoryItem(cat.df.at[i, "track_id"], count=1, timestamp=100 - n) for n, i in enumerate(list(metal) + list(jazz))]
    out = RelistenRetriever(cat, build_card_index(cat)).search(_ctx(cat, "jazz", history))
    assert {c.track_id for c in out} == {h.track_id for h in history}
    assert {c.track_id for c in out[:3]} == {cat.df.at[i, "track_id"] for i in jazz}


def test_query_terms_use_country_and_decade(data):
    """Запрос relisten к локальному индексу: страна и эпоха — отдельными токенами."""
    ctx = _ctx(data.catalog, "american rock from the 80s")
    assert {"c:us", "t:80s"} <= set(query_terms(ctx.summary))


def test_audio_from_history(data):
    cat = data.catalog
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


def test_card_index_fields_and_cache(data, tmp_path):
    """Поля карточки складываются с весами; сохранённый индекс даёт те же скоры."""
    import pandas as pd
    from recsys.retrieval import CardIndex
    from recsys.retrieval.sources import query_terms
    cat = data.catalog
    row = cat.df.iloc[5]
    meta = pd.DataFrame({"m4a_id": [row["track_id"]], "lyrics": ["purple submarine dancing on the moon"]})
    index = build_card_index(cat, meta)
    assert set(index.fields) == {"meta", "lyrics"}  # остальных колонок CARD_FIELDS в meta нет
    q = query_terms(_ctx(cat, "purple submarine").summary)
    lyr = index.scores(q, {"lyrics": 1.0})
    assert lyr.argmax() == 5 and (lyr > 0).sum() == 1
    assert not index.scores(q, {"meta": 1.0})[5] and not index.scores(q, {"lyrics": 0.0}).any()  # вес 0 — поля нет
    path = str(tmp_path / "idx.npz")
    index.save(path)
    again = CardIndex.load(path)
    assert np.allclose(again.scores(q), index.scores(q)) and set(again.fields) == set(index.fields)


def test_bm25_searches_catalog(data):
    cat = data.catalog
    req = data.requests[0]
    ctx = _ctx(cat, "jazz", req.history)
    found = BM25Retriever(cat, build_card_index(cat), top_k=50).search(ctx)
    assert 0 < len(found) <= 50 and all("jazz" in " ".join(cat.tags(c.track_id) + cat.genres(c.track_id))
                                        for c in found[:10])
