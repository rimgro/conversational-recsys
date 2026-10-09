from recsys.dialog import RuleSummarizer
from recsys.fusion import rrf
from recsys.llm import extract_json
from recsys.schemas import Candidate, Message, Request
from recsys.text import normalize_tag, tag_matches


def test_normalize_tag():
    assert normalize_tag("Hip-Hop") == "hip hop"
    assert normalize_tag("Lo-Fi") == "lo fi"
    assert normalize_tag("R&B") == "rnb"


def test_tag_matches_word_boundary():
    assert tag_matches("rap", ["gangsta rap"])
    assert not tag_matches("rap", ["trap"])


def test_rrf_merges_and_weights():
    lists = {
        "a": [Candidate("x", "a", 1.0, 1), Candidate("y", "a", 0.5, 2)],
        "b": [Candidate("y", "b", 1.0, 1)],
    }
    fused = rrf(lists, k=60)
    assert fused[0].track_id == "y" and set(fused[0].ranks) == {"a", "b"}
    fused = rrf(lists, k=60, weights={"b": 0.0})
    assert fused[0].track_id == "x"


def test_extract_json():
    assert extract_json('blah ```json\n{"a": {"b": 1}}\n``` tail') == {"a": {"b": 1}}
    assert extract_json("no json") is None


def test_rule_summarizer_negation_and_override(data):
    s = RuleSummarizer(data.catalog)
    req = Request(dialog=[Message("user", "I want calm folk, but no pop."),
                          Message("assistant", "ok"),
                          Message("user", "Actually pop is fine, just no metal")])
    out = s.summarize(req)
    assert "folk" in out.include_tags and "pop" in out.include_tags
    assert "metal" in out.exclude_tags and "pop" not in out.exclude_tags
    assert out.energy == "low"


def test_rule_summarizer_artist(data):
    artist = data.catalog.df["artist"].iloc[0]
    out = RuleSummarizer(data.catalog).summarize(Request(dialog=[Message("user", f"something like {artist}")]))
    assert artist in out.seed_artists


def test_energy_change_drops_opposite_tags(data):
    req = Request(dialog=[Message("user", "something calm and acoustic"),
                          Message("user", "now more energetic")])
    out = RuleSummarizer(data.catalog).summarize(req)
    assert out.energy == "high" and "calm" not in out.include_tags and "acoustic" in out.include_tags


def test_config_warns_on_typo():
    import warnings

    from recsys.config import deep_update
    base = {"ranker": {"type": "stub", "weights": {}}}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = deep_update(base, {"ranker": {"tpye": "heuristic", "weights": {"rrf_score": 1.0}}})
    assert [str(x.message) for x in w] == ["config: ключа 'ranker.tpye' нет в базовом конфиге (опечатка?)"]
    assert out["ranker"]["type"] == "stub" and out["ranker"]["weights"] == {"rrf_score": 1.0}


def test_rule_summarizer_english(data):
    s = RuleSummarizer(data.catalog)

    def run(q):
        return s.summarize(Request(dialog=[Message("user", q)]))

    out = run("calm british indie rock from the 80s, but no female vocalists")
    assert {"calm", "british", "indie rock", "80s"} <= set(out.include_tags)
    assert out.exclude_tags == ["female vocalists"] and out.countries == ["GB"] and out.energy == "low"
    # «no vocals» = хочу инструментал, а не исключение
    out = run("calm music without vocals")
    assert "instrumental" in out.include_tags and not out.exclude_tags
    # отрицание не меняет энергию и не вычищает сказанное раньше
    out = run("aggressive music but not dance")
    assert out.include_tags == ["aggressive"] and out.exclude_tags == ["dance"]
    out = run("american rap from 2003, just no pop")
    assert out.years == [2003] and "00s" in out.include_tags and out.exclude_tags == ["pop"]
    assert "US" in out.countries
    out = run("songs sung in french")
    assert out.languages == ["fr"] and not out.countries


def test_similar_to_reference_and_novelty(data):
    s = RuleSummarizer(data.catalog)
    row = data.catalog.df.iloc[3]
    q = f"tracks like {row['title'].lower()} by {row['artist'].lower()} but from other artists, more calm"
    out = s.summarize(Request(dialog=[Message("user", q)]))
    assert out.seed_track_ids == [row["track_id"]] and out.exclude_artists == [row["artist"]]
    assert row["artist"] not in out.seed_artists and out.new_tracks and not out.new_artists
    assert "calm" in out.include_tags
    out = s.summarize(Request(dialog=[Message("user", "new artists please, something i haven't heard")]))
    assert out.new_tracks and out.new_artists
    out = s.summarize(Request(dialog=[Message("user", "smooth 80s new wave track")]))
    assert not out.new_tracks


def test_llm_summary_merge():
    from recsys.dialog import LLMSummarizer
    from recsys.schemas import DialogSummary
    out = LLMSummarizer._merge(DialogSummary(include_tags=["rock"]), {
        "include_tags": ["Hip-Hop"], "countries": ["us"], "years": [2003, "x"], "track_title": "Suga Suga",
        "query": "american hip hop", "summary": "s"})
    assert out.include_tags == ["hip hop"] and out.countries == ["US"] and out.years == [2003]
    assert out.query == "american hip hop Suga Suga" and out.source == "llm+rule"


def test_profile_numbers_are_not_tastes(data):
    req = Request(dialog=[Message("user", "jazz")],
                  user_info="Listening since 2010; 300 plays before this period. Favourite genres: folk, metal. "
                            "Song languages: en 99%.")
    out = RuleSummarizer(data.catalog).summarize(req)
    assert out.user_tags == ["folk", "metal"]


def test_gemma_service_llm(tmp_path, cfg, data):
    """Сервис DataSphere подменяем модулем во временной папке; без модуля описание падает на шаблон."""
    from recsys.config import deep_update
    from recsys.explain import LLMExplainer
    from recsys.llm import GemmaServiceLLM
    from recsys.pipeline import Pipeline
    (tmp_path / "fake_gemma.py").write_text(
        "def measure_request(prompt):\n    return {'response': ' Enjoy! ' + prompt[-1], 'latency': 0.1}\n",
        encoding="utf8")
    llm = GemmaServiceLLM(str(tmp_path), module="fake_gemma")
    assert llm.generate([{"role": "system", "content": "a"}, {"role": "user", "content": "b"}]) == "Enjoy! b"
    assert GemmaServiceLLM._text(("text", 0.2)) == "text" and GemmaServiceLLM._text(None) == ""
    c = deep_update(cfg, {"explainer": {"type": "llm"}})
    pipe = Pipeline.from_config(c, data.catalog, llm=llm)
    assert isinstance(pipe.explainer, LLMExplainer) and pipe.run(data.requests[0]).text.startswith("Enjoy!")
    missing = GemmaServiceLLM(str(tmp_path), module="no_such_service")
    assert Pipeline.from_config(c, data.catalog, llm=missing).run(data.requests[0]).text.startswith("Hi!")


def test_query_embedder_device_and_cache(tmp_path, monkeypatch):
    from recsys.retrieval import query_embedder as qe
    assert qe.pick_device(None, None, cuda_available=True) == ("cuda", "float16")
    assert qe.pick_device(None, None, cuda_available=False) == ("cpu", None)       # CPU — float32
    assert qe.pick_device("cpu", None, cuda_available=True) == ("cpu", None)       # явное из конфига важнее
    assert qe.pick_device("cuda", "float32", cuda_available=True) == ("cuda", "float32")
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(qe, "DATASPHERE_PROJECT", str(tmp_path))                  # «как в DataSphere»
    assert qe.setup_cache() == str(tmp_path / "hf_cache")
    monkeypatch.setenv("HF_HOME", "/somewhere")                                    # заданный явно не трогаем
    assert qe.setup_cache() == "/somewhere"


def test_make_embed_stops_with_instructions(capsys, monkeypatch):
    import make_embed
    monkeypatch.setattr(make_embed, "check_packages", lambda: False)
    assert make_embed.main(["--no-server"]) == 1
