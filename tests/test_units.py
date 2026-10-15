from recsys.dialog import RuleSummarizer
from recsys.llm import extract_json
from recsys.schemas import Message, Request
from recsys.text import normalize_tag, tag_matches


def test_normalize_tag():
    assert normalize_tag("Hip-Hop") == "hip hop"
    assert normalize_tag("Lo-Fi") == "lo fi"
    assert normalize_tag("R&B") == "rnb"


def test_tag_matches_word_boundary():
    assert tag_matches("rap", ["gangsta rap"])
    assert not tag_matches("rap", ["trap"])


def test_extract_json():
    assert extract_json('blah ```json\n{"a": {"b": 1}}\n``` tail') == {"a": {"b": 1}}
    assert extract_json("no json") is None


def test_rule_summarizer_negation_and_override(catalog):
    s = RuleSummarizer(catalog)
    req = Request(dialog=[Message("user", "I want calm folk, but no pop."),
                          Message("assistant", "ok"),
                          Message("user", "Actually pop is fine, just no metal")])
    out = s.summarize(req)
    assert "folk" in out.include_tags and "pop" in out.include_tags
    assert "metal" in out.exclude_tags and "pop" not in out.exclude_tags
    assert out.energy == "low"


def test_rule_summarizer_artist(catalog):
    artist = catalog.df["artist"].iloc[0]
    out = RuleSummarizer(catalog).summarize(Request(dialog=[Message("user", f"something like {artist}")]))
    assert artist in out.seed_artists


def test_energy_change_drops_opposite_tags(catalog):
    req = Request(dialog=[Message("user", "something calm and acoustic"),
                          Message("user", "now more energetic")])
    out = RuleSummarizer(catalog).summarize(req)
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


def test_rule_summarizer_english(catalog):
    s = RuleSummarizer(catalog)

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


def test_similar_to_reference_and_novelty(catalog):
    s = RuleSummarizer(catalog)
    row = catalog.df.iloc[3]
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


def test_profile_numbers_are_not_tastes(catalog):
    req = Request(dialog=[Message("user", "jazz")],
                  user_info="Listening since 2010; 300 plays before this period. Favourite genres: folk, metal. "
                            "Song languages: en 99%.")
    out = RuleSummarizer(catalog).summarize(req)
    assert out.user_tags == ["folk", "metal"]


def test_gemma_service_llm(tmp_path, cfg, res, requests):
    """Сервис DataSphere подменяем модулем во временной папке; без модуля описание падает на шаблон."""
    from recsys.config import deep_update
    from recsys.explain import LLMExplainer
    from recsys.llm import GemmaServiceLLM
    from recsys.pipeline import Pipeline
    from recsys.resources import Resources
    (tmp_path / "fake_gemma.py").write_text(
        "def measure_request(prompt):\n    return {'response': ' Enjoy! ' + prompt[-1], 'latency': 0.1}\n",
        encoding="utf8")
    llm = GemmaServiceLLM(str(tmp_path), module="fake_gemma")
    assert llm.generate([{"role": "system", "content": "a"}, {"role": "user", "content": "b"}]) == "Enjoy! b"
    assert GemmaServiceLLM._text(("text", 0.2)) == "text" and GemmaServiceLLM._text(None) == ""
    c = deep_update(cfg, {"explainer": {"type": "llm"}})
    pipe = Pipeline(c, Resources(c, dataset=res.dataset, llm=llm, verbose=False))
    assert isinstance(pipe.explainer, LLMExplainer) and pipe.run(requests[0]).text.startswith("Enjoy!")
    missing = GemmaServiceLLM(str(tmp_path), module="no_such_service")
    pipe = Pipeline(c, Resources(c, dataset=res.dataset, llm=missing, verbose=False))
    assert pipe.run(requests[0]).text.startswith("Hi!")


def test_query_embedder_device_and_cache(tmp_path, monkeypatch):
    from recsys.candgen import query_embedder as qe
    assert qe.pick_device(None, None, cuda_available=True) == ("cuda", "float16")
    assert qe.pick_device(None, None, cuda_available=False) == ("cpu", None)       # CPU — float32
    assert qe.pick_device("cpu", None, cuda_available=True) == ("cpu", None)       # явное из конфига важнее
    assert qe.pick_device("cuda", "float32", cuda_available=True) == ("cuda", "float32")
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(qe, "DATASPHERE_PROJECT", str(tmp_path))                  # «как в DataSphere»
    assert qe.setup_cache() == str(tmp_path / "hf_cache")
    monkeypatch.setenv("HF_HOME", "/somewhere")                                    # заданный явно не трогаем
    assert qe.setup_cache() == "/somewhere"


def test_parse_constraints():
    from recsys.text import parse_constraints
    assert parse_constraints("recommend 2018 progressive rock, high energy but minor key") == \
        {"year_min": 2017, "year_max": 2019, "energy_min": 0.5, "mode": 0}
    assert parse_constraints("late 90s american pop punk") == {"year_min": 1995, "year_max": 2007}
    assert parse_constraints("energetic alt rock around 120 bpm") == {"bpm": 120.0, "energy_min": 0.5}
    assert parse_constraints("relaxing jazz without vocals") == {"instrumental": True}
    assert parse_constraints("70s prog rock with male vocals")["vocals"] == "male"
    # отрицания ограничений не дают: «not major» — не обязательно минор, «not too energetic» — не «low energy»
    assert parse_constraints("danceable but not major key") == {}
    assert parse_constraints("something upbeat but not too energetic") == {}
    assert parse_constraints("music for a road trip") == {}


def test_rule_summarizer_fills_constraints(catalog):
    s = RuleSummarizer(catalog).summarize(Request(dialog=[Message("user", "80s synthpop in a minor key"),
                                                               Message("user", "actually make it major")]))
    assert s.constraints["mode"] == 1 and s.constraints["year_min"] == 1980  # позже сказанное перекрывает


def _negation_catalog():
    """Маленький каталог: артист-тег (nirvana), жанр-тег (rock) и названия с «no / without / don't»."""
    import pandas as pd
    from recsys.data.catalog import Catalog
    rows = [("Nirvana", "Come as You Are", ["grunge", "rock", "nirvana"]),
            ("Nirvana", "Lithium", ["grunge", "nirvana"]),
            ("Nirvana", "Dumb", ["grunge", "nirvana"]),
            ("Foo Fighters", "Everlong", ["rock", "nirvana"]),
            ("Arctic Monkeys", "No Buses", ["indie rock", "rock"]),
            ("Belinda Carlisle", "World Without You", ["pop", "80s"]),
            ("Hollywood Undead", "Hear Me Now", ["rap rock", "rock"]),
            ("Billy Idol", "Eyes Without a Face", ["80s", "rock"]),
            ("Metallica", "One", ["thrash", "metal", "rock"]),
            ("Metallica", "Fuel", ["thrash", "metal"]),
            ("Dio", "Holy Diver", ["metal", "80s"]),
            ("Rock", "Song", ["pop"]),
            ("Daft Punk", "Around the World", ["electronic", "dance", "pop"]),
            ("Massive Attack", "Teardrop", ["trip hop", "electronic", "sad"]),
            ("Adele", "Hello", ["pop", "sad", "soul"]),
            ("Eminem", "Lose Yourself", ["rap", "hip hop"]),
            ("Kanye West", "Stronger", ["rap", "hip hop"])]
    df = pd.DataFrame({"track_id": [f"t{i}" for i in range(len(rows))], "artist": [r[0] for r in rows],
                       "title": [r[1] for r in rows], "tags": [r[2] for r in rows]})
    return Catalog(df)


def test_negation_rules():
    """Исключается только распознанное отрицание; названия, текст песни и «not too X» — не исключения;
    search_text — реплика без исключённого."""
    s = RuleSummarizer(_negation_catalog(), min_tag_count=1)

    def run(q):
        return s.summarize(Request(dialog=[Message("user", q)]))

    out = run("no nirvana")                       # и артист, и тег: тег почти только его -> артист
    assert out.exclude_artists == ["Nirvana"] and not out.exclude_tags and out.search_text == ""
    out = run("no rock!")                         # артист «Rock» есть, но тег rock не его -> жанр
    assert out.exclude_tags == ["rock"] and not out.exclude_artists and out.search_text == ""
    out = run("80s metal but no thrash")
    assert out.exclude_tags == ["thrash"] and out.search_text == "80s metal"
    out = run("i don't want rap, play some rock")
    assert out.exclude_tags == ["rap"] and "rock" in out.include_tags
    out = run("sad electronic but not too dance")   # мягко: не просим и не исключаем
    assert not out.exclude_tags and "dance" not in out.include_tags and "dance" not in out.search_text.split()
    for q in ("world without you belinda carlisle", "no buses arctic monkeys", "hear me no hollywood undead",
              "play eyes without a face", "you are not alone in this sad life",
              "lyrics about not being sad and no rock anymore"):
        out = run(q)                              # название, предложение, текст песни
        assert not out.exclude_tags and not out.exclude_artists, q
        assert out.search_text == " ".join(__import__("recsys.text", fromlist=["tokenize"]).tokenize(q)), q
    out = run("songs like lose yourself by eminem but from other artists, more rap")
    assert out.exclude_artists == ["Eminem"] and "eminem" not in out.search_text and "rap" in out.search_text


def test_fuzzy_artist_after_negation():
    """summarizer.fuzzy_artists: опечатка в имени после отрицания; обычные слова и выключенная настройка — нет."""
    from recsys.data.history import build_profile
    from recsys.schemas import HistoryItem
    catalog = _negation_catalog()
    on, off = RuleSummarizer(catalog, min_tag_count=1, fuzzy_artists=True), RuleSummarizer(catalog, min_tag_count=1)

    def run(s, q, profile=None):
        return s.summarize(Request(dialog=[Message("user", q)]), profile)

    out = run(on, "rock, but no nirnava")
    assert out.exclude_artists == ["Nirvana"] and out.search_text == "rock"
    assert run(on, "no metalica").exclude_artists == ["Metallica"]
    assert not run(off, "rock, but no nirnava").exclude_artists          # по умолчанию выключено
    assert not run(on, "no matter what").exclude_artists                  # нет похожего имени
    assert not run(on, "no adel").exclude_artists                         # 4 буквы — только артисты истории
    profile = build_profile([HistoryItem("t14", 1, 1)], catalog)          # Adele в истории
    assert run(on, "no adel", profile).exclude_artists == ["Adele"]


def test_entity_resolver_votes_and_summarizer_hook():
    """Артист — если один источник уверен или два согласны; разбор спрашивает только нераспознанное после «no»."""
    from types import SimpleNamespace
    from recsys.entity import EntityResolver
    r = EntityResolver.__new__(EntityResolver)
    r.single, r.pair = 14, 8
    assert r.decide({"bm25": ("A", 15), "lazy": ("B", 3)}) == "A"
    assert r.decide({"bm25": ("A", 9), "lazy": ("A", 8), "v2": ("C", 2)}) == "A"
    assert r.decide({"bm25": ("A", 9), "lazy": ("B", 9)}) is None
    assert r.decide({"v2": ("Sade", 12)}) is None

    asked = []
    fake = SimpleNamespace(artist=lambda p: asked.append(p) or ("Red Hot Chili Peppers" if p == "chilli pepers" else None))
    s = RuleSummarizer(_negation_catalog(), min_tag_count=1, resolver=fake)
    out = s.summarize(Request(dialog=[Message("user", "rock, but no chilli pepers please")]))
    assert out.exclude_artists == ["Red Hot Chili Peppers"] and out.search_text == "rock please"
    s.summarize(Request(dialog=[Message("user", "no nirvana")]))           # узнано правилами — поиск не нужен
    s.summarize(Request(dialog=[Message("user", "lyrics about no regrets")]))   # не отрицание
    assert asked == ["chilli pepers"]
def test_negation_word_typos():
    """«noting like nirvana», «witout» — опечатки в слове отрицания; обычные слова — нет."""
    s = RuleSummarizer(_negation_catalog(), min_tag_count=1)

    def run(q):
        return s.summarize(Request(dialog=[Message("user", q)]))

    assert run("noting like nirvana").exclude_artists == ["Nirvana"]
    assert run("rock witout nirvana").exclude_artists == ["Nirvana"]
    assert run("rock but nothng by metallica").exclude_artists == ["Metallica"]
    out = run("something like nirvana")                   # «something» — не опечатка «nothing»
    assert out.seed_artists == ["Nirvana"] and not out.exclude_artists
