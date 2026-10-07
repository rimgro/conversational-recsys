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
