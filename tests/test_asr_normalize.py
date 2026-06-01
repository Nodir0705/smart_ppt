from pathlib import Path
import json
import pytest
from src.asr_normalize import build_alias_map, load_aliases, normalize


def test_normalize_replaces_spaced_letters():
    amap = {"s t t": "stt"}
    assert normalize("the s t t engine dominates", amap) == "the stt engine dominates"


def test_normalize_is_word_boundary_safe():
    amap = {"em": "match"}  # contrived
    # 'em' inside 'them' must NOT be replaced
    assert normalize("them em", amap) == "them match"


def test_normalize_longest_variant_first():
    amap = {"s t": "x", "s t t": "stt"}
    assert normalize("s t t", amap) == "stt"


def test_normalize_case_insensitive():
    amap = {"s t t": "stt"}
    assert normalize("The S T T Engine", amap).lower() == "the stt engine"


def test_build_alias_map_covers_stt_spaced_letters():
    index = {"slides": [{"slide_number": 4, "elements": [
        {"id": "s4_e2", "type": "chart_category", "text": "STT"}]}]}
    amap = build_alias_map(index)
    assert amap.get("s t t") == "stt"
    # a spelled-letter variant maps to stt too
    assert any(v == "stt" and " " in k for k, v in amap.items())


def test_build_alias_map_ignores_non_acronyms():
    index = {"slides": [{"slide_number": 1, "elements": [
        {"id": "s1_e0", "type": "body", "text": "The Smart Pointer"}]}]}
    amap = build_alias_map(index)
    # no spurious aliases from a normal Title Case phrase
    assert "the" not in amap and "smart" not in amap


def test_load_aliases_absent_returns_empty(tmp_path):
    assert load_aliases(tmp_path / "nope.json") == {}


def test_load_aliases_reads_overrides(tmp_path):
    p = tmp_path / "aliases.json"
    p.write_text(json.dumps({"Speech To Text": "stt"}))
    d = load_aliases(p)
    assert d.get("speech to text") == "stt"
