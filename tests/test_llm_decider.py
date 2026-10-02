"""LLM decider client: prompt building, option parsing, and the JSON-lines subprocess
protocol against a fake server (no model, no llama_cpp needed)."""
import sys
import textwrap

import pytest

from src.deciders.llm import (LETTERS, LLMDecider, LLMDeciderUnavailable, NONE_LETTER,
                              build_prompt, parse_scores, unit_line)

UNITS = [{"uid": "u1", "text": "How it works", "summary": "Title: how it works", "talk": [], "terms": []},
         {"uid": "u2", "text": "python nano.py handout.pdf", "summary": "The single command",
          "talk": ["you just run one script"], "terms": ["nano py", "p d f"]}]


def test_prompt_prefix_depends_only_on_units():
    p1, s1, o1 = build_prompt("then it grades", "first it reads", UNITS)
    p2, s2, _ = build_prompt("something else", "", UNITS)
    assert p1 == p2 and s1 != s2                     # prefix is cacheable per slide
    assert o1 == ["A", "B", NONE_LETTER]
    assert "A. Title: how it works" in p1 and "B. The single command" in p1
    assert "[nano py, p d f]" in p1                  # terms carry ASR-style spellings
    assert '"then it grades"' in s1 and s1.endswith("<|im_start|>assistant\n")


def test_letters_skip_none_letter_and_cap():
    assert NONE_LETTER not in LETTERS and len(set(LETTERS)) == len(LETTERS) == 25
    many = [{"uid": f"u{i}", "summary": f"s{i}"} for i in range(14)]
    assert build_prompt("x", "", many)[2][12:] == ["M", "O", NONE_LETTER]
    with pytest.raises(ValueError):
        build_prompt("x", "", [{"uid": str(i)} for i in range(26)])


def test_unit_line_truncates_text_and_falls_back_to_text():
    line = unit_line("C", {"summary": "", "text": "word " * 100})
    assert line.startswith("C. word") and line.endswith("...") and len(line) < 200
    assert unit_line("A", {"summary": "S", "text": "T"}) == 'A. S (on slide: "T")'


def test_parse_scores():
    out = parse_scores({"ok": True, "scores": [-1.0, -2.0, -0.5]}, UNITS, ["A", "B", "N"])
    assert out == {"u1": -1.0, "u2": -2.0, "none": -0.5}
    with pytest.raises(LLMDeciderUnavailable, match="boom"):
        parse_scores({"ok": False, "error": "boom"}, UNITS, ["A", "B", "N"])
    with pytest.raises(LLMDeciderUnavailable):
        parse_scores({"ok": True, "scores": [0.0]}, UNITS, ["A", "B", "N"])


FAKE = """
import json, sys
print(json.dumps({"ok": True, "model": "fake"}), flush=True)
for line in sys.stdin:
    r = json.loads(line)
    if "die" in r["suffix"]:
        sys.exit(3)
    # option i: -1 - 0.1*i, except the letter named as "pick X" in the clause gets 0
    s = [0.0 if f"pick {o}" in r["suffix"] else -1.0 - 0.1 * i for i, o in enumerate(r["options"])]
    print(json.dumps({"ok": True, "scores": s, "ms": 1.5}), flush=True)
"""


@pytest.fixture
def fake(tmp_path):
    server, model = tmp_path / "fake_server.py", tmp_path / "m.gguf"
    server.write_text(FAKE)
    model.write_text("")
    return dict(python=sys.executable, server=server, model=model)


def test_protocol_with_fake_server(fake):
    with LLMDecider(fake["model"], python=fake["python"], server=fake["server"], calibrate=False) as d:
        assert d.info["model"] == "fake"
        assert d.decide("pick B", "", UNITS) == {"u1": -1.0, "u2": 0.0, "none": -1.2}
        assert d.last_ms == 1.5
        assert d.decide("x", "", []) == {"none": 0.0}


def test_calibration_subtracts_empty_clause_prior(fake):
    with LLMDecider(fake["model"], python=fake["python"], server=fake["server"]) as d:
        d.warm(UNITS)
        out = d.decide("pick N", "", UNITS)
        assert out == pytest.approx({"u1": 0.0, "u2": 0.0, "none": 1.2})
        assert max(out, key=out.get) == "none"


def test_server_death_and_failed_start(fake, tmp_path):
    with LLMDecider(fake["model"], python=fake["python"], server=fake["server"], calibrate=False) as d:
        with pytest.raises(LLMDeciderUnavailable, match="exited"):
            d.decide("die", "", UNITS)
    bad = tmp_path / "bad.py"
    bad.write_text('import json; print(json.dumps({"ok": False, "error": "no llama_cpp"}))')
    with pytest.raises(LLMDeciderUnavailable, match="no llama_cpp"):
        LLMDecider(fake["model"], python=fake["python"], server=bad)


def test_missing_venv_or_model_is_a_clear_error(fake, tmp_path):
    with pytest.raises(LLMDeciderUnavailable, match="venv python not found"):
        LLMDecider(fake["model"], python=tmp_path / "nope/python", server=fake["server"])
    with pytest.raises(LLMDeciderUnavailable, match="model not found"):
        LLMDecider(tmp_path / "missing.gguf", python=fake["python"], server=fake["server"])
