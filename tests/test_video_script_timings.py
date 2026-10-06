from tools.video.script import norm, sentences, words
from tools.video.timings import align, match


def _asr(*spec):
    """("word", start, end), ... -> Whistle-style word dicts."""
    return [{"word": w, "start": s, "end": e} for w, s, e in spec]


def _starts(res):
    return [r["start"] for r in res]


# ---------- script ----------

def test_sentences_skips_comments_and_blanks(tmp_path):
    p = tmp_path / "slide_001.md"
    p.write_text("# a comment\n\n  First sentence here.  \n   \n#another\nSecond one.\n")
    assert sentences(p) == ["First sentence here.", "Second one."]


def test_words_whitespace_only():
    s = "Stage one turns pixels into text, so the printed stuff — all in one go."
    assert words(s) == ["Stage", "one", "turns", "pixels", "into", "text,", "so", "the",
                        "printed", "stuff", "—", "all", "in", "one", "go."]


def test_norm():
    assert norm("Text,") == "text"
    assert norm("go.") == "go"
    assert norm("8") == "eight"
    assert norm("—") == ""
    assert norm("...") == ""


# ---------- align ----------

def test_exact_match():
    sw = ["Hi", "everyone,", "thanks"]
    res = align(sw, _asr(("Hi", 0.08, 0.4), ("everyone,", 0.4, 0.88), ("thanks", 1.28, 1.6)))
    assert res == [{"word": "Hi", "start": 0.08, "end": 0.4},
                   {"word": "everyone,", "start": 0.4, "end": 0.88},
                   {"word": "thanks", "start": 1.28, "end": 1.6}]


def test_substituted_word_fills_gap_between_neighbours():
    # contiguous stamps (as Whistle emits): the substitute occupies exactly k's slot
    res = align(["Stage", "one", "turns"],
                _asr(("Stage", 0.0, 0.4), ("won", 0.4, 0.6), ("turns", 0.6, 0.9)))
    assert res[1] == {"word": "one", "start": 0.4, "end": 0.6}


def test_missing_run_shares_gap_evenly():
    res = align(["a", "bee", "cee", "dee"],
                _asr(("a", 0.0, 0.2), ("dee", 0.8, 1.0)))
    assert [(r["start"], r["end"]) for r in res[1:3]] == [(0.2, 0.5), (0.5, 0.8)]
    assert res[3]["start"] == 0.8


def test_dropped_word_with_contiguous_neighbours():
    res = align(["one", "two", "three"], _asr(("one", 0.0, 0.3), ("three", 0.3, 0.6)))
    assert res[1] == {"word": "two", "start": 0.3, "end": 0.3}


def test_extra_asr_word_ignored():
    res = align(["the", "cat"], _asr(("the", 0.0, 0.2), ("um", 0.2, 0.5), ("cat", 0.5, 0.8)))
    assert [(r["start"], r["end"]) for r in res] == [(0.0, 0.2), (0.5, 0.8)]


def test_leading_and_trailing_clamp():
    res = align(["so", "the", "cat", "sat", "down"],
                _asr(("the", 0.5, 0.7), ("cat", 0.7, 1.0)))
    assert res[0] == {"word": "so", "start": 0.5, "end": 0.5}
    assert res[3] == {"word": "sat", "start": 1.0, "end": 1.0}
    assert res[4] == {"word": "down", "start": 1.0, "end": 1.0}


def test_punctuation_takes_previous_word_time():
    res = align(["stuff", "—", "all"], _asr(("stuff", 0.0, 0.4), ("all", 0.5, 0.7)))
    assert res[1] == {"word": "—", "start": 0.0, "end": 0.4}


def test_leading_punctuation_clamps_to_first():
    res = align(["—", "hello"], _asr(("hello", 0.3, 0.6)))
    assert res[0] == {"word": "—", "start": 0.3, "end": 0.3}


def test_digits_and_multi_token_words():
    # ASR "8" == script "eight"; ASR "100%" splits its span over three script words
    res = align(["eight", "one", "hundred", "percent"],
                _asr(("8", 0.0, 0.3), ("100%", 0.3, 0.9)))
    assert res[0]["start"] == 0.0
    assert [round(r["start"], 3) for r in res[1:]] == [0.3, 0.5, 0.7]
    assert round(res[3]["end"], 3) == 0.9


def test_no_aligned_words_is_all_zero():
    for asr in ([], _asr(("xyz", 0.1, 0.5))):
        res = align(["hello", "world"], asr)
        assert [(r["start"], r["end"]) for r in res] == [(0.0, 0.0), (0.0, 0.0)]


def test_starts_non_decreasing_with_overlapping_asr():
    res = align(["a", "bee", "cee", "dee"],
                _asr(("a", 0.0, 0.6), ("cee", 0.3, 0.5), ("dee", 0.2, 0.4)))
    assert len(res) == 4
    assert _starts(res) == sorted(_starts(res))


def test_length_and_monotone_on_messy_input():
    sw = words("Stage one turns pixels into text, so the printed stuff — all in one go.")
    asr = _asr(("stage", 0.0, 0.3), ("1", 0.3, 0.5), ("turns", 0.5, 0.8), ("pixel", 0.8, 1.1),
               ("into", 1.1, 1.3), ("text", 1.3, 1.6), ("uh", 1.6, 1.8), ("the", 1.8, 1.9),
               ("printed", 1.9, 2.3), ("stuff", 2.3, 2.6), ("all", 2.8, 3.0), ("in", 3.0, 3.1),
               ("one", 3.1, 3.3), ("go", 3.3, 3.6))
    res = align(sw, asr)
    assert [r["word"] for r in res] == sw
    assert _starts(res) == sorted(_starts(res))
    m = match(sw, asr)
    assert m[3] is None and m[6] is None                  # "pixels" (!= "pixel"), "so"
    assert m[1] == (0.3, 0.5)                             # "one" == norm("1")
