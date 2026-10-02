"""Unit tests for `select_line` — the line-level highlight selector (pure).

`select_line` decides WHICH wrapped line of a soft-wrapped element the speaker
is currently on, from the recent spoken tail. It is the whole value of line-
level highlighting, so it's tested here directly (no refine/soffice/model): a
reader walking the paragraph must advance monotonically and land on the last
line; degenerate inputs must fall back (None) to the union bbox.
"""

from __future__ import annotations

from src.live_controller import select_line

LINES = [
    "this is a single long paragraph that must",
    "wrap across several visual lines inside one",
    "narrow text box for the final demo",
]


def test_single_line_returns_none():
    assert select_line("anything here", ["only one line"]) is None


def test_empty_tail_returns_none():
    assert select_line("", LINES) is None
    assert select_line("   ", LINES) is None


def test_zero_overlap_returns_none():
    # nothing the speaker said touches any line → caller uses the union bbox
    assert select_line("completely unrelated words", LINES) is None


def test_picks_the_line_with_the_recent_words():
    # tail sitting on line 1's distinctive words selects line 1
    assert select_line("wrap across several visual lines", LINES) == 1
    assert select_line("narrow text box final demo", LINES) == 2


def test_reading_head_is_monotonic_and_ends_on_last_line():
    # simulate a reader: slide an 8-word window across the whole paragraph and
    # assert the selected line never goes BACKWARD and finishes on the last line.
    words = " ".join(LINES).split()
    win = 8
    picks = []
    for end in range(1, len(words) + 1):
        tail = " ".join(words[max(0, end - win):end])
        idx = select_line(tail, LINES)
        if idx is not None:
            picks.append(idx)
    assert picks, "selector never engaged over a full read"
    assert picks == sorted(picks), f"line selection went backward: {picks}"
    assert picks[-1] == len(LINES) - 1, f"did not end on the last line: {picks}"


def test_stopwords_do_not_dominate():
    # "the" appears in lines 0 and 2; a tail that is mostly stopwords plus one
    # distinctive line-1 word must still resolve to line 1 (df-discounting works)
    assert select_line("that the one inside", LINES) == 1


def test_tie_breaks_to_later_line():
    # identical, equally-distinctive content in two lines; the shared word makes
    # both score equally → reading-forward tie-break picks the later line
    lines = ["alpha unique", "alpha unique"]
    assert select_line("alpha", lines) == 1


def test_deck_punctuation_does_not_hide_words():
    # line_texts keep the author's punctuation; STT output has none. Each line's
    # only distinctive word carries punctuation, so a bare split() never sees it.
    lines = ["the pipeline will Extract,", "the pipeline then renders.", "the pipeline (offline)"]
    assert select_line("so we extract", lines) == 0
    assert select_line("and it renders", lines) == 1
    assert select_line("all offline", lines) == 2
