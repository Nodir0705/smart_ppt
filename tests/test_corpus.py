"""Generalization regression net over the generated deck corpus.

`tools/make_corpus.build_corpus` emits small decks whose pointable texts are
known by construction. Here we parse each and assert no author-intended element
is silently dropped — the failure mode that hides when everything is tuned to
one deck.

Decks that target a *known* parser gap (e.g. grouped shapes) assert the gap
still exists, so when the parser is later fixed this test fails loudly and tells
us to retire the `known_gap` marker — a self-updating regression net.
"""

from __future__ import annotations

import pytest

from src.parser import _clean_alt_text, parse_deck
from tools.make_corpus import build_corpus


@pytest.mark.parametrize("raw,expected", [
    ("Architecture diagram", "Architecture diagram"),  # real alt text kept
    ("Logo", "Logo"),
    ("/home/x/photo.png", ""),                          # file path rejected
    ("C:\\img\\a.PNG", ""),
    ("photo.jpg", ""),                                   # bare filename rejected
    ("Picture 1", ""),                                   # auto-name rejected
    ("image", ""),
    ("  ", ""),
])
def test_clean_alt_text_rejects_junk(raw, expected):
    assert _clean_alt_text(raw) == expected


def _element_texts(index: dict) -> list[str]:
    return [
        (e.get("text") or "").strip()
        for s in index["slides"]
        for e in s["elements"]
    ]


def _survives(needle: str, haystack: list[str]) -> bool:
    n = needle.lower()
    return any(n in h.lower() for h in haystack)


def test_corpus_no_silent_drops(tmp_path):
    decks = build_corpus(tmp_path)
    assert decks, "corpus generator produced no decks"

    for d in decks:
        index = parse_deck(d.path)  # must not raise (covers empty/edge decks)
        texts = _element_texts(index)
        n_elements = len(texts)
        present = [t for t in d.expected_texts if _survives(t, texts)]

        if d.known_gap:
            # The deck targets a documented gap: its texts SHOULD survive but
            # don't yet. If they start surviving, the gap is fixed — fail so we
            # remove the marker rather than silently keep xfailing forever.
            assert len(present) < len(d.expected_texts), (
                f"{d.name}: known gap appears FIXED ({d.known_gap}). "
                f"Retire known_gap in tools/make_corpus.py and assert survival."
            )
            continue  # element-count expectations describe the post-fix deck

        missing = [t for t in d.expected_texts if not _survives(t, texts)]
        assert not missing, f"{d.name}: parser silently dropped {missing}"
        assert n_elements >= d.expected_min_elements, (
            f"{d.name}: expected >= {d.expected_min_elements} elements, got {n_elements}"
        )


def test_corpus_parse_is_deterministic(tmp_path):
    # The generator + parser are deterministic: same build → same parsed texts.
    a = build_corpus(tmp_path / "a")
    b = build_corpus(tmp_path / "b")
    by_name_b = {d.name: d for d in b}
    for da in a:
        ta = _element_texts(parse_deck(da.path))
        tb = _element_texts(parse_deck(by_name_b[da.name].path))
        assert ta == tb, f"{da.name}: non-deterministic parse {ta!r} != {tb!r}"
