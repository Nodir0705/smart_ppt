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

        if d.expected_types:
            from collections import Counter
            got = Counter(e["type"] for s in index["slides"] for e in s["elements"])
            need = Counter(d.expected_types)
            for typ, count in need.items():
                assert got[typ] >= count, (
                    f"{d.name}: expected >= {count} '{typ}' elements, got {got[typ]}"
                )


def test_rotated_shape_bbox_is_aabb(tmp_path):
    # a 6x1in box rotated 90deg -> AABB should swap to ~1x6 (tall, narrow)
    decks = {d.name: d for d in build_corpus(tmp_path)}
    index = parse_deck(decks["rotated_shape"].path)
    el = next(e for s in index["slides"] for e in s["elements"]
              if (e.get("text") or "").startswith("Rotated"))
    b = el["bbox"]
    assert b["h"] > b["w"], f"rotation not applied: w={b['w']:.2f} h={b['h']:.2f}"
    assert abs(b["w"] - 1.0) < 0.05 and abs(b["h"] - 6.0) < 0.05, b


def test_parse_deck_isolates_a_failing_slide(tmp_path, monkeypatch, caplog):
    # A malformed slide must warn + yield an empty slide, not abort the deck.
    import src.parser as P
    deck = {d.name: d for d in build_corpus(tmp_path)}["title_and_bullets"]
    real = P._parse_slide

    def boom(slide, i, w, h):
        if i == 1:
            raise ValueError("synthetic parse failure")
        return real(slide, i, w, h)

    monkeypatch.setattr(P, "_parse_slide", boom)
    with caplog.at_level("WARNING"):
        index = P.parse_deck(deck.path)  # must NOT raise

    assert index["slides"], "deck aborted instead of isolating the bad slide"
    assert index["slides"][0]["slide_number"] == 1
    assert index["slides"][0]["elements"] == []  # failed slide → empty, numbered
    assert any("failed to parse" in r.message for r in caplog.records)


def test_unsupported_chart_is_needs_vision_not_fabricated(tmp_path):
    decks = {d.name: d for d in build_corpus(tmp_path)}
    index = parse_deck(decks["radar_chart"].path)
    els = [e for s in index["slides"] for e in s["elements"]]
    charts = [e for e in els if e["type"] == "chart"]
    cats = [e for e in els if e["type"] == "chart_category"]
    assert len(charts) == 1 and charts[0]["needs_vision"] is True
    assert cats == [], f"radar fabricated {len(cats)} category bars"


def test_grouped_chart_has_distinct_subcolumns(tmp_path):
    decks = {d.name: d for d in build_corpus(tmp_path)}
    index = parse_deck(decks["grouped_bar"].path)
    els = [e for s in index["slides"] for e in s["elements"]]
    chart = next(e for e in els if e["type"] == "chart")
    cats = [e for e in els if e["type"] == "chart_category"]

    assert chart["multi_series"] is True
    assert len(cats) == 6, "expected 3 categories x 2 series"
    # distinct, non-overlapping sub-columns in category-major x order
    xs = [c["bbox"]["x"] for c in cats]
    assert xs == sorted(xs) and len(set(xs)) == 6
    for a, b in zip(cats, cats[1:]):
        assert a["bbox"]["x"] + a["bbox"]["w"] <= b["bbox"]["x"] + 1e-6, "sub-columns overlap"
    # value-proportional height: Revenue 30 (tallest) > Cost 5 (shortest)
    h = {(c["text"]): c["bbox"]["h"] for c in cats}
    assert h["Q3 — Revenue"] > h["Q1 — Cost"]
    # series name reaches the triggers (disambiguation handle)
    rev = next(c for c in cats if c["series"] == "Revenue" and "Q2" in c["text"])
    assert any("revenue" in p for p in rev["trigger_phrases"])


def test_stacked_chart_segments_stack_within_category(tmp_path):
    decks = {d.name: d for d in build_corpus(tmp_path)}
    index = parse_deck(decks["stacked_bar"].path)
    els = [e for s in index["slides"] for e in s["elements"]]
    assert next(e for e in els if e["type"] == "chart")["multi_series"] is True
    cats = [e for e in els if e["type"] == "chart_category"]
    assert len(cats) == 6

    # within a category the two segments share x and stack contiguously (the
    # upper series' bottom == the lower series' top), no overlap
    q1 = [c for c in cats if c["text"].startswith("Q1")]
    base, bonus = q1[0]["bbox"], q1[1]["bbox"]   # category-major: Base then Bonus
    assert abs(base["x"] - bonus["x"]) < 1e-6
    assert base["y"] > bonus["y"]                 # Base lower on screen than Bonus
    assert abs((bonus["y"] + bonus["h"]) - base["y"]) < 1e-6, "segments not contiguous"
    # taller stack for the larger category total (Q3 total 42 > Q1 total 15)
    q3_base = next(c for c in cats if c["text"] == "Q3 — Base")
    assert q3_base["bbox"]["h"] > base["h"]


def test_multiseries_scoped_phrases_are_speakable(tmp_path):
    # scope_index builds chart_category phrases from the element label; the
    # multi-series label is "Cat — Series", which must not leak the em-dash (STT
    # never says it) or produce "... bar chart chart".
    from src.group_scope import scope_index
    decks = {d.name: d for d in build_corpus(tmp_path)}
    for name in ("grouped_bar", "stacked_bar"):
        index = parse_deck(decks[name].path)
        scope_index(index)
        phrases = [p for s in index["slides"] for e in s["elements"]
                   if e["type"] == "chart_category" for p in e["trigger_phrases"]]
        assert phrases, f"{name}: no scoped phrases"
        assert not [p for p in phrases if "—" in p], f"{name}: em-dash leaked into a phrase"
        assert not [p for p in phrases if "chart chart" in p], f"{name}: double 'chart'"
        # the series name still reaches a phrase for disambiguation
        assert any("revenue" in p or "base" in p for p in phrases)


def test_pie_has_distinct_value_proportional_wedges(tmp_path):
    decks = {d.name: d for d in build_corpus(tmp_path)}
    index = parse_deck(decks["pie_chart"].path)
    els = [e for s in index["slides"] for e in s["elements"]]
    chart = next(e for e in els if e["type"] == "chart")
    cats = [e for e in els if e["type"] == "chart_category"]
    # pie has sound geometry -> computed wedges, not a needs_vision punt
    assert chart["needs_vision"] is False
    assert len(cats) == 4
    # the four wedge AABBs are distinct (each slice is a different region)
    boxes = {(round(c["bbox"]["x"], 3), round(c["bbox"]["y"], 3),
              round(c["bbox"]["w"], 3), round(c["bbox"]["h"], 3)) for c in cats}
    assert len(boxes) == 4, "pie wedges collapsed to identical boxes"


def test_nested_groups_map_into_slide_space(tmp_path):
    decks = {d.name: d for d in build_corpus(tmp_path)}
    index = parse_deck(decks["nested_groups"].path)
    els = {(e.get("text") or ""): e for s in index["slides"] for e in s["elements"]}
    # depth-2 child: identity child-extents -> stays at its authored slide coords
    deep = els["Deep node"]["bbox"]
    assert abs(deep["x"] - 2.0) < 0.05 and abs(deep["y"] - 2.0) < 0.05, deep
    mid = els["Mid node"]["bbox"]
    assert abs(mid["x"] - 5.0) < 0.05 and abs(mid["y"] - 3.0) < 0.05, mid


def test_duplicate_text_not_deduped(tmp_path):
    decks = {d.name: d for d in build_corpus(tmp_path)}
    index = parse_deck(decks["duplicate_text"].path)
    live = [e for s in index["slides"] for e in s["elements"]
            if (e.get("text") or "") == "Live"]
    assert len(live) == 2, "identical-text boxes were collapsed"
    xs = sorted(round(e["bbox"]["x"], 1) for e in live)
    assert xs == [1.0, 6.0], xs


@pytest.mark.slow
@pytest.mark.skipif(__import__("shutil").which("soffice") is None,
                    reason="soffice required to render the PDF text layer")
def test_wrapping_paragraph_refines_to_stacked_line_bboxes(tmp_path):
    """The refine step must split ONE soft-wrapping paragraph into multiple
    per-line boxes that stack top-to-bottom without overlap and union back to
    the element box — the contract line-level highlighting consumes.

    Structural asserts only (count >= 2, monotonic y, left-aligned, contiguous
    union) so it survives soffice font/version differences in exact wrapping.
    """
    from src.refine_text import refine_index
    from src.render import deck_pdf

    decks = {d.name: d for d in build_corpus(tmp_path)}
    deck = decks["wrapping_paragraph"]
    index = parse_deck(deck.path)
    # one element pre-refine (no paragraph breaks to split on)
    els = [e for s in index["slides"] for e in s["elements"]]
    assert len(els) == 1, f"wrap paragraph split into {len(els)} elements pre-refine"

    pdf = deck_pdf(deck.path, tmp_path)
    refine_index(index, pdf)
    el = [e for s in index["slides"] for e in s["elements"]][0]
    lbs = el["line_bboxes"]

    assert el["bbox_source"] == "pdf_text"
    assert len(lbs) >= 2, f"expected a multi-line wrap, got {len(lbs)} line(s)"
    # stacked top-to-bottom, no vertical overlap between consecutive lines
    ys = [b["y"] for b in lbs]
    assert ys == sorted(ys), "line_bboxes not ordered top-to-bottom"
    for a, b in zip(lbs, lbs[1:]):
        assert a["y"] + a["h"] <= b["y"] + 1e-6, "consecutive lines overlap vertically"
    # left-aligned: all lines start within a small x band
    xs = [b["x"] for b in lbs]
    assert max(xs) - min(xs) < 0.1, f"lines not left-aligned: {xs}"
    # element bbox is the union of its line boxes
    ux = min(xs)
    uy = min(ys)
    ux1 = max(b["x"] + b["w"] for b in lbs)
    uy1 = max(b["y"] + b["h"] for b in lbs)
    bb = el["bbox"]
    assert abs(bb["x"] - ux) < 1e-3 and abs(bb["y"] - uy) < 1e-3
    assert abs(bb["w"] - (ux1 - ux)) < 1e-3 and abs(bb["h"] - (uy1 - uy)) < 1e-3


def test_corpus_parse_is_deterministic(tmp_path):
    # The generator + parser are deterministic: same build → same parsed texts.
    a = build_corpus(tmp_path / "a")
    b = build_corpus(tmp_path / "b")
    by_name_b = {d.name: d for d in b}
    for da in a:
        ta = _element_texts(parse_deck(da.path))
        tb = _element_texts(parse_deck(by_name_b[da.name].path))
        assert ta == tb, f"{da.name}: non-deterministic parse {ta!r} != {tb!r}"
