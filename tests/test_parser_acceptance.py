"""Stage 1 acceptance test against test_deck.pptx.

The bullets here mirror the "Stage 1 acceptance test" section of CLAUDE.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# allow `python -m pytest` from repo root with src/ as a package
ROOT = Path(__file__).resolve().parents[1]

from src.parser import parse_deck  # noqa: E402


@pytest.fixture(scope="module")
def index() -> dict:
    deck = ROOT / "test_deck.pptx"
    assert deck.exists(), f"test deck missing: {deck}"
    return parse_deck(str(deck))


def _elements(index: dict, slide_no: int) -> list[dict]:
    for s in index["slides"]:
        if s["slide_number"] == slide_no:
            return s["elements"]
    raise AssertionError(f"slide {slide_no} not in index")


def _types(elements: list[dict]) -> list[str]:
    return [e["type"] for e in elements]


def _texts_of(elements: list[dict], type_: str) -> list[str | None]:
    return [e["text"] for e in elements if e["type"] == type_]


# ---- 7 slides ----

def test_seven_slides(index):
    assert len(index["slides"]) == 7


def test_slide_size_recorded(index):
    sz = index["slide_size"]
    assert sz["units"] == "inches"
    assert abs(sz["w"] - 13.33) < 0.05
    assert abs(sz["h"] - 7.50) < 0.05


# ---- Slide 2: bullets + "< 1 ms" stat ----

def test_slide2_has_bullets(index):
    els = _elements(index, 2)
    bullets = [e for e in els if e["type"] == "bullet"]
    assert len(bullets) >= 3, f"expected several bullets on slide 2, got {len(bullets)}"
    joined = " | ".join((b["text"] or "") for b in bullets).lower()
    assert "parse" in joined
    assert "index" in joined


def test_slide2_has_stat_callout(index):
    els = _elements(index, 2)
    stats = [e for e in els if e["type"] == "stat"]
    assert len(stats) == 1, f"expected one stat element, got {len(stats)}: {stats}"
    stat = stats[0]
    assert "1 ms" in (stat["text"] or "").lower() or "< 1" in (stat["text"] or "")
    # stat should carry the container box bbox (taller than the inner text shape)
    assert stat["bbox"]["h"] > 1.5, f"stat bbox should be the container box, got {stat['bbox']}"


# ---- Slide 3: table cells ----

def test_slide3_table_cells(index):
    els = _elements(index, 3)
    cells = [e for e in els if e["type"] == "table_cell"]
    assert len(cells) == 15, f"expected 15 table cells (5x3), got {len(cells)}"
    texts = {(e["text"] or "").lower() for e in cells}
    for required in ["stage", "when", "cost", "parse & index", "offline", "slow (once)",
                     "speech to text", "live", "streaming", "match & rank",
                     "sub-millisecond", "highlight overlay", "one frame"]:
        assert required in texts, f"missing cell text: {required!r}"


def test_slide3_cells_have_distinct_bboxes(index):
    els = _elements(index, 3)
    cells = [e for e in els if e["type"] == "table_cell"]
    bboxes = {(round(e["bbox"]["x"], 2), round(e["bbox"]["y"], 2)) for e in cells}
    assert len(bboxes) == 15, "each cell should have a unique (x,y) — got duplicates"


# ---- Slide 4: two charts with their categories ----

def test_slide4_two_charts(index):
    els = _elements(index, 4)
    charts = [e for e in els if e["type"] == "chart"]
    assert len(charts) == 2, f"expected 2 charts on slide 4, got {len(charts)}"


def test_slide4_chart_categories(index):
    els = _elements(index, 4)
    cats = {(e["text"] or "") for e in els if e["type"] == "chart_category"}
    for required in ["STT", "Match", "Render", "Sync", "Parsing", "Matching", "Overlay UI"]:
        assert required in cats, f"missing chart category: {required!r}"


# ---- Slide 5: one line chart with 5 threshold categories ----

def test_slide5_line_chart_with_5_categories(index):
    els = _elements(index, 5)
    charts = [e for e in els if e["type"] == "chart"]
    assert len(charts) == 1, f"expected 1 chart on slide 5, got {len(charts)}"
    cats = [e for e in els if e["type"] == "chart_category"]
    assert len(cats) == 5, f"expected 5 chart_category elements, got {len(cats)}"
    texts = {(e["text"] or "") for e in cats}
    assert texts == {"0.1", "0.3", "0.5", "0.7", "0.9"}, f"got categories {texts}"


# ---- Slide 6: 4 diagram_labels with BOX bbox; arrows recorded but empty ----

def test_slide6_four_diagram_labels(index):
    els = _elements(index, 6)
    labels = [e for e in els if e["type"] == "diagram_label"]
    assert len(labels) == 4, f"expected 4 diagram_label elements, got {len(labels)}: {[(e['text']) for e in labels]}"
    texts = {(e["text"] or "") for e in labels}
    assert texts == {"Microphone", "STT Engine", "Matcher", "Overlay"}, f"got {texts}"


def test_slide6_diagram_labels_use_box_bbox(index):
    els = _elements(index, 6)
    labels = [e for e in els if e["type"] == "diagram_label"]
    # Each box in the test deck is 2.40 x 1.40 inches; bbox should reflect that
    # (not a zero-height text shape, and not a tiny text-line bbox).
    for lab in labels:
        assert lab["bbox"]["w"] > 1.5, f"label {lab['text']!r} has too-narrow bbox: {lab['bbox']}"
        assert lab["bbox"]["h"] > 1.0, f"label {lab['text']!r} has too-short bbox: {lab['bbox']}"


def test_slide6_arrows_recorded(index):
    els = _elements(index, 6)
    arrows = [e for e in els if e["type"] == "arrow"]
    assert len(arrows) >= 3, f"expected ≥3 arrow elements on slide 6, got {len(arrows)}"
    for a in arrows:
        assert a["text"] in (None, "")
        assert a["trigger_phrases"] == []


# ---- Slide 7: one image with needs_vision = True ----

def test_slide7_image_needs_vision(index):
    els = _elements(index, 7)
    images = [e for e in els if e["type"] == "image"]
    assert len(images) == 1, f"expected 1 image on slide 7, got {len(images)}"
    assert images[0]["needs_vision"] is True
    assert images[0]["trigger_phrases"] == []
