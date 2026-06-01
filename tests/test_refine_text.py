"""Tests for PDF caching (render.deck_pdf) and text-bbox refinement."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "test_deck.pptx"

SOFFICE_MISSING = shutil.which("soffice") is None


@pytest.mark.skipif(SOFFICE_MISSING, reason="soffice not installed")
def test_deck_pdf_caches(tmp_path):
    from src.render import deck_pdf

    pdf = deck_pdf(DECK, tmp_path)
    assert pdf.exists()
    assert pdf.suffix == ".pdf"
    mtime = pdf.stat().st_mtime

    # Second call must be a cache hit: same path, not regenerated.
    pdf2 = deck_pdf(DECK, tmp_path)
    assert pdf2 == pdf
    assert pdf2.stat().st_mtime == mtime


def test_words_to_inches_scale_one():
    from src.refine_text import words_to_inches

    words = [{"text": "Hi", "x0": 72.0, "x1": 144.0, "top": 36.0, "bottom": 72.0}]
    out = words_to_inches(words, 1.0, 1.0)
    assert out[0]["text"] == "Hi"
    assert out[0]["x0"] == pytest.approx(1.0)
    assert out[0]["x1"] == pytest.approx(2.0)
    assert out[0]["y0"] == pytest.approx(0.5)
    assert out[0]["y1"] == pytest.approx(1.0)


def test_assign_words_smallest_box_wins():
    from src.refine_text import assign_words

    big = {"x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0, "units": "inches"}
    small = {"x": 1.0, "y": 1.0, "w": 2.0, "h": 2.0, "units": "inches"}
    word = {"text": "x", "x0": 1.5, "y0": 1.5, "x1": 2.0, "y1": 2.0}  # center (1.75,1.75)
    assigned = assign_words([("big", big), ("small", small)], [word])
    assert assigned["small"] == [word]
    assert assigned["big"] == []


def test_assign_words_drops_outside():
    from src.refine_text import assign_words

    box = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "units": "inches"}
    word = {"text": "x", "x0": 5.0, "y0": 5.0, "x1": 5.5, "y1": 5.5}
    assigned = assign_words([("e", box)], [word])
    assert assigned["e"] == []


def test_cluster_lines_two_rows():
    from src.refine_text import cluster_lines

    w = lambda t, x0, y0: {"text": t, "x0": x0, "y0": y0, "x1": x0 + 0.5, "y1": y0 + 0.2}
    words = [w("b", 2.0, 0.0), w("a", 1.0, 0.0), w("c", 1.0, 1.0)]  # row0: a,b ; row1: c
    lines = cluster_lines(words, y_tol=0.03)
    assert [[x["text"] for x in line] for line in lines] == [["a", "b"], ["c"]]


def test_union_bbox_tight():
    from src.refine_text import union_bbox

    words = [
        {"x0": 1.0, "y0": 1.0, "x1": 2.0, "y1": 1.5},
        {"x0": 2.5, "y0": 1.1, "x1": 3.0, "y1": 1.4},
    ]
    box = union_bbox(words)
    assert box == {"x": 1.0, "y": 1.0, "w": 2.0, "h": 0.5, "units": "inches"}


def test_apply_refinement_no_words_keeps_shape():
    from src.refine_text import _apply_refinement

    e = {"id": "e1", "type": "body", "text": "hi",
         "bbox": {"x": 1.0, "y": 1.0, "w": 4.0, "h": 2.0, "units": "inches"}}
    _apply_refinement(e, [])
    assert e["bbox_source"] == "shape"
    assert e["line_bboxes"] == []
    assert e["shape_bbox"] == {"x": 1.0, "y": 1.0, "w": 4.0, "h": 2.0, "units": "inches"}
    assert e["bbox"] == e["shape_bbox"]


def test_apply_refinement_sets_pdf_text():
    from src.refine_text import _apply_refinement

    e = {"id": "e1", "type": "body", "text": "hi there",
         "bbox": {"x": 0.0, "y": 0.0, "w": 9.0, "h": 3.0, "units": "inches"}}
    words = [
        {"text": "hi", "x0": 1.0, "y0": 1.0, "x1": 1.5, "y1": 1.2},
        {"text": "there", "x0": 1.6, "y0": 1.0, "x1": 2.4, "y1": 1.2},
    ]
    _apply_refinement(e, words)
    assert e["bbox_source"] == "pdf_text"
    assert e["shape_bbox"]["w"] == 9.0
    assert e["bbox"] == {"x": 1.0, "y": 1.0, "w": 1.4, "h": 0.2, "units": "inches"}
    assert len(e["line_bboxes"]) == 1
    assert e["bbox"]["w"] < e["shape_bbox"]["w"]


def test_cluster_lines_no_cascade_on_sliver_overlap():
    from src.refine_text import cluster_lines

    # consecutive line boxes overlap by only 0.05 (tight leading) -> 3 distinct lines
    w = lambda t, y0: {"text": t, "x0": 1.0, "y0": y0, "x1": 1.5, "y1": y0 + 0.2}
    words = [w("r1", 1.0), w("r2", 1.15), w("r3", 1.30)]
    lines = cluster_lines(words, y_tol=0.03)
    assert [[x["text"] for x in ln] for ln in lines] == [["r1"], ["r2"], ["r3"]]


def test_cluster_lines_groups_within_y_tol():
    from src.refine_text import cluster_lines

    # two short words, boxes do not overlap but centers are within y_tol -> same line
    a = {"text": "a", "x0": 1.0, "y0": 1.000, "x1": 1.4, "y1": 1.010}
    b = {"text": "b", "x0": 1.5, "y0": 1.011, "x1": 1.9, "y1": 1.021}
    lines = cluster_lines([a, b], y_tol=0.03)
    assert [[x["text"] for x in ln] for ln in lines] == [["a", "b"]]
