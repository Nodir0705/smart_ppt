"""Tests for tools.coverage: PDF-word vs element-region bookkeeping, overlay, CLI."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tools.coverage import draw_overlay, main, slide_coverage

ROOT = Path(__file__).resolve().parents[1]
DECK_DIR = ROOT / "out" / "bench" / "test_deck" / "default"
SIZE = {"w": 13.333, "h": 7.5}


def _el(eid, etype, x, y, w, h, line_bboxes=None, needs_vision=False):
    e = {"id": eid, "type": etype, "text": "t", "needs_vision": needs_vision,
         "bbox": {"x": x, "y": y, "w": w, "h": h}}
    if line_bboxes is not None:
        e["line_bboxes"] = [{"x": a, "y": b, "w": c, "h": d} for a, b, c, d in line_bboxes]
    return e


def _w(text, x0, y0, x1, y1):
    return {"text": text, "x0": x0, "y0": y0, "x1": x1, "y1": y1}


def _cov(elements, words):
    return slide_coverage({"slide_number": 1, "elements": elements}, words, SIZE)


def test_word_inside_element_is_claimed():
    r = _cov([_el("s1_e0", "body", 1, 1, 2, 0.5)], [_w("hello", 1.1, 1.1, 1.6, 1.3)])
    assert r["words"] == 1 and r["claimed"] == 1
    assert r["unclaimed_words"] == 0 and r["unclaimed_lines"] == []
    assert r["empty_text"] == []


def test_word_outside_element_is_unclaimed_line():
    r = _cov([_el("s1_e0", "body", 1, 1, 2, 0.5)],
             [_w("hello", 1.1, 1.1, 1.6, 1.3), _w("stray", 5.0, 3.0, 5.6, 3.2)])
    assert r["claimed"] == 1 and r["unclaimed_words"] == 1
    assert len(r["unclaimed_lines"]) == 1
    line = r["unclaimed_lines"][0]
    assert line["text"] == "stray" and line["footer"] is False
    assert line["bbox"] == {"x": 5.0, "y": 3.0, "w": 0.6, "h": 0.2}


def test_words_under_two_alphanumerics_are_ignored():
    r = _cov([], [_w("a", 5.0, 3.0, 5.1, 3.2), _w("1.", 5.2, 3.0, 5.3, 3.2),
                  _w("ok", 6.0, 3.0, 6.3, 3.2)])
    assert r["unclaimed_words"] == 1
    assert [ln["text"] for ln in r["unclaimed_lines"]] == ["ok"]


def test_footer_line_flagged_and_excluded_from_word_count():
    r = _cov([], [_w("page", 6.0, 7.1, 6.5, 7.3), _w("body", 6.0, 3.0, 6.5, 3.2)])
    flags = {ln["text"]: ln["footer"] for ln in r["unclaimed_lines"]}
    assert flags == {"page": True, "body": False}
    assert r["unclaimed_words"] == 1
    assert len(r["unclaimed_lines"]) == 2


def test_text_element_without_words_is_empty_but_image_is_not():
    r = _cov([_el("s1_e0", "body", 1, 1, 2, 0.5), _el("s1_e1", "image", 5, 1, 2, 2),
              _el("s1_e2", "bullet", 1, 4, 2, 0.5)],
             [_w("x", 1.1, 4.1, 1.2, 4.3)])  # 1-char word still backs s1_e2
    assert r["empty_text"] == ["s1_e0"]


def test_line_bboxes_preferred_over_bbox():
    el = _el("s1_e0", "body", 1, 1, 4, 2, line_bboxes=[(1, 1, 4, 0.3)])
    r = _cov([el], [_w("inside", 1.2, 1.1, 1.8, 1.25), _w("below", 1.2, 2.5, 1.8, 2.7)])
    assert r["claimed"] == 1
    assert [ln["text"] for ln in r["unclaimed_lines"]] == ["below"]
    assert r["empty_text"] == []


def test_overlay_keeps_size_and_draws_something():
    img = Image.new("RGB", (400, 225), "white")
    out = draw_overlay(img, [_el("s1_e0", "body", 1, 1, 2, 0.5),
                             _el("s1_e1", "image", 5, 2, 3, 2, needs_vision=True)], SIZE)
    assert out.size == (400, 225)
    assert out.getcolors(maxcolors=400 * 225) != [(400 * 225, (255, 255, 255))]


@pytest.mark.skipif(not (DECK_DIR / "index.scoped.json").exists(),
                    reason="test_deck not built under out/bench")
def test_main_writes_report_and_overlays(tmp_path):
    assert main([str(DECK_DIR), "--out", str(tmp_path)]) == 0
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()
    assert sorted(p.name for p in tmp_path.glob("slide_*.png")) == [
        f"slide_{n:03d}.png" for n in range(1, 8)]
