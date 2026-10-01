"""Unit tests for tools.bakeoff_score — hand-built tiny inputs, no I/O."""
from __future__ import annotations

from tools.bakeoff_score import (
    icon_false_positive_rate,
    normalize_words,
    score_slide,
)


def _el(eid, text, x, y, w, h, etype="body"):
    return {"id": eid, "type": etype, "text": text,
            "bbox": {"x": x, "y": y, "w": w, "h": h}}


def _ocr(text, x, y, w, h, conf=90.0):
    return {"text": text, "bbox": [x, y, w, h], "conf": conf}


def test_normalize_words_strips_punct_and_lowers():
    assert normalize_words("Hello, World!") == ["hello", "world"]
    assert normalize_words("  multi   space  ") == ["multi", "space"]
    assert normalize_words("") == []


def test_perfect_match():
    gt = [_el("e1", "Net Revenue", 1.0, 1.0, 2.0, 0.5)]
    ocr = [_ocr("Net Revenue", 1.0, 1.0, 2.0, 0.5)]
    s = score_slide(gt, ocr)
    assert s["word_overlap"] == s["gt_word_count"] == s["ocr_word_count"] == 2
    assert s["element_hits"] == 1
    assert s["gt_element_count"] == 1


def test_missing_word_hurts_recall_not_precision():
    gt = [_el("e1", "Net Revenue Growth", 1.0, 1.0, 2.0, 0.5)]
    ocr = [_ocr("Net Revenue", 1.0, 1.0, 2.0, 0.5)]  # "Growth" never detected
    s = score_slide(gt, ocr)
    assert s["gt_word_count"] == 3
    assert s["word_overlap"] == 2
    # recall = 2/3 < 1; precision = 2/2 = 1 (nothing hallucinated)
    assert s["ocr_word_count"] == 2
    # element coverage 2/3 < 0.8 threshold -> element miss
    assert s["element_hits"] == 0


def test_hallucinated_word_hurts_precision_not_recall():
    gt = [_el("e1", "Net Revenue", 1.0, 1.0, 2.0, 0.5)]
    ocr = [_ocr("Net Revenue Bogus", 1.0, 1.0, 2.0, 0.5)]
    s = score_slide(gt, ocr)
    assert s["word_overlap"] == s["gt_word_count"] == 2
    assert s["ocr_word_count"] == 3  # precision = 2/3 < 1
    # element: candidate words cover both GT words -> still a hit (extra token is tolerated)
    assert s["element_hits"] == 1


def test_bbox_mismatch_causes_element_miss_even_with_word_overlap():
    gt = [_el("e1", "Net Revenue", 1.0, 1.0, 2.0, 0.5)]
    # Same words, but OCR box is far away (no intersection) -> element should not count,
    # though the words still count toward recall/precision (OCR found the text, just
    # not grounded at the right place).
    ocr = [_ocr("Net Revenue", 8.0, 6.0, 1.0, 0.3)]
    s = score_slide(gt, ocr)
    assert s["word_overlap"] == 2
    assert s["element_hits"] == 0


def test_icon_false_positive_rate():
    icons = {
        "deck/1/e1": [],                      # clean crop, no filtered tokens -> no FP
        "deck/1/e2": [{"text": "os", "conf": 61.0}],  # survived filter -> FP
        "deck/1/e3": [],
    }
    r = icon_false_positive_rate(icons)
    assert r["n"] == 3
    assert r["fp"] == 1
    assert r["rate"] == 1 / 3


def test_icon_false_positive_rate_empty():
    assert icon_false_positive_rate({}) == {"n": 0, "fp": 0, "rate": None}
