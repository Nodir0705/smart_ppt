"""Positional element-id remapping between two index builds. Synthetic indexes only."""
from __future__ import annotations

from tools.remap_ids import build_id_map


def _el(eid, type, text, x, y, w=2.0, h=0.4):
    return {"id": eid, "type": type, "text": text, "bbox": {"x": x, "y": y, "w": w, "h": h}}


def _index(*slides, deck="deck.pptx"):
    """slides: lists of (type, text, x, y[, w, h]); ids are positional like the parser's."""
    out = []
    for n, els in enumerate(slides, 1):
        out.append({"slide_number": n, "elements": [
            _el(f"s{n}_e{i}", *e) for i, e in enumerate(els)]})
    return {"deck": deck, "slide_size": {"w": 13.333, "h": 7.5, "units": "inches"}, "slides": out}


OLD = [("title", "Where the Time Goes", 0.6, 0.4, 12.0, 0.7),
       ("arrow", None, 1.0, 2.0, 1.5, 0.05),
       ("bullet", "STT dominates", 1.0, 3.0),
       ("body", "Match is cheap", 1.0, 3.5)]


def test_insertion_shifts_later_ids():
    new = [OLD[0], ("body", "x = y + 1", 5.0, 1.5), *OLD[1:]]   # equation box now emitted at e1
    r = build_id_map(_index(OLD), _index(new))
    assert r["deck"] == "deck.pptx"
    assert r["map"] == {"s1_e0": "s1_e0", "s1_e1": "s1_e2", "s1_e2": "s1_e3", "s1_e3": "s1_e4"}
    assert r["new"] == ["s1_e1"] and r["dropped"] == [] and r["retyped"] == []


def test_deletion_drops_old_id_and_shifts_down():
    r = build_id_map(_index(OLD), _index([OLD[0], *OLD[2:]]))   # the tiny arrow is gone
    assert r["map"] == {"s1_e0": "s1_e0", "s1_e1": None, "s1_e2": "s1_e1", "s1_e3": "s1_e2"}
    assert r["dropped"] == ["s1_e1"] and r["new"] == []


def test_retype_without_text_change_is_matched_and_reported():
    old = [("diagram_label", "Pipeline", 1.0, 1.0, 6.0, 1.0)]
    new = [("title", "Pipeline", 1.0, 1.2, 4.0, 0.6)]        # plain title: text bbox, not the box
    r = build_id_map(_index(old), _index(new))
    assert r["map"] == {"s1_e0": "s1_e0"}
    assert r["retyped"] == [{"old": "s1_e0", "new": "s1_e0", "from": "diagram_label", "to": "title"}]


def test_duplicate_texts_resolved_by_iou_not_order():
    old = [("chart_category", "Match", 1.0, 2.0, 0.5, 3.0), ("chart_category", "Match", 7.0, 2.0, 0.5, 3.0)]
    new = [("chart_category", "Match", 7.0, 2.0, 0.5, 3.0), ("chart_category", "Match", 1.0, 2.0, 0.5, 3.0)]
    r = build_id_map(_index(old), _index(new))
    assert r["map"] == {"s1_e0": "s1_e1", "s1_e1": "s1_e0"}


def test_textless_arrows_match_by_iou_or_drop():
    old = [("arrow", None, 1.0, 2.0, 1.5, 0.05), ("arrow", None, 8.0, 6.0, 0.1, 0.1),
           ("arrow", None, 3.0, 4.0, 2.0, 0.0)]                 # zero-height line
    new = [("arrow", None, 1.1, 2.0, 1.5, 0.05), ("arrow", None, 3.0, 4.0, 2.0, 0.0)]
    r = build_id_map(_index(old), _index(new))
    assert r["map"] == {"s1_e0": "s1_e0", "s1_e1": None, "s1_e2": "s1_e1"}
    assert r["dropped"] == ["s1_e1"]


def test_symbol_only_text_matches_like_textless():
    old = [("body", "✓", 1.0, 1.0, 0.3, 0.3), ("body", "✗", 2.0, 1.0, 0.3, 0.3)]
    r = build_id_map(_index(old), _index(old))
    assert r["map"] == {"s1_e0": "s1_e0", "s1_e1": "s1_e1"}


def test_slides_only_on_one_side():
    r = build_id_map(_index(OLD, OLD), _index(OLD))
    assert r["dropped"] == ["s2_e0", "s2_e1", "s2_e2", "s2_e3"]
    r = build_id_map(_index(OLD), _index(OLD, OLD))
    assert r["new"] == ["s2_e0", "s2_e1", "s2_e2", "s2_e3"]
