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


# ---- applying the map ----

from tools.remap_ids import main, remap_file, remap_json   # noqa: E402
from tools.slide_map import validate   # noqa: E402

NEW = [OLD[0], ("body", "x = y + 1", 5.0, 1.5), OLD[0 + 2], OLD[3]]   # arrow dropped, equation inserted at e1
MAP = {"s1_e0": "s1_e0", "s1_e1": None, "s1_e2": "s1_e2", "s1_e3": "s1_e3"}


def test_remap_json_keys_values_lists_no_chaining():
    chain = {"s1_e1": "s1_e2", "s1_e2": "s1_e3", "s1_e3": None}
    obj = {"s1_e1": ["s1_e1", "s1_e2", "s1_e3", "free text"], "s1_e3": 1,
           "active_id": "s1_e3", "nested": [{"s1_e2": "s1_e1"}], "slide": 1}
    out, dropped = remap_json(obj, chain)
    assert out == {"s1_e2": ["s1_e2", "s1_e3", "free text"], "active_id": None,
                   "nested": [{"s1_e3": "s1_e2"}], "slide": 1}
    assert dropped == ["s1_e3"]
    assert obj["s1_e1"] == ["s1_e1", "s1_e2", "s1_e3", "free text"]   # input untouched


def _slide_map():
    return {"slide": 1, "message": "latency breakdown",
            "items": {"s1_e0": {"role": "title", "pointable": True},
                      "s1_e1": {"role": "connector", "pointable": True},
                      "s1_e2": {"role": "bullet", "pointable": True},
                      "s1_e3": {"role": "body", "pointable": True}},
            "units": [{"uid": "u1", "members": ["s1_e0"], "summary": "title", "talk": ["a", "b", "c"], "terms": []},
                      {"uid": "u2", "members": ["s1_e1"], "summary": "arrow", "talk": ["a", "b", "c"], "terms": []},
                      {"uid": "u3", "members": ["s1_e2", "s1_e3"], "summary": "stt", "talk": ["a", "b", "c"],
                       "terms": []}],
            "relations": [{"type": "flow", "from": "u2", "to": "u3"}, {"type": "explains", "from": "u3", "to": "u1"}],
            "order": ["u1", "u2", "u3"]}


def test_remap_slide_map_drops_empty_unit_adds_new_id_and_validates():
    new_index = _index(NEW)
    m, rep = remap_file(_slide_map(), MAP, new_index)
    assert set(m["items"]) == {"s1_e0", "s1_e1", "s1_e2", "s1_e3"}
    assert m["items"]["s1_e1"] == {"role": "body", "pointable": True}      # the new equation box
    assert [u["uid"] for u in m["units"]] == ["u1", "u3", "u4"]
    assert m["units"][2]["members"] == ["s1_e1"] and m["units"][2]["summary"] == "x = y + 1"
    assert m["relations"] == [{"type": "explains", "from": "u3", "to": "u1"}]
    assert m["order"] == ["u1", "u3", "u4"]
    assert rep["dropped"] == ["s1_e1"] and rep["new"] == ["s1_e1"]
    packet = {"slide": 1, "elements": new_index["slides"][0]["elements"]}
    assert validate(m, packet) == []


def test_remap_presenter_script_lists_dropped_segments():
    script = {"deck": "deck.pptx", "slides": [{"slide": 1, "segments": [
        {"text": "Look at the arrow and the title.", "refers_to": ["s1_e0", "s1_e1"]},
        {"text": "STT dominates.", "refers_to": ["s1_e2"]}]}]}
    out, rep = remap_file(script, {**MAP, "s1_e2": "s1_e5"}, _index(NEW))
    assert out["slides"][0]["segments"][0]["refers_to"] == ["s1_e0"]
    assert out["slides"][0]["segments"][1]["refers_to"] == ["s1_e5"]
    assert rep["dropped"] == ["s1_e1"] and rep["remapped"] == 1 and rep["new"] == []
    assert rep["segments"] == [(1, ["s1_e1"], "Look at the arrow and the title.")]


def test_apply_rewrites_tree_in_place(tmp_path, capsys):
    import json
    (tmp_path / "old.json").write_text(json.dumps(_index(OLD)))
    (tmp_path / "new.json").write_text(json.dumps(_index(NEW)))
    maps = tmp_path / "maps"
    (maps / "packets").mkdir(parents=True)
    (maps / "slide_001.json").write_text(json.dumps(_slide_map(), indent=1) + "\n")
    packet = json.dumps({"slide": 1, "png": "x.png", "elements": [{"id": "s1_e1"}]})
    (maps / "packets" / "slide_001.json").write_text(packet)
    (maps / "notes.txt").write_text("s1_e1 stays")
    script = tmp_path / "presenter_script.json"
    script.write_text(json.dumps({"deck": "d", "slides": [{"slide": 1, "segments": [
        {"text": "The arrow.", "refers_to": ["s1_e1", "s1_e3"]}]}]}, indent=2))
    assert main([str(tmp_path / "old.json"), str(tmp_path / "new.json"), "--out", str(tmp_path / "map.json"),
                 "--apply", str(maps), str(script)]) == 0
    assert json.loads((tmp_path / "map.json").read_text())["map"] == MAP
    m = json.loads((maps / "slide_001.json").read_text())
    assert [u["uid"] for u in m["units"]] == ["u1", "u3", "u4"]
    assert (maps / "slide_001.json").read_text().startswith('{\n "slide": 1,')      # indent=1 kept
    assert (maps / "packets" / "slide_001.json").read_text() == packet
    assert (maps / "notes.txt").read_text() == "s1_e1 stays"
    raw = script.read_text()
    assert json.loads(raw)["slides"][0]["segments"][0]["refers_to"] == ["s1_e3"]
    assert raw.startswith('{\n  "deck"') and not raw.endswith("\n")             # indent=2, no newline kept
    out = capsys.readouterr().out
    assert "The arrow." in out and "s1_e1" in out and "packets/slide_001.json" in out
