import json

from tools.local_slide_map import (compare, fallback_map, fixup, generate, map_schema, speak_terms, spoken,
                                   spoken_terms)
from tools.slide_map import validate

PACKET = {"slide": 2, "slide_size": {"w": 13.33, "h": 7.5, "units": "inches"}, "png": "unused.png",
          "elements": [
              {"id": "s2_e0", "type": "title", "text": "How it works", "bbox": {"x": .7, "y": .8, "w": 4, "h": .4}},
              {"id": "s2_e1", "type": "arrow", "text": None, "bbox": {"x": 1, "y": 2, "w": 1, "h": .1}},
              {"id": "s2_e2", "type": "bullet", "text": "Stage A reads the page", "bbox": {"x": 1, "y": 3, "w": 3, "h": .3}},
              {"id": "s2_e3", "type": "body", "text": "with a vision model", "bbox": {"x": 1, "y": 3.4, "w": 3, "h": .3}},
              {"id": "s2_e4", "type": "body", "text": "2", "bbox": {"x": 12.6, "y": 7.07, "w": .05, "h": .08}}]}

GOOD = {"slide": 2, "message": "two stages",
        "items": {"s2_e0": {"role": "title", "pointable": True}, "s2_e1": {"role": "decoration", "pointable": False},
                  "s2_e2": {"role": "card_heading", "pointable": True}, "s2_e3": {"role": "body", "pointable": True},
                  "s2_e4": {"role": "footer", "pointable": False}},
        "units": [{"uid": "u1", "members": ["s2_e0"], "summary": "title", "talk": ["a", "b", "c"], "terms": []},
                  {"uid": "u2", "members": ["s2_e2", "s2_e3"], "summary": "stage a", "talk": ["a", "b", "c"],
                   "terms": []}],
        "relations": [], "order": ["u1", "u2"]}


def test_schema_requires_every_id_and_limits_members():
    s = map_schema(PACKET)
    ids = [e["id"] for e in PACKET["elements"]]
    assert s["properties"]["items"]["required"] == ids
    assert s["properties"]["units"]["items"]["properties"]["members"]["items"]["enum"] == ids


def test_fallback_is_valid_and_skips_furniture():
    m = fallback_map(PACKET)
    assert validate(m, PACKET) == []
    assert not m["items"]["s2_e1"]["pointable"] and not m["items"]["s2_e4"]["pointable"]
    assert [u["members"] for u in m["units"]] == [["s2_e0"], ["s2_e2"], ["s2_e3"]]


def _client(replies):
    calls = []

    def chat(messages, schema):
        calls.append(messages)
        return {"message": {"content": replies[len(calls) - 1]}}
    return chat, calls


def test_repair_then_fallback():
    bad = dict(GOOD, message="")   # not bookkeeping: fix-up leaves it to the repair turn
    chat, calls = _client([json.dumps(bad), "```json\n" + json.dumps(GOOD) + "\n```"])
    m, st = generate(PACKET, chat, image=False)
    assert st["result"] == "repaired" and m == GOOD
    assert "message missing" in calls[1][-1]["content"]   # errors fed back
    chat, _ = _client([json.dumps(bad), "not json"])
    m, st = generate(PACKET, chat, image=False)
    assert st["result"] == "fallback" and validate(m, PACKET) == []


def test_compare_pairs_and_agreement():
    c = compare(fallback_map(PACKET), GOOD)
    assert (c["tp"], c["fp"], c["fn"]) == (0, 0, 1)   # missed the s2_e2+s2_e3 card
    assert c["point_eq"] == 5 and c["role_eq"] == 4     # only s2_e2 differs (bullet vs card_heading)


def test_fixup_is_a_noop_on_a_valid_map():
    assert fixup(GOOD, PACKET) == (GOOD, {})


def test_fixup_repairs_bookkeeping_without_regrouping():
    items = {**GOOD["items"], "s2_e3": {"role": "body", "pointable": False},      # member, not pointable
             "s2_e1": {"role": "decoration", "pointable": False}}
    bad = dict(GOOD, items=items, units=[
        dict(GOOD["units"][0], members=["s2_e0", "s2_e9"]),                       # unknown id
        dict(GOOD["units"][1], members=["s2_e0", "s2_e2", "s2_e3", "s2_e1"],      # dup + decoration
             talk=["a", " ", "b", "c"]),
        {"uid": "u7", "members": ["s2_e1"], "summary": "x", "talk": ["a", "b", "c"], "terms": []}],
        relations=[{"from": "u1", "to": "u2", "type": "flow"}, {"from": "u2", "to": "u7", "type": "flow"}],
        order=["u2", "u2", "u9"])
    m, fx = fixup(bad, PACKET)
    assert validate(m, PACKET) == []
    assert [u["members"] for u in m["units"]] == [["s2_e0"], ["s2_e2", "s2_e3"]]
    assert m["items"]["s2_e3"]["pointable"] and m["units"][1]["talk"] == ["a", "b", "c"]
    assert m["relations"] == [{"from": "u1", "to": "u2", "type": "flow"}] and m["order"] == ["u2", "u1"]
    assert fx == {"unknown_id": 1, "duplicate_member": 1, "nonpointable_member_removed": 2,
                  "member_made_pointable": 1, "blank_talk": 1, "empty_unit": 1, "bad_relation": 1,
                  "order_rebuilt": 1}
    assert bad["units"][0]["members"] == ["s2_e0", "s2_e9"]                     # input untouched


def test_fixup_gives_an_orphan_its_own_unit_with_spoken_terms():
    pk = dict(PACKET, elements=PACKET["elements"] + [
        {"id": "s2_e5", "type": "body", "text": "~1 min", "bbox": {"x": 8, "y": 5, "w": 1, "h": .3}}])
    bad = dict(GOOD, items={**GOOD["items"], "s2_e5": {"role": "stat", "pointable": True}})
    m, fx = fixup(bad, pk)
    assert validate(m, pk) == [] and fx == {"orphan_unit": 1, "order_rebuilt": 1}
    assert m["units"][-1]["uid"] == "u3" and m["units"][-1]["members"] == ["s2_e5"]
    assert m["units"][-1]["terms"] == ["about one minute"] and m["order"] == ["u1", "u2", "u3"]


def test_spoken_terms():
    assert spoken_terms("15–33 items, 30+ times, 8 GB, 2.5x, 95%") == [
        "fifteen to thirty three", "thirty plus", "eight gigabytes", "two point five times", "ninety five percent"]
    assert spoken_terms("Qwen3 and OCR2 v2") == []


def test_fixup_rescues_before_repair():
    bad = dict(GOOD, order=["u1"], items={**GOOD["items"], "s2_e3": {"role": "body", "pointable": False}})
    chat, calls = _client([json.dumps(bad)])
    m, st = generate(PACKET, chat, image=False)
    assert st["result"] == "fixed" and len(calls) == 1 and validate(m, PACKET) == []
    assert st["fixes"] == [{"member_made_pointable": 1, "order_rebuilt": 1}]


def test_fixup_slide_number_and_missing_items():
    items = {k: v for k, v in GOOD["items"].items() if k != "s2_e1"}   # an arrow left out
    m, fx = fixup(dict(GOOD, slide=0, items=items), PACKET)
    assert validate(m, PACKET) == [] and fx == {"slide_number": 1, "missing_item": 1}
    assert m["items"]["s2_e1"] == {"role": "decoration", "pointable": False}


def test_spoken_forms_of_model_terms():
    assert [spoken(t) for t in ["Nanonets-OCR2", "qwen3-4b-instruct", "8 GB", "15-33", "GPUs", "A/F",
                                "python nano.py", "THE PROBLEM", "2.5x"]] == [
        "nanonets o c r two", "qwen three four b instruct", "eight gigabytes", "fifteen to thirty three",
        "g p u s", "a f", "python nano dot py", "the problem", "two point five times"]
    m = speak_terms(dict(GOOD, units=[dict(GOOD["units"][0], terms=["OCR", "o c r", "", "~1 min"])]))
    assert m["units"][0]["terms"] == ["o c r", "about one minute"]
