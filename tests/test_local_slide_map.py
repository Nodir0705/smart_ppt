import json

from tools.local_slide_map import compare, fallback_map, generate, map_schema
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
    bad = dict(GOOD, order=["u1"])
    chat, calls = _client([json.dumps(bad), "```json\n" + json.dumps(GOOD) + "\n```"])
    m, st = generate(PACKET, chat, image=False)
    assert st["result"] == "repaired" and m == GOOD
    assert "order must list every uid" in calls[1][-1]["content"]   # errors fed back
    chat, _ = _client([json.dumps(bad), "not json"])
    m, st = generate(PACKET, chat, image=False)
    assert st["result"] == "fallback" and validate(m, PACKET) == []


def test_compare_pairs_and_agreement():
    c = compare(fallback_map(PACKET), GOOD)
    assert (c["tp"], c["fp"], c["fn"]) == (0, 0, 1)   # missed the s2_e2+s2_e3 card
    assert c["point_eq"] == 5 and c["role_eq"] == 4     # only s2_e2 differs (bullet vs card_heading)
