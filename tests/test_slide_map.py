from tools.slide_map import validate

PACKET = {"slide": 1, "elements": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}


def _map(**over):
    m = {"slide": 1, "message": "point",
         "items": {"a": {"role": "card_heading", "pointable": True},
                   "b": {"role": "body", "pointable": True},
                   "c": {"role": "footer", "pointable": False}},
         "units": [{"uid": "u1", "members": ["a", "b"], "summary": "s",
                    "talk": ["x one", "x two", "x three"], "terms": []}],
         "relations": [], "order": ["u1"]}
    m.update(over)
    return m


def test_valid_map():
    assert validate(_map(), PACKET) == []


def test_catches_coverage_ownership_and_order_errors():
    errs = validate(_map(units=[{"uid": "u1", "members": ["a", "c"], "summary": "s",
                                 "talk": ["1", "2", "3"]}], order=[]), PACKET)
    text = " ".join(errs)
    assert "not pointable" in text          # footer put in a unit
    assert "pointable b belongs to no unit" in text
    assert "order must list" in text
    assert validate(_map(items={"a": {"role": "body", "pointable": True}}), PACKET)
