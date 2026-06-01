from src.group_scope import chart_noun, group_label_for_chart, scope_index, load_expansions


def test_chart_noun():
    assert chart_noun("COLUMN_CLUSTERED (51)") == "bar"
    assert chart_noun("PIE (5)") == "slice"
    assert chart_noun("LINE_MARKERS (65)") == "point"
    assert chart_noun("RADAR (3)") == "item"


def test_group_label_from_slide_title():
    chart = {"id": "c1", "type": "chart", "text": "COLUMN_CLUSTERED (51)"}
    assert group_label_for_chart(chart, "Where the Time Goes") == "latency"
    assert group_label_for_chart(chart, "Build Effort Breakdown") == "effort"


def test_scope_assigns_group_id_and_label():
    index = {"slides": [{"slide_number": 4, "elements": [
        {"id": "s4_e0", "type": "title", "text": "Where the Time Goes", "bbox": {}, "trigger_phrases": []},
        {"id": "s4_e1", "type": "chart", "text": "COLUMN_CLUSTERED (51)", "bbox": {}, "trigger_phrases": [], "parent_id": None},
        {"id": "s4_e3", "type": "chart_category", "text": "Match", "bbox": {}, "trigger_phrases": ["match"], "parent_id": "s4_e1"},
        {"id": "s4_e6", "type": "chart", "text": "PIE (5)", "bbox": {}, "trigger_phrases": [], "parent_id": None},
        {"id": "s4_e8", "type": "chart_category", "text": "Matching", "bbox": {}, "trigger_phrases": ["matching"], "parent_id": "s4_e6"},
    ]}]}
    out = scope_index(index, expansions={})
    e3 = out["slides"][0]["elements"][2]
    assert e3["group_id"] == "s4_e1"
    assert e3["group_label"] == "latency"
    # scoped phrases added
    assert "the match bar" in e3["trigger_phrases"]
    assert "match latency" in e3["trigger_phrases"]
    # pie slice gets slice noun + (effort? slide title has no effort word -> fallback)
    e8 = out["slides"][0]["elements"][4]
    assert "the matching slice" in e8["trigger_phrases"]
    assert e8["group_id"] == "s4_e6"


def test_scope_adds_acronym_expansion():
    index = {"slides": [{"slide_number": 6, "elements": [
        {"id": "s6_e1", "type": "diagram_label", "text": "STT Engine", "bbox": {}, "trigger_phrases": ["stt engine"], "parent_id": None},
    ]}]}
    out = scope_index(index, expansions={"STT": "speech to text"})
    ph = out["slides"][0]["elements"][0]["trigger_phrases"]
    assert "speech to text" in ph
    assert "speech to text engine" in ph
    assert out["slides"][0]["elements"][0]["group_id"] == "s6_e1"  # singleton


def test_scope_is_idempotent():
    index = {"slides": [{"slide_number": 6, "elements": [
        {"id": "s6_e1", "type": "diagram_label", "text": "STT Engine", "bbox": {}, "trigger_phrases": ["stt engine"], "parent_id": None}]}]}
    once = scope_index(index, expansions={"STT": "speech to text"})
    n1 = len(once["slides"][0]["elements"][0]["trigger_phrases"])
    twice = scope_index(once, expansions={"STT": "speech to text"})
    n2 = len(twice["slides"][0]["elements"][0]["trigger_phrases"])
    assert n1 == n2
