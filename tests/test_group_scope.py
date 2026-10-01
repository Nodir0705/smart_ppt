from src.group_scope import chart_noun, group_label_for_chart, scope_index


def test_chart_noun():
    assert chart_noun("COLUMN_CLUSTERED (51)") == "bar"
    assert chart_noun("PIE (5)") == "slice"
    assert chart_noun("LINE_MARKERS (65)") == "point"
    assert chart_noun("RADAR (3)") == "item"


def test_group_label_prefers_chart_title_and_cuts_head():
    chart = {"type": "chart", "text": "COLUMN_CLUSTERED (51)",
             "chart_title": "Latency by stage (ms)", "series_names": ["Latency (ms)"]}
    assert group_label_for_chart(chart, "Where the Time Goes") == ("latency", "latency by stage")


def test_group_label_uses_sole_series_name_when_untitled():
    chart = {"type": "chart", "text": "PIE (5)", "chart_title": None, "series_names": ["Effort"]}
    assert group_label_for_chart(chart, "Where the Time Goes") == ("effort", "effort")


def test_group_label_skips_default_series_name():
    chart = {"type": "chart", "text": "PIE (5)", "chart_title": None, "series_names": ["Series 1"]}
    assert group_label_for_chart(chart, "Where the Time Goes") == (
        "where the time goes", "where the time goes")


def test_group_label_multiseries_falls_back_to_slide_title():
    chart = {"type": "chart", "text": "LINE_MARKERS (65)", "chart_title": None,
             "series_names": ["Correct highlights", "False pointing"]}
    assert group_label_for_chart(chart, "Accuracy vs. Confidence Threshold") == (
        "accuracy", "accuracy vs. confidence threshold")


def test_group_label_noun_fallback():
    chart = {"type": "chart", "text": "COLUMN_CLUSTERED (51)"}
    assert group_label_for_chart(chart, None) == ("bar chart", "bar chart")


def test_group_label_legacy_index_uses_real_chart_text():
    # old indexes have no chart_title field; a non-chart-type text is a title
    chart = {"type": "chart", "text": "Quarterly Revenue"}
    assert group_label_for_chart(chart, "Anything") == ("quarterly revenue", "quarterly revenue")


def test_head_label_does_not_cut_at_of():
    chart = {"type": "chart", "text": "PIE (5)", "chart_title": "Share of revenue"}
    assert group_label_for_chart(chart, None) == ("share of revenue", "share of revenue")


def test_scope_assigns_group_id_and_label():
    index = {"slides": [{"slide_number": 4, "elements": [
        {"id": "s4_e0", "type": "title", "text": "Where the Time Goes", "bbox": {}, "trigger_phrases": []},
        {"id": "s4_e1", "type": "chart", "text": "COLUMN_CLUSTERED (51)", "bbox": {}, "trigger_phrases": [],
         "parent_id": None, "chart_title": "Latency by stage (ms)", "series_names": ["Latency (ms)"]},
        {"id": "s4_e3", "type": "chart_category", "text": "Match", "bbox": {}, "trigger_phrases": ["match"], "parent_id": "s4_e1"},
        {"id": "s4_e6", "type": "chart", "text": "PIE (5)", "bbox": {}, "trigger_phrases": [],
         "parent_id": None, "chart_title": "Build effort split", "series_names": ["Effort"]},
        {"id": "s4_e8", "type": "chart_category", "text": "Matching", "bbox": {}, "trigger_phrases": ["matching"], "parent_id": "s4_e6"},
    ]}]}
    out = scope_index(index, expansions={})
    e3 = out["slides"][0]["elements"][2]
    assert e3["group_id"] == "s4_e1"
    assert e3["group_label"] == "latency"
    ph3 = e3["trigger_phrases"]
    assert "the match bar" in ph3
    assert "match latency" in ph3
    assert "the match bar in the latency chart" in ph3
    # full-title variant emitted too
    assert "match latency by stage" in ph3
    assert "the match bar in the latency by stage chart" in ph3
    # the pie is labelled by ITS title, no longer by the slide-title keyword
    e8 = out["slides"][0]["elements"][4]
    assert e8["group_id"] == "s4_e6"
    assert e8["group_label"] == "build effort split"
    assert "the matching slice" in e8["trigger_phrases"]
    assert "the matching slice in the build effort split chart" in e8["trigger_phrases"]
    assert not any("latency" in p for p in e8["trigger_phrases"])


def test_scope_head_equals_full_emits_no_duplicates():
    index = {"slides": [{"slide_number": 1, "elements": [
        {"id": "c", "type": "chart", "text": "PIE (5)", "bbox": {}, "trigger_phrases": [],
         "chart_title": "Build effort split"},
        {"id": "k", "type": "chart_category", "text": "Parser", "bbox": {}, "trigger_phrases": [], "parent_id": "c"},
    ]}]}
    ph = scope_index(index, expansions={})["slides"][0]["elements"][1]["trigger_phrases"]
    assert len(ph) == len(set(ph))
    assert ph.count("the parser slice in the build effort split chart") == 1


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


def test_stat_phrases_expands_symbol_number_unit():
    from src.group_scope import stat_phrases
    out = stat_phrases("< 1 ms")
    assert "one millisecond" in out
    assert "less than one millisecond" in out
    assert "sub-millisecond" in out
    assert stat_phrases("How It Works") == []   # not a stat


def test_bullets_share_one_group():
    from src.group_scope import scope_index
    index = {"slides": [{"slide_number": 2, "elements": [
        {"id": "s2_e1", "type": "title", "text": "How", "bbox": {}, "trigger_phrases": []},
        {"id": "s2_e2", "type": "bullet", "text": "Parse", "bbox": {}, "trigger_phrases": [], "parent_id": None},
        {"id": "s2_e3", "type": "bullet", "text": "Extract", "bbox": {}, "trigger_phrases": [], "parent_id": None},
        {"id": "s2_e0", "type": "stat", "text": "< 1 ms", "bbox": {}, "trigger_phrases": [], "parent_id": None},
    ]}]}
    out = scope_index(index, expansions={})
    els = {e["id"]: e for e in out["slides"][0]["elements"]}
    assert els["s2_e2"]["group_id"] == els["s2_e3"]["group_id"]   # bullets grouped
    assert els["s2_e2"]["group_id"] == "s2_e2"
    assert els["s2_e1"]["group_id"] == "s2_e1"                    # title not in bullet group
    assert "less than one millisecond" in els["s2_e0"]["trigger_phrases"]
