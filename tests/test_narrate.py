from tools.narrate import narrate


def _el(eid, typ, text, x, y, **kw):
    return {"id": eid, "type": typ, "text": text,
            "bbox": {"x": x, "y": y, "w": 1.0, "h": 0.5}, "needs_vision": False, **kw}


def _deck(*elements, sno=1):
    return {"slides": [{"slide_number": sno, "elements": list(elements)}]}


def test_title_first_then_banded_rows_left_to_right():
    deck = _deck(
        _el("s1_e1", "body", "Right column top", 6.0, 1.50),
        _el("s1_e2", "body", "Left column top", 1.0, 1.60),   # same band (within 0.3in)
        _el("s1_e3", "body", "Left column lower", 1.0, 3.00),
        _el("s1_e0", "title", "The Title", 1.0, 0.20),
    )
    script, expected = narrate(deck)
    assert script == [{"slide": 1, "segments": [
        "The Title", "Left column top", "Right column top", "Left column lower"]}]
    assert expected == [{"slide": 1, "expected": [
        ["s1_e0"], ["s1_e2"], ["s1_e1"], ["s1_e3"]]}]


def test_allowlist_and_needs_vision_skipped():
    deck = _deck(
        _el("s1_e0", "arrow", "→", 1, 1),
        _el("s1_e1", "image", None, 1, 2, needs_vision=True),
        _el("s1_e2", "image", "Architecture diagram", 1, 3),    # alt text -> narrated
        _el("s1_e3", "chart", "PIE (5)", 1, 4),                  # container not narrated
        _el("s1_e4", "body", "Kept", 1, 5),
    )
    script, expected = narrate(deck)
    assert script[0]["segments"] == ["Architecture diagram", "Kept"]
    assert expected[0]["expected"] == [["s1_e2"], ["s1_e4"]]


def test_chart_categories_phrased_and_kept_in_index_order():
    deck = _deck(
        _el("s1_e0", "chart", "COLUMN_CLUSTERED (51)", 1, 2),
        _el("s1_e1", "chart_category", "Parse", 3.0, 2, parent_id="s1_e0", value=12.0),
        _el("s1_e2", "chart_category", "Match", 1.5, 2, parent_id="s1_e0", value=0.5),
        _el("s1_e3", "chart_category", "Q1 — Revenue", 2.0, 2, parent_id="s1_e0",
            value=40.0, series="Revenue"),
        _el("s1_e4", "chart_category", "Untitled", 2.5, 2, parent_id="s1_e0", value=None),
    )
    script, _ = narrate(deck)
    assert script[0]["segments"] == [
        "Parse is 12", "Match is 0.5", "Revenue Q1 is 40", "Untitled"]


def test_duplicates_narrated_once_with_alternates():
    deck = _deck(_el("s1_e0", "table_cell", "Live", 1, 1),
                 _el("s1_e1", "table_cell", "live ", 3, 1),
                 _el("s1_e2", "table_cell", "Offline", 5, 1))
    script, expected = narrate(deck)
    assert script[0]["segments"] == ["Live", "Offline"]
    assert expected[0]["expected"] == [["s1_e0", "s1_e1"], ["s1_e2"]]


def test_long_text_truncated_to_15_words():
    words = " ".join(f"w{i}" for i in range(30))
    script, _ = narrate(_deck(_el("s1_e0", "body", words, 1, 1)))
    assert script[0]["segments"][0].split() == [f"w{i}" for i in range(15)]


def test_slide_with_nothing_narratable_is_omitted():
    deck = {"slides": [
        {"slide_number": 1, "elements": [_el("s1_e0", "arrow", None, 1, 1)]},
        {"slide_number": 2, "elements": [_el("s2_e0", "title", "Hi", 1, 1)]},
    ]}
    script, expected = narrate(deck)
    assert [s["slide"] for s in script] == [2]
    assert [s["slide"] for s in expected] == [2]


def test_chart_values_are_speakable_digits():
    deck = _deck(_el("s1_e0", "chart", "COLUMN_CLUSTERED (51)", 1, 2),
                 _el("s1_e1", "chart_category", "Big", 1, 2, parent_id="s1_e0", value=-1234567.891),
                 _el("s1_e2", "chart_category", "Tiny", 1, 2, parent_id="s1_e0", value=0.125))
    assert narrate(deck)[0][0]["segments"] == ["Big is -1234567.89", "Tiny is 0.12"]
