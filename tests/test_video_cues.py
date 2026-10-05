"""Cue sheet schema + validator (tools/video/cues.py) on a synthetic slide."""
import copy
import math

from tools.video.cues import content, effective_until, resolve, stem, validate


def box(x, y, w, h):
    return {"x": x, "y": y, "w": w, "h": h, "units": "inches"}


L0, L1 = box(1.0, 3.0, 2.0, 0.2), box(1.0, 3.3, 2.4, 0.2)

SLIDE = {
    "slide_number": 1,
    "elements": [
        {"id": "e0", "type": "title", "text": "Grading pipeline overview", "bbox": box(0.7, 0.8, 6.0, 0.4),
         "line_bboxes": [box(0.7, 0.8, 6.0, 0.4)], "line_texts": ["Grading pipeline overview"],
         "pointable": True, "map_unit": "u1"},
        {"id": "e1", "type": "picture", "text": None, "bbox": box(1.0, 2.0, 0.5, 0.5),
         "pointable": True, "map_unit": "u2"},
        {"id": "e2", "type": "bullet", "text": "Turns pixels into text catches hand drawn circles",
         "bbox": box(1.0, 3.0, 2.4, 0.5), "line_bboxes": [L0, L1],
         "line_texts": ["Turns pixels into text", "catches hand drawn circles"],
         "pointable": True, "map_unit": "u2"},
        {"id": "e3", "type": "body", "text": "8 GB GPU memory", "bbox": box(5.0, 3.0, 2.0, 0.3), "value": "8 GB",
         "line_bboxes": [box(5.0, 3.0, 2.0, 0.3)], "line_texts": ["8 GB GPU memory"],
         "pointable": True, "map_unit": "u3"},
        {"id": "e4", "type": "caption", "text": "Page 1", "bbox": box(12.0, 7.0, 0.5, 0.1),
         "line_bboxes": [box(12.0, 7.0, 0.5, 0.1)], "line_texts": ["Page 1"], "pointable": False},
        {"id": "e5", "type": "arrow", "text": None, "bbox": box(3.5, 3.0, 1.0, 0.1), "pointable": False},
    ],
    "map": {
        "message": "A pipeline that reads, then fits in eight gigabytes.",
        "units": {"u1": ["e0"], "u2": ["e1", "e2"], "u3": ["e3"]},
        "info": {
            "u1": {"summary": "Title.", "talk": ["how it works"], "terms": ["pipeline"]},
            "u2": {"summary": "Stage A reads.", "talk": ["first it reads the page"], "terms": ["stage a", "ocr"]},
            "u3": {"summary": "Memory.", "talk": ["it runs on a modest gpu"], "terms": ["eight gigabytes"]},
        },
        "relations": [{"from": "u2", "to": "u3", "type": "flow"}],
        "order": ["u1", "u2", "u3"],
    },
}

SENTENCES = [s.split() for s in [
    "So here is how the grading pipeline works.",
    "Stage one turns pixels into text, so the printed stuff — all in one go.",
    "It even catches the circles kids draw by hand.",
    "Next, everything has to squeeze onto cheap hardware.",
    "That takes eight gigabytes of memory.",
]]


def at(s, w):
    return {"s": s, "w": w}


VALID = {"slide": 1, "version": 1, "cues": [
    {"effect": "focus", "target": "u1", "at": at(0, 5), "until": at(4, 0)},   # replaced by the next focus
    {"effect": "focus", "target": "u2", "at": at(1, 0), "spotlight": True},
    {"effect": "underline", "target": "e2", "at": at(2, 2)},
    {"effect": "zoom", "target": "e2", "at": at(2, 4), "until": at(2, 8), "region": [0, 0.5, 1, 0.5]},
    {"effect": "glide", "target": "u3", "at": at(3, 0)},                      # no shared word: flow + "Next"
    {"effect": "ring", "target": "e3", "at": at(4, 2)},
]}

# word w of sentence s starts at 10*s + w seconds
TIMINGS = [[{"word": w, "start": 10.0 * s + i, "end": 10.0 * s + i + 0.5} for i, w in enumerate(ws)]
           for s, ws in enumerate(SENTENCES)]


def sheet_with(edit):
    sheet = copy.deepcopy(VALID)
    edit(sheet["cues"])
    return sheet


def errors_for(edit, timings=None):
    return validate(sheet_with(edit), SLIDE, SENTENCES, timings)


def has(errors, cue, word):
    assert any(e.startswith(f"cue {cue}:") and word in e for e in errors), errors
    return True


# ---------- stemming ----------

def test_stem_and_content_words():
    assert stem("reads") == stem("reading") == "read"
    assert stem("marked") == "mark" and stem("boxes") == "box"
    assert stem("pass") == "pass"                         # len <= 4: untouched
    assert content("The marks, and reading!") == {"mark", "read"}


# ---------- validate ----------

def test_valid_sheet():
    assert validate(VALID, SLIDE, SENTENCES) == []
    assert validate(VALID, SLIDE, SENTENCES, TIMINGS) == []


def test_unknown_or_unpointable_target():
    assert has(errors_for(lambda c: c[5].update(target="u9")), 5, "unknown target")
    assert has(errors_for(lambda c: c[5].update(target="e4")), 5, "not pointable")


def test_bad_line_index():
    assert has(errors_for(lambda c: c[2].update(line=2)), 2, "line")
    assert errors_for(lambda c: c[2].update(line=1)) == []


def test_positions_out_of_range_and_until_order():
    assert has(errors_for(lambda c: c[5].update(at=at(5, 0))), 5, "range")
    assert has(errors_for(lambda c: c[5].update(at=at(4, 6))), 5, "range")
    assert has(errors_for(lambda c: c[3].update(until=at(2, 4))), 3, "until")   # == at
    assert has(errors_for(lambda c: c[3].update(until=at(2, 1))), 3, "until")   # before at


def test_second_focus_replaces_the_first():
    # cue 0 says until (4,0) but cue 1 is a new focus -> allowed, not a conflict
    assert validate(VALID, SLIDE, SENTENCES, TIMINGS) == []


def test_third_mark_at_once_is_an_error():
    # glides do not end marks: ring e1 + underline e2 + ring e3 all live at (4,2)
    errs = errors_for(lambda c: c.append({"effect": "ring", "target": "e1", "at": at(1, 1)}))
    assert has(errs, 5, "concurrency")


def test_overlapping_zoom_is_an_error():
    errs = errors_for(lambda c: c.append({"effect": "zoom", "target": "e2", "at": at(2, 6), "until": at(2, 8)}))
    assert has(errs, 6, "zoom")
    # a zoom after the first one ends is fine
    assert errors_for(lambda c: c.append({"effect": "zoom", "target": "e3", "at": at(4, 3)})) == []


def test_spotlight_only_on_focus():
    assert has(errors_for(lambda c: c[2].update(spotlight=True)), 2, "spotlight")


def test_glide_needs_a_current_focus():
    sheet = {"cues": [{"effect": "glide", "target": "u3", "at": at(4, 0)}]}
    assert has(validate(sheet, SLIDE, SENTENCES), 0, "glide")


def test_why_rule():
    # sentence 0 does not talk about the memory stat
    assert has(errors_for(lambda c: c[5].update(at=at(0, 1))), 5, "why")
    # "Next" + no shared word, but u1 is not a flow successor of u2
    assert has(errors_for(lambda c: c[4].update(target="u1")), 4, "why")
    # flow successor of the focus, but no transition word in the sentence
    sheet = {"cues": [{"effect": "focus", "target": "u2", "at": at(1, 0)},
                      {"effect": "glide", "target": "u3", "at": at(1, 3)}]}
    assert has(validate(sheet, SLIDE, SENTENCES), 1, "why")


def test_malformed_cues_are_errors_not_crashes():
    sheet = {"cues": [{"effect": "focus", "at": at(0, 0)},
                      {"effect": "ring", "target": "e3", "at": {"s": "4", "w": 2}},
                      {"effect": "sparkle", "target": "e3", "at": at(4, 2)},
                      "focus u1",
                      {"effect": "ring", "target": ["e3"], "at": at(4, 2)}]}
    errs = validate(sheet, SLIDE, SENTENCES)
    assert [e.split(":")[0] for e in errs] == [f"cue {i}" for i in range(5)]
    assert validate({"cues": "nope"}, SLIDE, SENTENCES) and validate([], SLIDE, SENTENCES)


# ---------- resolve ----------

def test_resolve_unit_and_element():
    r = resolve({"effect": "focus", "target": "u2", "at": at(1, 0)}, SLIDE, SENTENCES)
    assert r["ids"] == ["e1", "e2"]
    assert r["bbox"] == (1.0, 2.0, 2.4, 1.5)          # union of e1 and e2
    assert r["line"] is None
    r = resolve({"effect": "ring", "target": "e2", "at": at(2, 0)}, SLIDE, SENTENCES)
    assert r["ids"] == ["e2"] and r["bbox"] == (1.0, 3.0, 2.4, 0.5)


def test_resolve_default_underline_line():
    def line(s, sentences=SENTENCES, **kw):
        return resolve({"effect": "underline", "target": "e2", "at": at(s, 0), **kw}, SLIDE, sentences)

    assert line(2)["line"] == 1 and line(2)["line_bbox"] == (1.0, 3.3, 2.4, 0.2)   # catches/hand/circles
    assert line(1)["line"] == 0                                                   # turns/pixels/text
    assert line(4)["line"] == 0                                                   # no overlap -> first
    assert line(0, [["text", "hand"]])["line"] == 0                               # tie -> first
    assert line(1, line=1)["line"] == 1                                           # explicit wins


# ---------- effective_until ----------

def test_effective_until_defaults():
    iv = effective_until(VALID, TIMINGS)
    assert iv == [(5.0, 10.0),        # focus clipped by the next focus despite until (4,0)
                  (10.0, 30.0),       # focus (and its spotlight) ends at the glide
                  (22.0, math.inf),   # underline: no later focus
                  (24.0, 28.0),       # zoom with explicit until
                  (30.0, math.inf),   # glide holds to the end
                  (42.0, math.inf)]   # ring


def test_marks_end_at_next_focus_not_glide_and_zoom_lasts_3s():
    sheet = {"cues": [{"effect": "focus", "target": "u2", "at": at(1, 0)},
                      {"effect": "underline", "target": "e2", "at": at(2, 2)},
                      {"effect": "glide", "target": "u3", "at": at(3, 0)},
                      {"effect": "focus", "target": "u3", "at": at(4, 0)},
                      {"effect": "zoom", "target": "e3", "at": at(4, 2)}]}
    assert effective_until(sheet, TIMINGS) == [(10.0, 30.0), (22.0, 40.0), (30.0, 40.0),
                                               (40.0, math.inf), (42.0, 45.0)]
