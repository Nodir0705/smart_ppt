"""Pointable units (cards): grouping rules on synthetic slides, the policy
switching between units, and the controller drawing the unit's union box."""
import json
from types import SimpleNamespace

from src.live_controller import LiveController
from src.matcher import ElementRef, Index, MatchPolicy, PolicyConfig, pointable_units
from src.stt import Partial

W, H = 13.333, 7.5


def el(eid, text, x, y, w, h, type="body", shape=None, lines=None, phrases=True):
    e = {"id": eid, "type": type, "text": text,
         "bbox": {"x": x, "y": y, "w": w, "h": h, "units": "inches"},
         "trigger_phrases": [text or eid] if phrases else []}
    if shape:
        e["shape_bbox"] = dict(zip("xywh", shape))
    if lines:
        e["line_texts"] = [t for t, _ in lines]
        e["line_bboxes"] = [b for _, b in lines]
    return e


def units_of(*elements):
    return pointable_units({"slide_size": {"w": W, "h": H},
                            "slides": [{"slide_number": 1, "elements": list(elements)}]})


def groups(uo):
    out = {}
    for i, u in uo.items():
        out.setdefault(u, set()).add(i)
    return sorted(sorted(g) for g in out.values() if len(g) > 1)


# ---------- grouping rules ----------

def test_heading_and_its_description_are_one_unit():
    uo = units_of(el("h", "Slow & repetitive", 1.0, 2.03, 1.7, 0.20),
                  el("d", "A teacher marks the same worksheet 30 times in one evening.", 1.0, 2.40, 5.6, 0.16))
    assert uo == {"h": "h", "d": "h"}


def test_big_number_and_its_caption_are_one_unit():
    uo = units_of(el("n", "15–33", 8.0, 2.96, 1.1, 0.31),
                  el("c", "gradable items per page", 8.0, 3.47, 1.7, 0.15))
    assert groups(uo) == [["c", "n"]]


def test_heading_takes_its_bullets_until_a_short_heading_starts_a_new_card():
    uo = units_of(el("h", "HIDL", 1.9, 4.11, 0.5, 0.15),
                  el("b1", "Hardware Interface Definition Language", 1.9, 4.39, 3.7, 0.16),
                  el("b2", "Facilitates communication via Binder IPC", 1.9, 4.64, 3.7, 0.13),
                  el("m", "Memory", 1.9, 4.85, 0.7, 0.15),
                  el("md", "LPDDR4x, eMMC, QSPI", 1.9, 5.08, 1.4, 0.16))
    assert groups(uo) == [["b1", "b2", "h"], ["m", "md"]]


def test_picture_and_its_caption_are_one_unit():
    uo = units_of(el("img", "recognizing tenses read the text", 7.78, 1.9, 4.45, 4.6, type="image"),
                  el("cap", "The model reads handwriting in place", 8.28, 6.6, 3.44, 0.14))
    assert groups(uo) == [["cap", "img"]]


def test_card_label_joins_what_its_panel_holds():
    uo = units_of(el("lab", "STAGE A · READ", 1.56, 2.58, 1.33, 0.13, shape=(0.72, 2.35, 3.0, 2.7)),
                  el("hd", "Turn pixels into text", 0.92, 3.11, 1.7, 0.18),
                  el("b1", "• A vision-language model reads the whole page", 0.93, 3.55, 2.48, 0.30),
                  el("b2", "• A detector finds hand-drawn circles the model misses", 0.93, 3.95, 2.35, 0.30))
    assert groups(uo) == [["b1", "b2", "hd", "lab"]]


def test_a_board_of_several_cards_is_not_one_unit():
    uo = units_of(el("lab", "A single handout", 8.0, 2.31, 1.45, 0.18, shape=(7.7, 1.95, 4.9, 4.4)),
                  el("n1", "15–33", 8.0, 2.96, 1.1, 0.31), el("c1", "gradable items per page", 8.0, 3.47, 1.7, 0.15),
                  el("n2", "~1 min", 8.0, 5.24, 1.3, 0.33), el("c2", "to hand-mark every copy", 8.0, 5.77, 1.9, 0.15))
    assert groups(uo) == [["c1", "n1"], ["c2", "n2"]]
    assert uo["lab"] == "lab"


# ---------- never merged ----------

def test_chart_categories_and_table_cells_stay_points():
    uo = units_of(el("cat1", "STT", 1.0, 2.0, 0.5, 0.15, type="chart_category"),
                  el("cat2", "Speech to text engine latency", 1.0, 2.2, 3.0, 0.15, type="chart_category"),
                  el("c1", "Stage", 5.0, 2.0, 0.6, 0.15, type="table_cell"),
                  el("c2", "Speech recognition of the live audio", 5.0, 2.2, 3.0, 0.15, type="table_cell"))
    assert groups(uo) == []


def test_title_never_joins_body():
    uo = units_of(el("t", "Results", 0.7, 0.84, 2.0, 0.36, type="title"),
                  el("b", "Both scores are on genuine phone scans of real handouts", 0.7, 1.25, 6.0, 0.17))
    assert groups(uo) == []


def test_far_apart_or_other_column_stays_separate():
    uo = units_of(el("h", "Privacy matters", 1.0, 2.0, 1.5, 0.20),
                  el("far", "Student work cannot be uploaded to a cloud API at all", 1.0, 2.8, 5.0, 0.16),
                  el("h2", "Runs offline", 7.0, 4.0, 1.5, 0.20),
                  el("side", "One eight gigabyte gaming card is enough for both models", 1.0, 4.25, 5.0, 0.16))
    assert groups(uo) == []


def test_column_of_similar_bullets_is_not_merged():
    rows = ["Point it at a folder of scanned handouts", "Each AI model loads once and stays warm",
            "Filename becomes the student name in the sheet", "One corrupt file never stops the batch"]
    uo = units_of(*[el(f"b{k}", t, 1.0, 3.0 + 0.3 * k, 3.5, 0.16) for k, t in enumerate(rows)])
    assert groups(uo) == []


def test_unpointable_icons_and_furniture_are_not_unit_members():
    uo = units_of(el("icon", None, 1.0, 1.6, 0.3, 0.3, type="image", phrases=False),
                  el("cap", "Audio", 1.0, 2.0, 0.5, 0.15))
    assert "icon" not in uo and uo["cap"] == "cap"


def test_panel_holding_a_table_is_not_a_card():
    uo = units_of(el("lab", "Specs", 1.0, 1.1, 0.8, 0.15, shape=(0.8, 1.0, 5.0, 3.0)),
                  el("c1", "Interface", 1.0, 2.0, 0.8, 0.15, type="table_cell"),
                  el("b", "Supports up to eighty keys in a matrix", 1.0, 3.0, 3.0, 0.15))
    assert groups(uo) == []


# ---------- policy + controller ----------

def _unit_index(refs, unit_of, bbox=None):
    return SimpleNamespace(unit_of=unit_of, ref_by_id={r.id: r for r in refs},
                           unit_bbox=bbox or (lambda u: None))


class _Stub:
    def __init__(self, seq, index):
        self.seq, self.index, self.k = seq, index, 0

    def scored(self, spoken, slide_no):
        out = self.seq[min(self.k, len(self.seq) - 1)]
        self.k += 1
        return out


def _ref(i, group=None):
    return ElementRef(id=i, type="body", text=i, slide_no=1, phrases=[i], emb_start=0,
                      emb_end=1, group_id=group)


def test_talking_through_one_card_keeps_one_unit_lit():
    h, d, o = _ref("h"), _ref("d"), _ref("o")
    idx = _unit_index([h, d, o], {"h": "h", "d": "h", "o": "o"})
    seq = [[(0.80, h), (0.60, o)], [(0.82, d), (0.70, h), (0.60, o)],
           [(0.81, h), (0.79, d), (0.60, o)], [(0.85, d), (0.60, o)]]
    pol = MatchPolicy(_Stub(seq, idx), PolicyConfig(short_words=0, null_margin=None))
    assert pol.step("x", 1)[1:] == ("h", "switch")
    for _ in range(3):           # winner alternates heading/description: same unit
        assert pol.step("x", 1)[1:] == ("h", "hold")


def test_card_members_do_not_compete_on_gap():
    # heading and description within 0.01 of each other: as elements the gap
    # check would refuse both; as one unit the card is a confident winner.
    h, d, o = _ref("h"), _ref("d"), _ref("o")
    idx = _unit_index([h, d, o], {"h": "h", "d": "h", "o": "o"})
    pol = MatchPolicy(_Stub([[(0.80, d), (0.79, h), (0.60, o)]], idx),
                      PolicyConfig(short_words=0, null_margin=None))
    assert pol.step("x", 1)[1:] == ("h", "switch")


def test_unit_takes_the_winning_members_null_floor():
    h, d = _ref("h"), _ref("d")
    stub = _Stub([[(0.74, d)]], _unit_index([h, d], {"h": "h", "d": "h"}))
    stub.null_floor = {"h": 0.70, "d": 0.50}       # head alone would need 0.82
    pol = MatchPolicy(stub, PolicyConfig(short_words=0, null_margin=0.12))
    assert pol.step("x", 1)[1:] == ("h", "switch")


def test_controller_draws_the_unit_box_not_a_line(tmp_path, fake_encoder):
    line = lambda y: {"x": 1.0, "y": y, "w": 5.0, "h": 0.16, "units": "inches"}  # noqa: E731
    raw = {"slide_size": {"w": W, "h": H}, "slides": [{"slide_number": 1, "elements": [
        el("h", "Scans are messy", 1.0, 3.16, 1.6, 0.19),
        el("d", "Handouts come back as phone photos with skew and handwriting", 1.0, 3.52, 5.9, 0.37,
           lines=[("Handouts come back as phone photos", line(3.52)),
                  ("with skew and handwriting", line(3.73))])]}]}
    p = tmp_path / "index.json"
    p.write_text(json.dumps(raw))
    index = Index(p, fake_encoder)
    assert index.unit_of == {"h": "h", "d": "h"}
    stub = _Stub([[(0.9, index.ref_by_id["d"])]], index)
    c = LiveController(stub, raw, normalizer=lambda t: t)
    c.push(Partial("with skew and handwriting", is_final=True, timestamp=1.0))
    ev = c.feed()
    assert ev.active_id == "h" and ev.line_bbox is None
    assert ev.bbox is ev.unit_bbox
    assert ev.unit_bbox["y"] == 3.16 and abs(ev.unit_bbox["y"] + ev.unit_bbox["h"] - 3.89) < 1e-9
    assert index.unit_bbox("d") is None and index.unit_bbox("nope") is None   # only unit ids
