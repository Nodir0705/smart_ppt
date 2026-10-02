"""Slide understanding maps: index-build application, live units/furniture/flow,
and the bench tooling around them. Synthetic indexes and maps only."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.matcher import (ElementRef, MatchPolicy, PolicyConfig, flow_successors, furniture_ids,
                         pointable_units)
from src.slide_map import apply_slide_maps


def _el(eid, text, x, y, type="body"):
    return {"id": eid, "type": type, "text": text, "trigger_phrases": [text],
            "bbox": {"x": x, "y": y, "w": 2.0, "h": 0.4}}


def _index():
    """Slide 1: title + two cards (head + body each) + a logo. Slide 2: same
    layout, never mapped (heuristic units)."""
    def slide(n):
        return {"slide_number": n, "elements": [
            _el(f"s{n}_body1", "Reads the page with a model", 1.0, 2.5),   # listed before its head
            _el(f"s{n}_t", "Pipeline", 1.0, 0.5, type="title"),
            _el(f"s{n}_head1", "Stage A", 1.0, 2.0),
            _el(f"s{n}_head2", "Stage B", 5.0, 2.0),
            _el(f"s{n}_body2", "Checks every claim against sources", 5.0, 2.5),
            _el(f"s{n}_logo", "Acme", 11.0, 0.3)]}
    return {"slide_size": {"w": 13.33, "h": 7.5}, "slides": [slide(1), slide(2)]}


def _map(slide=1):
    p = f"s{slide}_"
    return {"slide": slide, "message": "Two stages turn a page into checked claims.",
            "items": {p + "t": {"role": "title", "pointable": True},
                      p + "head1": {"role": "card_heading", "pointable": True},
                      p + "body1": {"role": "body", "pointable": True},
                      p + "head2": {"role": "card_heading", "pointable": True},
                      p + "body2": {"role": "body", "pointable": True},
                      p + "logo": {"role": "decoration", "pointable": False}},
            "units": [{"uid": "u0", "members": [p + "t"], "summary": "title",
                       "talk": ["the pipeline", "how it works", "Pipeline"]},
                      {"uid": "u1", "members": [p + "body1", p + "head1"], "summary": "reading",
                       "talk": ["the reading stage", "Stage A", "how we  turn pixels into text"],
                       "terms": ["stage a", "O C R"]},
                      {"uid": "u2", "members": [p + "head2", p + "body2"], "summary": "checking",
                       "talk": ["the checking stage", "fact checking", "stage b"]}],
            "relations": [{"from": "u1", "to": "u2", "type": "flow"},
                          {"from": "u0", "to": "u1", "type": "explains"}],
            "order": ["u0", "u1", "u2"]}


def _write_map(d: Path, m: dict) -> None:
    d.mkdir(exist_ok=True)
    (d / f"slide_{m['slide']:03d}.json").write_text(json.dumps(m))


def test_apply_sets_roles_units_and_anchor_phrases(tmp_path):
    _write_map(tmp_path, _map())
    ix = apply_slide_maps(_index(), tmp_path)
    els = {e["id"]: e for e in ix["slides"][0]["elements"]}
    assert els["s1_logo"]["pointable"] is False and els["s1_logo"]["map_role"] == "decoration"
    assert "map_unit" not in els["s1_logo"]
    assert els["s1_body1"]["map_unit"] == els["s1_head1"]["map_unit"] == "u1"
    # anchor = first member in reading order (head1 sits above body1); map phrases
    # (talk, terms, summary) kept apart from the slide's own trigger phrases
    assert els["s1_head1"]["trigger_phrases"] == ["Stage A"]
    assert els["s1_head1"]["map_phrases"] == [
        "the reading stage", "how we turn pixels into text", "o c r", "reading"]
    assert "map_phrases" not in els["s1_body1"]
    # a title unit's talk is slide-intro speech: summary + terms only
    assert els["s1_t"]["map_phrases"] == ["title"]
    assert ix["slides"][0]["map"] == {
        "message": "Two stages turn a page into checked claims.",
        "relations": _map()["relations"], "order": ["u0", "u1", "u2"],
        "units": {"u0": ["s1_t"], "u1": ["s1_head1", "s1_body1"], "u2": ["s1_head2", "s1_body2"]}}
    assert "map" not in ix["slides"][1]                       # no map file -> untouched
    assert ix["slides"][1] == _index()["slides"][1]


def test_invalid_map_is_skipped_and_missing_dir_is_noop(tmp_path):
    bad = _map()
    del bad["items"]["s1_logo"]                               # items must cover every id
    _write_map(tmp_path, bad)
    assert apply_slide_maps(_index(), tmp_path) == _index()
    assert apply_slide_maps(_index(), tmp_path / "nowhere") == _index()


def test_units_from_map_on_mapped_slides_heuristic_elsewhere(tmp_path):
    _write_map(tmp_path, _map())
    ix = apply_slide_maps(_index(), tmp_path)
    units = pointable_units(ix)
    assert units["s1_body1"] == units["s1_head1"] == "s1_head1"
    assert units["s1_body2"] == units["s1_head2"] == "s1_head2"
    assert "s1_logo" not in units                             # non-pointable: in no unit
    heuristic = pointable_units(_index())
    assert {k: v for k, v in units.items() if k.startswith("s2_")} == \
        {k: v for k, v in heuristic.items() if k.startswith("s2_")}
    assert flow_successors(ix) == {"s1_head1": {"s1_head2"}}
    assert flow_successors(_index()) == {}


def test_furniture_includes_map_unpointable(tmp_path):
    _write_map(tmp_path, _map())
    ix = apply_slide_maps(_index(), tmp_path)
    assert "s1_logo" in furniture_ids(ix) and "s2_logo" not in furniture_ids(ix)
    assert furniture_ids({"slides": ix["slides"]}) == {"s1_logo"}   # no slide size still counts


def _ref(eid):
    return ElementRef(id=eid, type="x", text=eid, slide_no=1, phrases=[eid], emb_start=0, emb_end=1)


class FlowStub:
    def __init__(self, seq, flow_next):
        self._seq, self._i = list(seq), 0
        self.index = SimpleNamespace(flow_next=flow_next)

    def scored(self, spoken, slide_no):
        self._i += 1
        return self._seq[self._i - 1]


@pytest.mark.parametrize("flow_bonus,expected", [(0.03, "B"), (0.0, "C")])
def test_flow_bonus_lifts_the_flow_successor(flow_bonus, expected):
    A, B, C = _ref("A"), _ref("B"), _ref("C")
    # step 2: C leads B raw by 0.01; B follows A by flow -> +0.03 wins B the gap
    # on effective score (gap 0.0 so the tie-break itself is what's measured)
    seq = [[(0.80, A), (0.40, B), (0.40, C)], [(0.71, C), (0.70, B), (0.30, A)]]
    pol = MatchPolicy(FlowStub(seq, {"A": {"B"}}),
                      PolicyConfig(threshold=0.55, gap=0.0, stickiness=1, short_words=0,
                                   null_margin=None, stale_release=0, flow_bonus=flow_bonus))
    assert pol.step("x", 1)[1:] == ("A", "switch")
    res, active, why = pol.step("x", 1)
    assert (active, why) == (expected, "switch")
    if flow_bonus:
        assert res._eff_best == pytest.approx(0.73)


def test_presenter_out_dir_from_script_name():
    from tools.presenter_bench import out_dir_name
    assert out_dir_name("presenter_script.json") == "presenter"
    assert out_dir_name("presenter_script_jumpy.json") == "presenter_jumpy"
    assert out_dir_name("presenter_script_meaning.json") == "presenter_meaning"
    with pytest.raises(ValueError):
        out_dir_name("talk.json")


def test_map_variant_resolves_map_dir_per_deck():
    from tools.deck_bench import ROOT, variant_kwargs
    assert variant_kwargs("map", Path("decks/AI_Handout_Checker.pptx")) == {
        "slide_map_dir": ROOT / "out" / "maps" / "AI_Handout_Checker"}
    assert variant_kwargs("no-ocr", Path("decks/x.pptx")) == {"ocr": False}


def test_index_loads_map_units_flow_and_unpointables(tmp_path):
    import numpy as np
    from src.matcher import Index

    class Enc:
        dim, model_name = 4, "stub"
        def encode(self, texts):
            return np.ones((len(texts), 4), dtype=np.float32)

    _write_map(tmp_path, _map())
    p = tmp_path / "index.json"
    p.write_text(json.dumps(apply_slide_maps(_index(), tmp_path)))
    ix = Index(p, Enc())
    refs = {r.id: r for n in (1, 2) for r in ix.elements_on(n)}
    assert not refs["s1_logo"].is_pointable and refs["s2_logo"].is_pointable
    assert ix.flow_next == {"s1_head1": {"s1_head2"}}
    box = ix.unit_bbox("s1_head1")                            # union of the map unit's members
    assert box.pop("units") == "inches"
    assert box == pytest.approx({"x": 1.0, "y": 2.0, "w": 2.0, "h": 0.9})


def test_index_keeps_only_specific_map_phrases(tmp_path):
    """A map phrase joins its anchor's span only when it is no closer to generic
    talk (NULL_SPEECH) than the anchor's own text, and closer to its own unit's
    text than to any other unit's."""
    import numpy as np
    from src.matcher import NULL_SPEECH, Index

    vec = {"Stage A": [1, 0, 0.3, 0], "Stage B": [0, 1, 0, 0],   # null axis = dim 2
           "the reading stage": [0.9, 0.1, 0.25, 0],    # specific -> kept
           "how we turn pixels into text": [0.3, 0, 0.95, 0],   # generic (near null) -> dropped
           "o c r": [0.2, 0.9, 0, 0]}                   # closer to Stage B -> dropped

    class Enc:
        dim, model_name = 4, "stub"
        def encode(self, texts):
            v = np.array([vec.get(t, [0, 0, 1, 0] if t in NULL_SPEECH else [0, 0, 0.1, 0.1])
                          for t in texts], dtype=np.float32)
            return v / np.linalg.norm(v, axis=1, keepdims=True)

    _write_map(tmp_path, _map())
    ix = apply_slide_maps(_index(), tmp_path)
    p = tmp_path / "index.json"
    p.write_text(json.dumps(ix))
    refs = {r.id: r for r in Index(p, Enc()).elements_on(1)}
    assert refs["s1_head1"].phrases == ["Stage A", "the reading stage"]
    assert refs["s1_body1"].phrases == ["Reads the page with a model"]
    spans = sorted((r.emb_start, r.emb_end) for r in refs.values())
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))   # contiguous re-lay
