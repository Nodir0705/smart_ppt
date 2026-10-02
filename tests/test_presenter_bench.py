import math

import pytest

import json

from tools.presenter_bench import (expected_steps, false_ids, false_seconds, motion_summary,
                                   p90, sentence_at, slide_motion, step_sentences, tolerated_ids)


def test_expected_steps_skip_non_pointing_and_collapse_repeats():
    segs = [{"refers_to": []}, {"refers_to": ["a"]}, {"refers_to": ["a"]},
            {"refers_to": []}, {"refers_to": ["b", "c"]}, {"refers_to": ["a"]}]
    assert expected_steps(segs) == [["a"], ["b", "c"], ["a"]]
    assert [k for _, k in step_sentences(segs)] == [1, 4, 5]


def test_slide_motion_on_hand_built_events():
    # steps: a (sentence at 1.0 s), then b|c (sentence at 4.0 s); slide is 10 s long
    steps, starts = [["a"], ["b", "c"]], [1.0, 4.0]
    events = [(1.5, "x", ""),      # stray: 1.5 -> 2.0 = 0.5 false s
              (2.0, "a", ""),      # step a lit 1.0 s late
              (3.0, None, ""),     # nothing lit: free
              (3.5, "a", ""),      # same id again after None: still a switch
              (5.0, "c", ""),      # step b|c lit 1.0 s late
              (9.0, "y", "")]      # stray to the end: 1.0 false s
    m = slide_motion(events, steps, starts, end=10.0)
    assert m["switches"] == 5
    assert m["dwells"] == pytest.approx([0.5, 1.5, 1.5, 4.0])
    assert m["false_s"] == pytest.approx(1.5)
    assert m["lateness"] == pytest.approx([1.0, 1.0])


def test_slide_motion_missed_step_and_no_events():
    m = slide_motion([(2.0, "b", "")], [["a"], ["b"]], [0.0, 1.0], end=5.0)
    assert m["lateness"] == pytest.approx([1.0])          # a never lit: only b counts
    assert m == {**m, "switches": 1, "dwells": [], "false_s": 0.0}
    assert slide_motion([], [["a"]], [0.0], 5.0) == {"switches": 0, "dwells": [],
                                                      "false_s": 0.0, "lateness": []}


def test_motion_summary_pools_slides():
    a = {"switches": 4, "dwells": [1.0, 3.0], "false_s": 0.5, "lateness": [0.2, 0.4]}
    b = {"switches": 2, "dwells": [2.0], "false_s": 1.0, "lateness": [3.0]}
    s = motion_summary([a, b], n_steps=3)
    assert s["switches_per_step"] == 2.0 and s["median_dwell"] == 2.0
    assert s["false_s"] == 1.5 and s["late_median"] == 0.4 and s["late_p90"] == 3.0
    assert math.isnan(motion_summary([], 0)["late_p90"])
    assert s["false_s_adj"] is None and s["false_adj"] is None       # no adjudication
    s = motion_summary([{**a, "false_s_adj": 0.1, "false_adj": 1},
                        {**b, "false_s_adj": 0.2, "false_adj": 0}], n_steps=3)
    assert s["false_s_adj"] == pytest.approx(0.3) and s["false_adj"] == 1


def test_adjudicated_false_counts_tolerated_as_acceptable(tmp_path):
    (tmp_path / "presenter").mkdir()
    (tmp_path / "presenter" / "adjudication.json").write_text(json.dumps({"rows": [
        {"slide": 1, "active_id": "x", "verdict": "tolerated"},
        {"slide": 1, "active_id": "y", "verdict": "wrong"},
        {"slide": 2, "active_id": "y", "verdict": "tolerated"}]}))
    tol = tolerated_ids(tmp_path)
    assert tol == {1: {"x"}, 2: {"y"}}
    events = [(1.0, "x", ""), (2.0, "a", ""), (3.0, "y", ""), (4.0, None, ""), (6.0, "y", "")]
    strict, adj = {"a"}, {"a"} | tol[1]
    assert false_ids(events, strict) == 2 and false_ids(events, adj) == 1
    assert false_seconds(events, strict, end=8.0) == pytest.approx(1.0 + 1.0 + 2.0)
    assert false_seconds(events, adj, end=8.0) == pytest.approx(3.0)   # only y (wrong) counts
    assert tolerated_ids(tmp_path / "nowhere") is None


def test_p90_and_sentence_at():
    assert p90(list(range(1, 11))) == 9 and p90([5.0]) == 5.0
    assert [sentence_at([0.0, 2.0, 5.0], t) for t in (0.0, 1.9, 2.0, 7.0)] == [0, 0, 1, 2]
