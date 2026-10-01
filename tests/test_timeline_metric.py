"""Unit tests for the grounding metric (`score_slide`) — pure, no model/audio.

Covers ordered-recall AND the new precision side (false-highlight counting), so
the metric's logic is guarded in the fast suite rather than only via the slow,
opt-in end-to-end eval.
"""

from __future__ import annotations

from tools.timeline_audit import score_slide


def test_perfect_trajectory():
    hits, total, missed, relevant, false = score_slide([["a"], ["b"]], ["a", "b"])
    assert (hits, total, missed, relevant, false) == (2, 2, [], 2, 0)


def test_false_highlight_is_counted():
    # 'x' is never an acceptable id — recall still 1/1, but precision sees it
    hits, total, missed, relevant, false = score_slide([["a"]], ["a", "x"])
    assert hits == 1 and relevant == 1 and false == 1


def test_alternates_accepted_without_false():
    hits, total, missed, relevant, false = score_slide([["a", "b"]], ["b"])
    assert hits == 1 and relevant == 1 and false == 0


def test_out_of_order_hurts_recall_not_precision():
    # both are narration-relevant, so 0 false; but ordered recall only gets 1
    hits, total, missed, relevant, false = score_slide([["a"], ["b"]], ["b", "a"])
    assert hits == 1 and total == 2 and relevant == 2 and false == 0


def test_consecutive_dedup_and_none_ignored():
    hits, total, missed, relevant, false = score_slide(
        [["a"], ["b"]], [None, "a", "a", None, "b"]
    )
    assert (hits, relevant, false) == (2, 2, 0)


def test_repeated_false_id_counts_once():
    # a non-consecutive repeat of a false id is one distinct false highlight
    hits, total, missed, relevant, false = score_slide([["a"]], ["a", "x", "a", "x"])
    assert hits == 1 and relevant == 1 and false == 1


def test_silence_is_not_a_false_highlight():
    hits, total, missed, relevant, false = score_slide([["a"]], [None, None])
    assert hits == 0 and relevant == 0 and false == 0


def test_eval_items_skips_slides_missing_from_fixture():
    from tools.timeline_audit import _eval_items
    manifest = [{"slide": 1, "wav": "a.wav"}, {"slide": 2, "wav": "b.wav"},
                {"slide": 3, "wav": "c.wav"}]
    gt = {1: [["s1_e0"]], 3: [["s3_e0"]]}
    items, skipped = _eval_items(manifest, gt, only=None)
    assert [i["slide"] for i in items] == [1, 3]
    assert skipped == [2]
    items, skipped = _eval_items(manifest, gt, only=3)
    assert [i["slide"] for i in items] == [3] and skipped == []
