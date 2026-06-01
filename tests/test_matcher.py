"""Stage 3 matcher tests.

Two layers:

1. `TestPolicy*` — deterministic unit tests of MatchPolicy using a stub matcher.
   Fast, no embedding model required, covers all state-machine branches.

2. `TestSemanticAcceptance` — end-to-end against the real index + BGE encoder.
   A small set of paraphrase probes that must resolve to the right element.
   Marked with `slow` because it loads the embedding model.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from src.matcher import MatchResult, MatchPolicy, PolicyConfig  # noqa: E402


# ---------- stub matcher for policy tests ----------

class StubMatcher:
    """Returns a scripted sequence of (id, score, runner_up_score) per step."""

    def __init__(self, scripted: list[tuple[str | None, float, float]]) -> None:
        self._scripted = list(scripted)
        self._i = 0

    def match(self, spoken: str, slide_no: int) -> MatchResult:
        eid, s, ru = self._scripted[self._i]
        self._i += 1
        return MatchResult(
            element_id=eid,
            type="x",
            text=eid,
            score=s,
            runner_up_score=ru,
            runner_up_id=None,
            slide_no=slide_no,
        )


# ---------- policy unit tests ----------

class TestPolicy:
    def _policy(self, scripted, **cfg):
        return MatchPolicy(StubMatcher(scripted), PolicyConfig(**cfg))

    def test_single_confident_match_switches(self):
        # one frame, confident, stickiness=1 → switch immediately
        pol = self._policy(
            [("A", 0.8, 0.5)],
            threshold=0.55, gap=0.03, stickiness=1, clear_after_n_low=4,
        )
        res, active, why = pol.step("x", 1)
        assert why == "switch"
        assert active == "A"
        assert pol.active_id == "A"

    def test_stickiness_two_holds_first_frame(self):
        pol = self._policy(
            [("A", 0.8, 0.5), ("A", 0.8, 0.5)],
            threshold=0.55, gap=0.03, stickiness=2, clear_after_n_low=4,
        )
        _, active, why = pol.step("x", 1)
        assert why == "hold"
        assert active is None
        _, active, why = pol.step("x", 1)
        assert why == "switch"
        assert active == "A"

    def test_ambiguous_gap_blocks_emission(self):
        # best 0.7 vs runner-up 0.69 → gap 0.01 < 0.03 → low-conf, no switch
        pol = self._policy(
            [("A", 0.70, 0.69)],
            threshold=0.55, gap=0.03, stickiness=1, clear_after_n_low=4,
        )
        _, active, why = pol.step("x", 1)
        assert why == "noop"  # active was None already
        assert active is None

    def test_below_threshold_blocks(self):
        pol = self._policy(
            [("A", 0.40, 0.10)],
            threshold=0.55, gap=0.03, stickiness=1, clear_after_n_low=4,
        )
        _, active, why = pol.step("x", 1)
        assert why == "noop"
        assert active is None

    def test_hold_active_through_low_confidence(self):
        # first set active=A, then several low-conf frames; should hold for clear_after_n_low-1, then clear
        pol = self._policy(
            [
                ("A", 0.8, 0.5),     # switch to A
                ("A", 0.4, 0.3),     # low-conf 1 → hold
                ("A", 0.4, 0.3),     # low-conf 2 → hold
                ("A", 0.4, 0.3),     # low-conf 3 → hold
                ("A", 0.4, 0.3),     # low-conf 4 → CLEAR
            ],
            threshold=0.55, gap=0.03, stickiness=1, clear_after_n_low=4,
        )
        _, a, why = pol.step("x", 1); assert why == "switch" and a == "A"
        _, a, why = pol.step("x", 1); assert why == "hold"  and a == "A"
        _, a, why = pol.step("x", 1); assert why == "hold"  and a == "A"
        _, a, why = pol.step("x", 1); assert why == "hold"  and a == "A"
        _, a, why = pol.step("x", 1); assert why == "clear" and a is None

    def test_low_conf_streak_resets_on_confident(self):
        # one confident, two low-conf, then a confident hit on a NEW element with stickiness=1 should switch
        pol = self._policy(
            [
                ("A", 0.8, 0.5),   # switch A
                ("A", 0.4, 0.3),   # low → hold
                ("B", 0.8, 0.5),   # confident on B → switch (stickiness=1)
            ],
            threshold=0.55, gap=0.03, stickiness=1, clear_after_n_low=4,
        )
        _, a, _ = pol.step("x", 1); assert a == "A"
        _, a, _ = pol.step("x", 1); assert a == "A"
        _, a, why = pol.step("x", 1)
        assert why == "switch" and a == "B"

    def test_switch_requires_consecutive_confident_with_stickiness_2(self):
        pol = self._policy(
            [
                ("A", 0.8, 0.5),   # cand A (1)  → hold
                ("A", 0.8, 0.5),   # cand A (2)  → switch to A
                ("B", 0.8, 0.5),   # cand B (1)  → hold (active still A)
                ("B", 0.8, 0.5),   # cand B (2)  → switch to B
            ],
            threshold=0.55, gap=0.03, stickiness=2, clear_after_n_low=4,
        )
        _, a, why = pol.step("x", 1); assert (a, why) == (None, "hold")
        _, a, why = pol.step("x", 1); assert (a, why) == ("A", "switch")
        _, a, why = pol.step("x", 1); assert (a, why) == ("A", "hold")
        _, a, why = pol.step("x", 1); assert (a, why) == ("B", "switch")

    def test_alternating_confident_candidates_do_not_switch_with_stickiness_2(self):
        # Prime active=A with two consecutive A frames, then alternate B/C — never reach 2-in-a-row
        pol = self._policy(
            [
                ("A", 0.8, 0.5),   # cand A (1)  → hold
                ("A", 0.8, 0.5),   # cand A (2)  → switch (active=A)
                ("B", 0.8, 0.5),   # cand B (1)  → hold (active=A)
                ("C", 0.8, 0.5),   # cand C (1)  → hold (active=A)
                ("B", 0.8, 0.5),   # cand B (1, reset)  → hold (active=A)
                ("C", 0.8, 0.5),   # cand C (1)  → hold (active=A)
            ],
            threshold=0.55, gap=0.03, stickiness=2, clear_after_n_low=4,
        )
        for _ in range(6):
            pol.step("x", 1)
        # active should still be A — neither B nor C reached stickiness=2 consecutively
        assert pol.active_id == "A"

    def test_reset_clears_state(self):
        pol = self._policy([("A", 0.8, 0.5)], threshold=0.55, gap=0.03, stickiness=1, clear_after_n_low=4)
        pol.step("x", 1)
        assert pol.active_id == "A"
        pol.reset()
        assert pol.active_id is None


# ---------- end-to-end semantic acceptance (slow) ----------

@pytest.fixture(scope="module")
def loaded():
    if os.environ.get("SKIP_SEMANTIC"):
        pytest.skip("SKIP_SEMANTIC set")
    from src.encoder import load_default_encoder
    from src.matcher import Index, Matcher
    enc = load_default_encoder()
    idx = Index(ROOT / "out" / "index.json", enc)
    return Matcher(idx, enc), idx


@pytest.mark.slow
class TestSemanticAcceptance:
    @pytest.mark.parametrize("slide,spoken,expected_substr", [
        (6, "let me explain the matcher", "matcher"),
        (6, "audio comes in through the microphone first", "microphone"),
        (6, "and finally we draw the overlay box", "overlay"),
        (4, "the stt bar dominates the chart", "stt"),
        (4, "render only takes about sixteen milliseconds", "render"),
        (4, "sync is tiny", "sync"),
        (3, "the cost column shows it runs in sub millisecond time", "sub-millisecond"),
        (5, "at threshold zero point five", "0.5"),
        (2, "the headline number is sub one millisecond", "1 ms"),
        (2, "highlight it without a laser pointer", "laser pointer"),
    ])
    def test_paraphrase_resolves(self, loaded, slide, spoken, expected_substr):
        matcher, _ = loaded
        r = matcher.match(spoken, slide)
        assert r.text is not None
        assert expected_substr.lower() in r.text.lower(), (
            f"slide {slide} spoken={spoken!r} matched={r.text!r} score={r.score:.3f}")
        assert r.score >= 0.55, f"weak match for {spoken!r}: {r.score:.3f}"

    def test_filler_does_not_overshoot_threshold(self, loaded):
        matcher, _ = loaded
        # generic discourse markers should not lock onto any one element strongly
        r = matcher.match("as you can see in this diagram", 6)
        # either score is below threshold, OR gap to runner-up is tiny
        assert (r.score < 0.55) or ((r.score - r.runner_up_score) < 0.05), (
            f"filler scored too strongly: {r.score:.3f} vs ru {r.runner_up_score:.3f} on {r.text!r}")

    def test_cross_slide_isolation(self, loaded):
        # same query, two slides → different elements
        matcher, _ = loaded
        r_s4 = matcher.match("the stt bar", 4)
        r_s3 = matcher.match("the stt bar", 3)
        assert r_s4.element_id != r_s3.element_id
        assert r_s4.element_id.startswith("s4_")
        assert r_s3.element_id.startswith("s3_")
