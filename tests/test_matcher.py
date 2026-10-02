"""Stage 3 matcher tests.

Two layers:

1. `TestPolicy*` — deterministic unit tests of MatchPolicy using a stub matcher.
   Fast, no embedding model required, covers all state-machine branches.

2. `TestSemanticAcceptance` — end-to-end against the real index + BGE encoder.
   A small set of paraphrase probes that must resolve to the right element.
   Marked with `slow` because it loads the embedding model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from src.matcher import ElementRef, MatchResult, MatchPolicy, PolicyConfig  # noqa: E402


# ---------- stub matcher exposing scored() for group-aware policy tests ----------

def _ref(eid, group_id):
    return ElementRef(id=eid, type="x", text=eid, slide_no=1, phrases=[eid],
                      emb_start=0, emb_end=1, group_id=group_id)


class ScoredStubMatcher:
    """Returns a scripted list of (raw_score, ElementRef) per step via scored()."""

    def __init__(self, scored_seq):
        self._seq = list(scored_seq)
        self._i = 0

    def scored(self, spoken, slide_no):
        item = self._seq[self._i]
        self._i += 1
        return item


class TestGroupPolicy:
    def test_same_group_bonus_flips_a_close_call(self):
        # frame 1: A (group G) wins outright -> active A, active_group G
        # frame 2: raw best is B (group H, 0.70) but C (group G, 0.68) is in the
        #          active group; same_group_bonus (0.06) lifts C to 0.74 > B 0.70.
        A, B, C = _ref("A", "G"), _ref("B", "H"), _ref("C", "G")
        pol = MatchPolicy(
            ScoredStubMatcher([
                [(0.80, A), (0.50, B)],
                [(0.70, B), (0.68, C)],
            ]),
            PolicyConfig(threshold=0.55, gap=0.03, stickiness=1,
                         stay_bonus=0.08, same_group_bonus=0.06),
        )
        _, active, why = pol.step("x", 1)
        assert (active, why) == ("A", "switch")
        assert pol._active_group == "G"
        _, active, why = pol.step("x", 1)
        # C wins on effective score thanks to staying in group G
        assert (active, why) == ("C", "switch")

    def test_stay_bonus_holds_against_same_group_sibling(self):
        # active A (group G); a same-group sibling B leads raw by 0.01, but A's
        # stay_bonus (0.08) outweighs B's same_group_bonus (0.04):
        # A_eff = 0.70+0.08 = 0.78 > B_eff = 0.71+0.04 = 0.75 → A holds (no flicker).
        A, B = _ref("A", "G"), _ref("B", "G")
        pol = MatchPolicy(
            ScoredStubMatcher([
                [(0.80, A), (0.40, B)],
                [(0.71, B), (0.70, A)],
            ]),
            PolicyConfig(threshold=0.55, gap=0.03, stickiness=1,
                         stay_bonus=0.08, same_group_bonus=0.04),
        )
        pol.step("x", 1)                      # -> A
        _, active, why = pol.step("x", 1)
        assert active == "A" and why == "hold"

    def test_active_yields_freely_to_a_different_group(self):
        # active A (group G); rival B (group H) leads raw by 0.05 (> gap). Because
        # B is a DIFFERENT group, A gets NO stay bonus and yields (real switch).
        A, B = _ref("A", "G"), _ref("B", "H")
        pol = MatchPolicy(
            ScoredStubMatcher([
                [(0.80, A), (0.40, B)],
                [(0.75, B), (0.70, A)],
            ]),
            PolicyConfig(threshold=0.55, gap=0.03, stickiness=1,
                         stay_bonus=0.08, same_group_bonus=0.04),
        )
        pol.step("x", 1)                      # -> A
        _, active, why = pol.step("x", 1)
        assert active == "B" and why == "switch"


class TextStubMatcher:
    """scored() keyed by the spoken text, so long and short tails differ."""

    def __init__(self, table):
        self._table = table

    def scored(self, spoken, slide_no):
        return self._table.get(spoken, [])


class TestMultiScaleTail:
    """Full tail is primary; the last `short_words` words may switch to a new
    element only on the stricter `short_threshold` bar; the revert guard keeps
    the displaced element from bouncing straight back."""

    T, S, W = _ref("title", None), _ref("stage", None), _ref("when", None)
    CFG = dict(threshold=0.65, gap=0.03, stickiness=1, short_words=2,
               short_threshold=0.73, revert_guard=3)

    def test_dilution_short_tail_names_new_element(self):
        # 8-word tail still favours the title ("pipe line stages ..."), but the
        # last two words clearly name the "Stage" cell -> switch.
        pol = MatchPolicy(TextStubMatcher({
            "pipe line stages": [(0.92, self.T), (0.70, self.S)],
            "pipe line stages stage": [(0.92, self.T), (0.80, self.S)],
            "stages stage": [(0.93, self.S), (0.83, self.T)],
        }), PolicyConfig(**self.CFG))
        assert pol.step("pipe line stages", 1)[1:] == ("title", "switch")
        assert pol.step("pipe line stages stage", 1)[1:] == ("stage", "switch")

    def test_short_override_off_keeps_long_tail_decision(self):
        pol = MatchPolicy(TextStubMatcher({
            "pipe line stages": [(0.92, self.T), (0.70, self.S)],
            "pipe line stages stage": [(0.92, self.T), (0.80, self.S)],
            "stages stage": [(0.93, self.S), (0.83, self.T)],
        }), PolicyConfig(**{**self.CFG, "short_words": 0}))
        pol.step("pipe line stages", 1)
        assert pol.step("pipe line stages stage", 1)[1:] == ("title", "hold")

    def test_common_words_in_short_tail_do_not_switch(self):
        # "and the" sits below the short bar -> the full tail keeps the title
        pol = MatchPolicy(TextStubMatcher({
            "pipe line stages": [(0.92, self.T), (0.70, self.S)],
            "pipe line stages and the": [(0.88, self.T), (0.66, self.W)],
            "and the": [(0.70, self.W), (0.66, self.S)],
        }), PolicyConfig(**self.CFG))
        pol.step("pipe line stages", 1)
        assert pol.step("pipe line stages and the", 1)[1:] == ("title", "hold")

    def test_revert_guard_blocks_bounce_back_then_expires(self):
        pol = MatchPolicy(TextStubMatcher({
            "pipe line stages": [(0.92, self.T), (0.70, self.S)],
            "pipe line stages stage": [(0.92, self.T), (0.80, self.S)],
            "stages stage": [(0.93, self.S), (0.83, self.T)],
            # the old words linger: the full tail swings back to the title while
            # the short tail is weak
            "line stages stage um": [(0.90, self.T), (0.75, self.S)],
            "stage um": [(0.60, self.S), (0.55, self.T)],
        }), PolicyConfig(**self.CFG))
        pol.step("pipe line stages", 1)
        assert pol.step("pipe line stages stage", 1)[1] == "stage"
        for _ in range(3):   # revert_guard=3 steps: title barred
            assert pol.step("line stages stage um", 1)[1:] == ("stage", "hold")
        assert pol.step("line stages stage um", 1)[1:] == ("title", "switch")

    def test_reset_clears_revert_guard(self):
        pol = MatchPolicy(TextStubMatcher({
            "pipe line stages": [(0.92, self.T), (0.70, self.S)],
            "pipe line stages stage": [(0.92, self.T), (0.80, self.S)],
            "stages stage": [(0.93, self.S), (0.83, self.T)],
        }), PolicyConfig(**self.CFG))
        pol.step("pipe line stages", 1)
        pol.step("pipe line stages stage", 1)
        pol.reset()
        assert pol.step("pipe line stages", 1)[1:] == ("title", "switch")


class TestNullFloor:
    """An element must beat its own similarity to generic presenter talk
    (Matcher.null_floor) by `null_margin`: a 1-word label like "When" scores
    ~0.68 against any sentence, so generic filler must not light it."""

    W = _ref("when", None)
    TABLE = {"so let me just point out a couple of things": [(0.68, W)],
             "when": [(0.95, W)]}

    def _pol(self, margin):
        m = TextStubMatcher(self.TABLE)
        m.null_floor = {"when": 0.63}          # bar = max(0.65, 0.63 + margin)
        return MatchPolicy(m, PolicyConfig(threshold=0.65, gap=0.03, stickiness=1,
                                           short_words=0, null_margin=margin))

    def test_generic_sentence_does_not_light_one_word_element(self):
        pol = self._pol(0.12)
        assert pol.step("so let me just point out a couple of things", 1)[1:] == (None, "noop")

    def test_own_word_lights_it(self):
        assert self._pol(0.12).step("when", 1)[1:] == ("when", "switch")

    def test_without_null_margin_the_generic_sentence_lights_it(self):
        # the mechanism, not the threshold, is what blocks the filler
        pol = self._pol(None)
        assert pol.step("so let me just point out a couple of things", 1)[1:] == ("when", "switch")


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
def loaded(real_encoder):
    from src.matcher import Index, Matcher
    enc = real_encoder
    idx = Index(ROOT / "out" / "index.json", enc)
    return Matcher(idx, enc), idx


@pytest.mark.slow
def test_real_null_floor_blocks_filler_but_not_the_word(loaded):
    """Real encoder: slide 3's one-word "When" cell stays dark under generic
    presenter filler and lights when its word is spoken."""
    matcher, _idx = loaded
    filler = ("i want to move on to the ne",   # a real STT tail of the filler pass
              "so what happens next")
    on = PolicyConfig().null_margin
    cases = ([(f, None, True) for f in filler]          # without the null floor: lit
             + [(f, on, False) for f in filler] + [("when", on, True)])
    for spoken, margin, lit in cases:
        pol = MatchPolicy(matcher, PolicyConfig(null_margin=margin))
        pol.step(spoken, 3)
        assert (pol.active_id == "s3_e2") is lit, (spoken, margin)


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
        # either score is below the live threshold, OR gap to runner-up is tiny
        assert (r.score < PolicyConfig.threshold) or ((r.score - r.runner_up_score) < 0.05), (
            f"filler scored too strongly: {r.score:.3f} vs ru {r.runner_up_score:.3f} on {r.text!r}")

    @pytest.mark.parametrize("filler", ["and the", "so the", "of the", "it is", "this is", "to the", "in a"])
    def test_common_word_pair_stays_below_short_bar(self, loaded, filler):
        # the 2-word short tail must not clear the stricter short bar on filler
        matcher, idx = loaded
        for sno in idx.slide_numbers():
            per_el = matcher.scored(filler, sno)
            if per_el:
                assert per_el[0][0] < PolicyConfig.short_threshold, (
                    f"{filler!r} -> {per_el[0][1].text!r} {per_el[0][0]:.3f} on slide {sno}")

    def test_dilution_switches_on_short_tail(self, loaded):
        # "Pipeline Stages" (title) still dominates the full tail while the
        # speaker reads the "Stage" header cell; the short tail must switch.
        matcher, idx = loaded
        by_text = {(r.text or ""): r.id for r in idx.elements_on(3)}
        pol = MatchPolicy(matcher, PolicyConfig())
        _, active, _ = pol.step("pipe line stages", 3)
        assert active == by_text["Pipeline Stages"]
        res, active, why = pol.step("pipe line stages stage", 3)
        assert active == by_text["Stage"] and why == "switch", res

    def test_cross_slide_isolation(self, loaded):
        # same query, two slides → different elements
        matcher, _ = loaded
        r_s4 = matcher.match("the stt bar", 4)
        r_s3 = matcher.match("the stt bar", 3)
        assert r_s4.element_id != r_s3.element_id
        assert r_s4.element_id.startswith("s4_")
        assert r_s3.element_id.startswith("s3_")


# ---------- slide furniture ----------

def _el(eid, text, y, h=0.1, type="body"):
    return {"id": eid, "type": type, "text": text, "bbox": {"x": 1.0, "y": y, "w": 2.0, "h": h}}


def _furniture_deck():
    """5 slides, 7.5 in tall: a running footer (with a changing page count) and a
    page number in the bottom margin on every slide; content that must survive."""
    slides = []
    for n in range(1, 6):
        els = [_el(f"s{n}_title", f"Step {n}", 0.3, type="title"),  # repeats ("Step #"), in the margin
               _el(f"s{n}_head", f"Part {n}", 0.3),                # same, but not a title -> header
               _el(f"s{n}_foot", f"Acme Review | Page {n + 1}", 7.1),
               _el(f"s{n}_pg", str(n), 7.1),
               _el(f"s{n}_body", "Revenue grew", 3.0)]
        slides.append({"slide_number": n, "elements": els})
    slides[2]["elements"].append(_el("s3_stat", "3", 3.5))          # mid-slide stat "3" on slide 3
    slides[3]["elements"].append(_el("s4_num", "7", 7.1))           # margin number != slide number
    return {"slide_size": {"w": 13.33, "h": 7.5}, "slides": slides}


def test_furniture_ids_flags_footer_and_page_numbers_only():
    from src.matcher import furniture_ids
    got = furniture_ids(_furniture_deck())
    assert got == {f"s{n}_{k}" for n in range(1, 6) for k in ("foot", "pg", "head")}
    assert "s3_stat" not in got and "s4_num" not in got      # mid-slide / not its slide number
    deck = _furniture_deck()
    for s in deck["slides"]:
        s["elements"][1]["bbox"]["y"] = 2.0                          # the repeat moved mid-slide
    assert not {e for e in furniture_ids(deck) if e.endswith("_head")}


def test_furniture_needs_a_majority_of_slides():
    from src.matcher import furniture_ids
    deck = _furniture_deck()
    for s in deck["slides"][2:]:                                     # footer on 2 of 5 slides only
        s["elements"] = [e for e in s["elements"] if not e["id"].endswith("_foot")]
    assert not {e for e in furniture_ids(deck) if e.endswith("_foot")}
    assert furniture_ids({"slides": deck["slides"]}) == set()        # no slide size -> no geometry


def test_index_makes_furniture_unpointable(tmp_path):
    import json
    import numpy as np
    from src.matcher import Index

    class Enc:
        dim, model_name = 4, "stub"
        def encode(self, texts):
            return np.ones((len(texts), 4), dtype=np.float32)

    deck = _furniture_deck()
    for s in deck["slides"]:
        for e in s["elements"]:
            e["trigger_phrases"] = [e["text"]]
    p = tmp_path / "index.json"
    p.write_text(json.dumps(deck))
    refs = {r.id: r for r in Index(p, Enc()).elements_on(1)}
    assert not refs["s1_foot"].is_pointable and not refs["s1_pg"].is_pointable
    assert refs["s1_body"].is_pointable and refs["s1_title"].is_pointable

