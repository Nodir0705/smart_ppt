"""LiveController — the headless live-decision core.

These exercise the seam that the Tk viewer and the PNG/video simulators all
share, WITHOUT a display, a microphone, or the embedding model. A recording
stub matcher captures exactly what text reaches the policy, so we can assert the
controller windows the buffer tail and applies the alias normalizer before
scoring — the behaviour the three call sites had silently drifted on.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.live_controller import LiveController, MatchEvent
from src.stt import Partial


class _RecordingMatcher:
    """Captures each `spoken` string scored() receives; returns no elements
    (so the policy emits a no-op, which is all these wiring tests need)."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def scored(self, spoken: str, slide_no: int):
        self.seen.append(spoken)
        return []


def _identity(t: str) -> str:
    return t


def test_feed_returns_none_when_nothing_to_match():
    c = LiveController(_RecordingMatcher(), {"slides": []}, normalizer=_identity)
    assert c.feed() is None          # empty buffer, no fallback text
    assert c.feed("   ") is None     # whitespace-only fallback


def test_feed_windows_the_buffer_tail():
    m = _RecordingMatcher()
    c = LiveController(m, {"slides": []}, normalizer=_identity, tail_words=3)
    c.push(Partial("one two three four five", is_final=True, timestamp=1.0))
    ev = c.feed()
    assert isinstance(ev, MatchEvent)
    # only the last 3 words reach the policy/matcher
    assert m.seen[-1].split() == ["three", "four", "five"]


def test_feed_applies_the_normalizer_before_scoring():
    m = _RecordingMatcher()
    # the same class of fix the deck alias layer performs (mishearing repair)
    c = LiveController(m, {"slides": []}, normalizer=lambda t: t.replace("sink", "sync"))
    c.push(Partial("and sink is four", is_final=True, timestamp=1.0))
    c.feed()
    assert "sync" in m.seen[-1] and "sink" not in m.seen[-1]


def test_default_normalizer_is_derived_from_the_deck():
    # with no explicit normalizer, the controller builds one from the index;
    # a deck with no acronyms yields a None normalizer (text passes through).
    m = _RecordingMatcher()
    c = LiveController(m, {"slides": []})
    c.push(Partial("plain words", is_final=True, timestamp=1.0))
    c.feed()
    assert m.seen[-1] == "plain words"


class _OneElementMatcher:
    """Always returns one strong element so the policy commits to it (active),
    letting us exercise the line-selection wiring downstream of the match."""

    def __init__(self, ref) -> None:
        self.ref = ref

    def scored(self, spoken: str, slide_no: int):
        return [(0.9, self.ref)]


def _multiline_index():
    return {"slides": [{"slide_number": 1, "elements": [{
        "id": "e0", "type": "body", "text": "alpha one beta two",
        "line_texts": ["alpha one", "beta two"],
        "line_bboxes": [
            {"x": 0.0, "y": 0.0, "w": 1.0, "h": 0.2, "units": "inches"},
            {"x": 0.0, "y": 0.2, "w": 1.0, "h": 0.2, "units": "inches"},
        ],
    }]}]}


def _ref():
    return SimpleNamespace(id="e0", type="body", text="alpha one beta two",
                           group="e0", is_pointable=True)


def test_feed_sub_selects_the_wrapped_line_in_focus():
    c = LiveController(_OneElementMatcher(_ref()), _multiline_index(),
                       normalizer=_identity, tail_words=8)
    c.push(Partial("beta two", is_final=True, timestamp=1.0))
    ev = c.feed()
    assert ev.active_id == "e0"
    assert ev.line_index == 1
    assert ev.line_bbox["y"] == 0.2          # the second line's box


def test_feed_holds_last_line_through_a_momentary_miss():
    # once a line is chosen, a later frame whose tail matches NO line must HOLD
    # the last line (anti-flicker), not snap back to the union box.
    c = LiveController(_OneElementMatcher(_ref()), _multiline_index(),
                       normalizer=_identity, tail_words=8)
    c.push(Partial("beta two", is_final=True, timestamp=1.0))
    assert c.feed().line_index == 1
    c.push(Partial("zzz nothing matches", is_final=True, timestamp=2.0))
    ev = c.feed()
    assert ev.line_index == 1 and ev.line_bbox["y"] == 0.2   # held, not None


def test_feed_forgets_the_line_when_active_element_changes():
    idx = _multiline_index()
    idx["slides"][0]["elements"].append({
        "id": "e1", "type": "body", "text": "other", "line_texts": [], "line_bboxes": [],
    })
    ref = _ref()
    m = _OneElementMatcher(ref)
    c = LiveController(m, idx, normalizer=_identity, tail_words=8)
    c.push(Partial("beta two", is_final=True, timestamp=1.0))
    assert c.feed().line_index == 1
    # switch the active element; the single-line e1 has no line to select
    m.ref = SimpleNamespace(id="e1", type="body", text="other", group="e1", is_pointable=True)
    c.push(Partial("other", is_final=True, timestamp=2.0))
    ev = c.feed()
    assert ev.active_id == "e1" and ev.line_index is None and ev.line_bbox is None


def test_feed_line_bbox_is_none_for_single_line_element():
    idx = _multiline_index()
    el = idx["slides"][0]["elements"][0]
    el["line_texts"] = ["alpha one beta two"]          # collapses to one line
    el["line_bboxes"] = [el["line_bboxes"][0]]
    c = LiveController(_OneElementMatcher(_ref()), idx,
                       normalizer=_identity, tail_words=8)
    c.push(Partial("alpha one beta two", is_final=True, timestamp=1.0))
    ev = c.feed()
    assert ev.active_id == "e0"
    assert ev.line_index is None and ev.line_bbox is None
