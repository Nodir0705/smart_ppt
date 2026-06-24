"""LiveController — the headless live-decision core.

These exercise the seam that the Tk viewer and the PNG/video simulators all
share, WITHOUT a display, a microphone, or the embedding model. A recording
stub matcher captures exactly what text reaches the policy, so we can assert the
controller windows the buffer tail and applies the alias normalizer before
scoring — the behaviour the three call sites had silently drifted on.
"""

from __future__ import annotations

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
