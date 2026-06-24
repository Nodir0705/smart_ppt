"""Headless live-decision core, factored out of the Tk viewer.

The viewer, the PNG simulator, and the video simulator all run the same loop:

    buffer.tail(N) -> policy.step(tail, slide_no) -> react to the active element

They had drifted: the viewer and the video path built the policy *with* the
deck's alias normalizer, but the PNG path silently omitted it, so PNG previews
made different highlight decisions than the live app. ``LiveController`` owns
that core so every consumer makes identical decisions — and it can be unit
tested without Tk or a microphone.
"""

from __future__ import annotations

from dataclasses import dataclass

from .matcher import Matcher, MatchPolicy, PolicyConfig
from .stt import Partial, RollingBuffer


@dataclass
class MatchEvent:
    """One decision over the current buffer window."""
    active_id: str | None   # element the policy committed to (None = nothing lit)
    element_id: str | None  # best raw element this frame (may differ when hold/noop)
    score: float
    why: str                # "switch" | "hold" | "clear" | "noop"
    tail: str               # the windowed text actually matched


def _build_normalizer(raw_index: dict):
    """Deterministic post-STT alias normalizer derived from the deck (fixes
    acronym mishearings). Returns None when the deck yields no aliases."""
    from .asr_normalize import build_alias_map, normalize
    alias = build_alias_map(raw_index)
    return (lambda t: normalize(t, alias)) if alias else None


class LiveController:
    """Tk-free, audio-free live grounding decision unit.

    Owns the ``MatchPolicy`` (built with the deck normalizer) and the
    ``RollingBuffer``. Feed it Partials via :meth:`push`; call :meth:`feed` to
    run one decision over the current windowed tail.
    """

    def __init__(
        self,
        matcher: Matcher,
        raw_index: dict,
        *,
        policy_config: PolicyConfig | None = None,
        normalizer=None,
        tail_words: int = 8,
        window_seconds: float = 4.0,
        slide_no: int = 1,
    ) -> None:
        if normalizer is None:
            normalizer = _build_normalizer(raw_index)
        self.policy = MatchPolicy(matcher, policy_config or PolicyConfig(), normalizer=normalizer)
        self.buffer = RollingBuffer(window_seconds=window_seconds)
        self.tail_words = tail_words
        self.slide_no = slide_no

    def push(self, partial: Partial) -> None:
        self.buffer.push(partial)

    def reset(self) -> None:
        self.policy.reset()

    def feed(self, text: str | None = None) -> MatchEvent | None:
        """Run one decision over the current buffer tail (falling back to `text`
        when the buffer is empty). Returns None when there's nothing to match."""
        tail = self.buffer.tail(self.tail_words) or (text or "")
        if not tail.strip():
            return None
        res, active, why = self.policy.step(tail, self.slide_no)
        return MatchEvent(
            active_id=active, element_id=res.element_id,
            score=res.score, why=why, tail=tail,
        )
