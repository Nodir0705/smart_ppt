"""Headless live-decision core, factored out of the Tk viewer.

The viewer, the PNG simulator, and the video simulator all run the same loop:

    buffer.tail(N) -> policy.step(tail, slide_no) -> react to the active element

(``policy.step`` derives the multi-scale tail — per-phrase windows plus the
short-tail override — itself, so the bench audit, which also runs through this
class, measures exactly what the live app does.)

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
    line_index: int | None = None   # within a soft-wrapped element, the line in focus
    line_bbox: dict | None = None    # that line's sub-bbox (None = use the union bbox)


def select_line(tail: str, line_texts: list[str]) -> int | None:
    """Pick which wrapped line the speaker is currently on, or None.

    A soft-wrapped paragraph is ONE element with several `line_texts`; this
    chooses the line to highlight from the recent spoken tail. Deterministic,
    offline, no wordlist:

      score(line) = Σ over tail words w in the line of  recency(w) / df(w)

    where recency = the word's 1-based position in the tail (later words — the
    reading head — weigh more) and df = how many of the element's lines contain
    w. Dividing by df self-cancels stopwords ("the"/"of" sit in every line, so
    each contributes ~0) without a hand-maintained stoplist — purely from the
    element's own text. Ties break to the LATER line (reading moves forward).

    Returns None for single-line elements (nothing to sub-select) and when no
    tail word touches any line (caller falls back to the union bbox).
    """
    if not line_texts or len(line_texts) < 2:
        return None
    words = (tail or "").lower().split()
    if not words:
        return None
    line_sets = [set(lt.lower().split()) for lt in line_texts]
    df: dict[str, int] = {}
    for s in line_sets:
        for w in s:
            df[w] = df.get(w, 0) + 1
    scores = [0.0] * len(line_texts)
    for pos, w in enumerate(words):
        d = df.get(w)
        if not d:
            continue
        weight = (pos + 1) / d
        for i, s in enumerate(line_sets):
            if w in s:
                scores[i] += weight
    best = max(range(len(scores)), key=lambda i: (scores[i], i))
    return best if scores[best] > 0 else None


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
        # id -> (line_texts, line_bboxes) for elements that soft-wrap into >1
        # line; only these can sub-select a line. Built once from the index.
        self._lines_by_id: dict[str, tuple[list[str], list[dict]]] = {}
        for slide in raw_index.get("slides", []):
            for el in slide.get("elements", []):
                texts = el.get("line_texts") or []
                boxes = el.get("line_bboxes") or []
                if len(texts) > 1 and len(texts) == len(boxes):
                    self._lines_by_id[el["id"]] = (texts, boxes)
        # Line-level hysteresis: remember the line chosen for the active element so
        # a frame whose (retracted/rewritten) STT tail momentarily matches no line
        # HOLDS the last line instead of snapping back to the union box. Reset when
        # the active element changes. Owned here so the viewer and both simulators
        # draw the identical rectangle (the drift LiveController exists to prevent).
        self._line_active_id: str | None = None
        self._line_index: int | None = None

    def push(self, partial: Partial) -> None:
        self.buffer.push(partial)

    def reset(self) -> None:
        self.policy.reset()
        self._line_active_id = None
        self._line_index = None

    def feed(self, text: str | None = None) -> MatchEvent | None:
        """Run one decision over the current buffer tail (falling back to `text`
        when the buffer is empty). Returns None when there's nothing to match."""
        tail = self.buffer.tail(self.tail_words) or (text or "")
        if not tail.strip():
            return None
        res, active, why = self.policy.step(tail, self.slide_no)
        if active != self._line_active_id:
            self._line_active_id = active
            self._line_index = None          # new (or no) active element → forget the line
        line_index = line_bbox = None
        lines = self._lines_by_id.get(active) if active else None
        if lines is not None:
            texts, boxes = lines
            # Select on the NORMALIZED tail so it's in the same space as the
            # deck's literal line_texts (e.g. STT "off line" -> "offline").
            norm_tail = self.policy.normalizer(tail) if self.policy.normalizer else tail
            idx = select_line(norm_tail, texts)
            if idx is None:
                idx = self._line_index       # hold the last line through a momentary miss
            if idx is not None:
                self._line_index = idx
                line_index, line_bbox = idx, boxes[idx]
        return MatchEvent(
            active_id=active, element_id=res.element_id,
            score=res.score, why=why, tail=tail,
            line_index=line_index, line_bbox=line_bbox,
        )
