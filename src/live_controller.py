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

import functools
import re
from dataclasses import dataclass

from .asr_repair import cmudict
from .clause_pointer import ClauseParams, ClausePointer, slide_units
from .matcher import Matcher, MatchPolicy, PolicyConfig
from .stt import Partial, RollingBuffer


@dataclass
class MatchEvent:
    """One decision over the current buffer window."""
    active_id: str | None   # unit the policy committed to (its first element's id; None = nothing lit)
    element_id: str | None  # best raw element this frame (may differ when hold/noop)
    score: float
    why: str                # "switch" | "hold" | "clear" | "noop"
    tail: str               # the windowed text actually matched
    line_index: int | None = None   # within a soft-wrapped element, the line in focus
    line_bbox: dict | None = None    # that line's sub-bbox (None = use the union bbox)
    unit_bbox: dict | None = None    # a multi-element unit's union bbox (cards)

    @property
    def bbox(self) -> dict | None:
        """The box to draw instead of the active element's own bbox, or None:
        the focused line, else the unit's union box. Every consumer draws this."""
        return self.line_bbox or self.unit_bbox


def _words(text: str) -> list[str]:
    """Lowercased words without edge punctuation, so a deck's "Extract," matches
    the STT's bare "extract" (inner apostrophes stay: "teacher's")."""
    return re.findall(r"\w+(?:'\w+)*", text.lower())


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
    words = _words(tail or "")
    if not words:
        return None
    line_sets = [set(_words(lt)) for lt in line_texts]
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


@functools.lru_cache(maxsize=1)
def _default_decider():
    """The shipped decider, loaded once per process (consumers build a
    controller per slide). CPU in every consumer, so the bench measures live."""
    from .deciders.cross import CrossDecider
    return CrossDecider()


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
        pointer: str = "clause",
        decider=None,
        clause_params: ClauseParams | None = None,
    ) -> None:
        if normalizer is None:
            normalizer = _build_normalizer(raw_index)
        self.policy = MatchPolicy(matcher, policy_config or PolicyConfig(), normalizer=normalizer)
        # Clause pointer (src/clause_pointer.py) on slides that have an
        # understanding map; "tail" (or an unmapped slide) keeps the tail policy.
        if pointer not in ("clause", "tail"):
            raise ValueError(f"pointer must be 'clause' or 'tail', got {pointer!r}")
        self._mapped = {s["slide_number"]: s for s in raw_index.get("slides", [])
                        if pointer == "clause" and s.get("map")}
        self._clause_params = clause_params or ClauseParams()
        if decider is None and self._mapped:   # load the model now, not mid-talk
            decider = _default_decider()
        if self._mapped and self._clause_params.repair > 0:
            cmudict()                          # and the pronunciations (~80 ms)
        self._decider = decider
        self._slide_size = raw_index.get("slide_size")
        self._clause: ClausePointer | None = None
        self._clause_slide: int | None = None
        self._clause_head: dict[str, str] = {}
        self.buffer = RollingBuffer(window_seconds=window_seconds)
        self._origin = (self.buffer, 0)   # (buffer, word count) when the slide's speech began
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
        index = getattr(matcher, "index", None)
        self.unit_bbox = getattr(index, "unit_bbox", lambda _u: None)

    def push(self, partial: Partial) -> None:
        self.buffer.push(partial)

    def reset(self) -> None:
        """New slide (or restart): forget decisions; the next clause starts at the
        words heard from now on."""
        self.policy.reset()
        self._clause = self._clause_slide = None
        self._origin = (self.buffer, len(self.buffer.words()))
        self._line_active_id = None
        self._line_index = None

    def _clause_step(self) -> tuple[str | None, str | None, float, str, str]:
        """One clause-pointer update on a mapped slide ->
        (active unit id, best unit id, margin, why, clause text)."""
        if self._clause is None or self._clause_slide != self.slide_no:
            units, self._clause_head, low = slide_units(self._mapped[self.slide_no], self._slide_size)
            # the slide's speech starts at the last reset (a swapped-in buffer is
            # all this slide's); a slide change without reset starts now
            buf, n = self._origin
            start = (n if self._clause_slide is None else len(self.buffer.words())) if buf is self.buffer else 0
            self._clause = ClausePointer(self._decider, units, self._clause_params, low_prior=low,
                                         start=start,
                                         normalizer=self.policy.normalizer)
            self._clause_slide = self.slide_no
        d = self._clause.update(self.buffer.words(), self.buffer.boundaries)
        head = self._clause_head.get
        return head(d.active) if d.active else None, head(d.best) if d.best else None, d.margin, d.why, d.clause

    def feed(self, text: str | None = None) -> MatchEvent | None:
        """Run one decision over the current buffer tail (falling back to `text`
        when the buffer is empty). Returns None when there's nothing to match."""
        if self.slide_no in self._mapped:
            if not self.buffer.words():
                return None
            active, best, score, why, tail = self._clause_step()
        else:
            tail = self.buffer.tail(self.tail_words) or (text or "")
            if not tail.strip():
                return None
            res, active, why = self.policy.step(tail, self.slide_no)
            best, score = res.element_id, res.score
        if active != self._line_active_id:
            self._line_active_id = active
            self._line_index = None          # new (or no) active element → forget the line
        line_index = line_bbox = None
        unit_bbox = self.unit_bbox(active)
        # a multi-element unit draws its union box; line focus is per paragraph
        lines = self._lines_by_id.get(active) if active and unit_bbox is None else None
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
            active_id=active, element_id=best,
            score=score, why=why, tail=tail,
            line_index=line_index, line_bbox=line_bbox, unit_bbox=unit_bbox,
        )
