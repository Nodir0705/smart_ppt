"""Clause pointer: decide once per spoken clause which slide-map unit it is about,
point at it and hold.

The tail matcher (``matcher.MatchPolicy``) re-scores an 8-word sliding tail every
STT update, so the box hops as words enter and leave the window. The clause
pointer instead segments speech into clauses at audio pauses and asks a
*decider* which unit of the slide's understanding map the current clause (with
the previous clause as context) refers to:

* clause = the words since the last boundary; a boundary is speech resuming
  after a pause >= ``MIN_PAUSE_S`` (``PauseDetector`` on the PCM chunks) or the
  clause reaching ``max_words``;
* commit: move to unit U once U beats NONE by ``commit`` (plus ``title_extra``
  for title / section-label units) and the runner-up by ``gap`` — mid-clause,
  as soon as the clause has ``min_words`` words; a new clause that would move a
  box that is already lit needs ``jump_gap`` over the runner-up instead (the
  wrong jumps were confident 5-word fragments: "and on the right is");
* a title / section-label unit is a candidate only when the clause says one of
  its own words that no other unit on the slide has (intro talk otherwise lights
  the title as the slide's first box);
* after a commit in this clause, change only when another unit beats the
  committed one by ``switch`` (hysteresis);
* before scoring, misheard slide words are put back by sound
  (``asr_repair.SoundRepair``: "colonel" -> kernel, "h l" -> HAL);
* right card, wrong box: when the winner is part_of / explains another
  (non-title) unit whose score is within ``card`` of it, the card lights, and a
  move between a unit and its own card is no topic jump (``gap``, not
  ``jump_gap``);
* a clause that commits to nothing HOLDS the box; boundaries keep the box.
  There is no clear: only a slide change resets.

Decider interface (shared with ``src/deciders/*``):
``decide(clause, context, units) -> {uid: score, ..., "none": score}`` with
units = [{"uid", "text", "summary", "talk", "terms"}]; higher = more likely,
comparable within one call.

Pure and deterministic: no clocks, no threads; the caller feeds words and
boundaries (live: ``stt.RollingBuffer`` does both on the audio thread).
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass

import numpy as np

from .asr_repair import SoundRepair

MIN_PAUSE_S = 0.25
TITLE_ROLES = {"title", "section_label"}
CARD_RELATIONS = ("part_of", "explains")   # child -> the card / heading it belongs to


class PauseDetector:
    """Energy VAD on 16-bit mono PCM chunks.

    20 ms frames; a frame is silent when its RMS is below
    max(``floor``, ``rel`` x the 95th percentile of the last ``history_s`` of
    frame RMS) — a level that adapts to the mic gain. ``feed`` returns True when
    speech resumes after >= ``min_pause`` s of silence (a clause boundary).
    """

    def __init__(self, sample_rate: int = 16000, min_pause: float = MIN_PAUSE_S,
                 frame_ms: int = 20, floor: float = 200.0, rel: float = 0.05,
                 history_s: float = 30.0) -> None:
        self.hop = sample_rate * frame_ms // 1000
        self.frame_s = frame_ms / 1000
        self.min_pause, self.floor, self.rel = min_pause, floor, rel
        self._hist: deque[float] = deque(maxlen=int(history_s / self.frame_s))
        self._rest = np.zeros(0, dtype=np.float32)
        self._silent_run = 0

    def feed(self, chunk: bytes) -> bool:
        pcm = np.concatenate([self._rest, np.frombuffer(chunk, dtype=np.int16).astype(np.float32)])
        n = len(pcm) // self.hop
        self._rest = pcm[n * self.hop:]
        resumed = False
        for f in pcm[:n * self.hop].reshape(n, self.hop) if n else ():
            rms = float(np.sqrt(np.mean(f * f)))
            self._hist.append(rms)
            silent = rms < max(self.floor, self.rel * float(np.percentile(self._hist, 95)))
            if silent:
                self._silent_run += 1
            else:
                # frame count, not float seconds: 13 x 0.02 >= 0.25 exactly
                if self._silent_run and self._silent_run * self.frame_s >= self.min_pause - 1e-9:
                    resumed = True
                self._silent_run = 0
        return resumed


def _side(boxes: list[dict], size: dict) -> str:
    """'on the left top' etc. from the members' bbox centres (empty in the middle)."""
    cx = sum(b["x"] + b["w"] / 2 for b in boxes) / len(boxes) / size["w"]
    cy = sum(b["y"] + b["h"] / 2 for b in boxes) / len(boxes) / size["h"]
    pos = (["left"] if cx < 0.4 else ["right"] if cx > 0.6 else []) + \
          (["top"] if cy < 0.35 else ["bottom"] if cy > 0.7 else [])
    return "on the " + " ".join(pos) if pos else ""


def slide_units(slide: dict, size: dict | None = None) -> tuple[list[dict], dict[str, str], set[str]]:
    """A mapped index slide -> (decider units, uid -> unit id the policy emits
    (its first member), uids of title / section-label units).

    With the deck's ``slide_size``, each unit's terms gain where it sits ("on the
    right"): speakers point by place ("the number on the top right"), and a
    text-only re-ranker can't resolve that."""
    m = slide["map"]
    info = m.get("info") or {}
    by_id = {e["id"]: e for e in slide["elements"]}
    units, head, low = [], {}, set()
    for uid, members in m["units"].items():
        i = info.get(uid, {})
        units.append({"uid": uid, "text": " ".join(by_id[x].get("text") or "" for x in members).strip(),
                      "summary": i.get("summary", ""), "talk": list(i.get("talk", [])),
                      "terms": list(i.get("terms", [])),
                      "parents": [r["to"] for r in m.get("relations", [])
                                  if r["from"] == uid and r["type"] in CARD_RELATIONS and r["to"] in m["units"]]})
        boxes = [by_id[x]["bbox"] for x in members if (by_id[x].get("bbox") or {}).get("w") is not None]
        if size and boxes and (side := _side(boxes, size)):
            units[-1]["terms"].append(side)
        head[uid] = members[0]
        if all(by_id[x].get("map_role") in TITLE_ROLES for x in members):
            low.add(uid)
    return units, head, low


@dataclass(frozen=True)
class ClauseParams:
    """Locked on the original + jumpy talks (AI_Handout) with the default cross
    decider; every setting in commit -8..-5.5 (as the decider's NONE bar),
    switch 0.5..1.5, title_extra 2..3 at min_words 5 passed the gate.
    jump_gap 2 / title_extra 4 (with the title word gate and position terms):
    locked on original + jumpy of AI_Handout and Open-Q (tasks/lessons.md)."""
    min_words: int = 5        # clause words before its first decision (a boundary leaks 1-2 words of the last sentence; 6 is too late)
    max_words: int = 35       # fallback boundary when nobody pauses (oracle clauses: median 15, max 35)
    commit: float = 0.0       # winner over NONE (the decider's own bar)
    gap: float = 1.0          # winner over the runner-up
    jump_gap: float = 2.0     # winner over the runner-up when a new clause would move a lit box
    switch: float = 1.0       # rival over the unit already committed in this clause
    title_extra: float = 4.0  # extra over NONE for title / section-label units
    repair: float = 0.25      # sound-based ASR repair to the slide's words: max phoneme distance (<= 0 = off)
    card: float = 4.0         # light the parent card when it is within this of a winning child (< 0 = off)


@dataclass
class Decision:
    active: str | None   # uid lit after this update
    why: str             # "switch" | "hold" | "noop"
    best: str | None = None
    margin: float = 0.0  # best minus NONE on the last decider call
    clause: str = ""


class ClausePointer:
    """Clause segmentation + commit rule over one slide's map units."""

    def __init__(self, decider, units: list[dict], params: ClauseParams = ClauseParams(),
                 low_prior: set[str] = frozenset(), start: int = 0, normalizer=None) -> None:
        self.decider, self.units, self.p = decider, units, params
        self.low_prior = set(low_prior)
        self.normalizer = normalizer
        self.repair = SoundRepair(units, params.repair) if params.repair > 0 else None
        self.start = start            # word index where the current clause begins
        self.prev = ""                # previous clause text (context)
        self.active: str | None = None
        self.committed: str | None = None   # unit committed in THIS clause
        self._last_clause: str | None = None
        # title words no other unit on the slide has (``_words`` stems)
        others = {u["uid"]: _words(" ".join([u["summary"], u["text"], *u["talk"], *u["terms"]]))
                  for u in units}
        self.title_words = {t: _words(u["text"]) - set().union(*(w for v, w in others.items() if v != t))
                            for u in units if (t := u["uid"]) in self.low_prior}
        # child -> its cards (title / section labels excluded: they have their own gate)
        self.parents = {u["uid"]: [q for q in u.get("parents", []) if q not in self.low_prior] for u in units}

    def update(self, words: list[str], boundaries: list[int] = ()) -> Decision:
        """Fold in boundaries (word indices, ascending) and decide on the current clause."""
        for b in boundaries:
            if b > self.start:
                self.prev = " ".join(words[self.start:b])
                self.start, self.committed, self._last_clause = b, None, None
        if len(words) - self.start >= self.p.max_words:
            self.prev = " ".join(words[self.start:])
            self.start, self.committed, self._last_clause = len(words), None, None
        clause = " ".join(words[self.start:])
        if len(words) - self.start < self.p.min_words or clause == self._last_clause:
            return Decision(self.active, "hold" if self.active else "noop", clause=clause)
        self._last_clause = clause
        ctx = self.prev
        if self.normalizer is not None:
            clause, ctx = self.normalizer(clause), self.normalizer(ctx)
        if self.repair is not None:
            clause, ctx = self.repair(clause), self.repair(ctx)
        scores = self.decider.decide(clause, ctx, self.units)
        return self._commit(scores, clause)

    def _commit(self, scores: dict[str, float], clause: str) -> Decision:
        none = scores.get("none", float("-inf"))
        said = _words(clause)
        scores = {u: (float("-inf") if u in self.title_words and not said & self.title_words[u] else s)
                  for u, s in scores.items()}
        ranked = sorted((u["uid"] for u in self.units), key=lambda u: -scores.get(u, float("-inf")))
        best = ranked[0]
        s = scores[best]
        need = self.p.commit + (self.p.title_extra if best in self.low_prior else 0.0)
        clears = s - none >= need
        family = {best}
        if self.p.card >= 0 and (cards := [q for q in self.parents.get(best, []) if q in scores]):
            card = max(cards, key=scores.get)
            if s - scores[card] <= self.p.card:     # right card, wrong box: light the card
                best, family = card, {best, card}
        second = max((scores.get(u, float("-inf")) for u in ranked if u not in family), default=float("-inf"))
        why = "hold" if self.active else "noop"
        if self.committed is None:
            # child <-> its own card is no topic jump
            related = self.p.card >= 0 and (self.active in self.parents.get(best, ())
                                            or best in self.parents.get(self.active, ()))
            gap = self.p.jump_gap if self.active is not None and not related else self.p.gap
            if clears and s - second >= gap:
                self.committed = best
                if best != self.active:
                    self.active, why = best, "switch"
        elif best != self.committed and clears and s - scores[self.committed] >= self.p.switch:
            self.committed = best
            if best != self.active:
                self.active, why = best, "switch"
        return Decision(self.active, why, best, s - none, clause)


def _words(text: str) -> set[str]:
    """Content-word stems (first 5 letters of words >= 4 long) for the title gate."""
    return {w[:5] for w in re.findall(r"[a-z0-9][a-z0-9']+", text.lower()) if len(w) >= 4}
