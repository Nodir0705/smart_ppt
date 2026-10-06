"""Cue sheets (``out/video/<deck>/cues/slide_NNN.json``): resolve, default ends,
validate. Pure; schema and rules per spec §4.

A cue is ``{"effect", "target", "at": {"s", "w"}, ["until"], ["line"],
["spotlight"], ["region"]}``. ``sentences`` everywhere is the script as lists of
words (``script.words``); ``timings`` is ``[[{word, start, end}, ...], ...]``
per sentence (the ``"sentences"`` field of ``timings/slide_NNN.json``).

Decisions where the plan left details open:
- ``resolve(cue, slide, sentences)`` takes the sentences, which the default
  underline line needs. It returns ``{ids, bbox, line, line_bbox}`` in inches
  as ``(x, y, w, h)``. A unit's lines are its members' ``line_bboxes`` /
  ``line_texts`` concatenated in member order; ``line`` is ``None`` for
  effects other than underline unless the cue sets it.
- A unit target is pointable if any member is.
- ``effective_until(sheet, timings)`` returns ``(start, end)`` seconds per cue,
  in sheet order. A focus/glide ends at the next focus or glide (sheet order
  breaks ties), even if its own ``until`` is later: a new focus replaces the
  old one. Underline/ring: their ``until`` if given, else the next *focus*
  that starts strictly later (glides do not end them). Zoom lasts
  3.0 s. Nothing later -> ``math.inf`` (hold to the end of the slide). The
  spotlight shares its focus cue's interval.
- ``validate(sheet, slide, sentences, timings=None)``: ``at``/``until``
  ordering is positional ``(s, w)``; overlaps (marks, zooms) use the same
  intervals the renderer uses. Without timings a nominal clock stands in
  (0.4 s per word, 0.35 s between sentences), which only matters for the 3 s
  zoom default. Each structurally broken cue reports its first problem only;
  timeline checks (glide source, why, concurrency) then run on the rest.
  Concurrency is counted in start order against the cues already accepted,
  so only the extra mark/zoom is flagged (it is the one that gets dropped).
  Errors are ``"cue <i>: <keyword> ..."`` in cue order.
- Why rule: the sentence's content words are ``script.norm`` (numbers as
  words) + the plan's stemmer + stopwords. The stemmer strips ``ing``/``ed``/
  ``s`` (len > 4; not after ``ss``: ``class``, ``process``), then one trailing ``e`` (if 3+ letters remain), so
  ``pages``/``page`` and ``grades``/``grading`` meet; the target's words are its text
  (or ``line_texts``) plus its unit's map ``talk`` and ``terms``. The flow
  exception needs a transition word anywhere in the anchor sentence (plan),
  not exactly at the anchor word (spec).
- ``region`` (zoom only) must be four fractions inside the target: ``x, y >= 0``,
  ``w, h > 0``, ``x + w <= 1``, ``y + h <= 1``.
- Ring needs a compact target (cue-local, so ``playable`` drops just that cue):
  the target's union bbox (inches, unpadded) may be at most ``RING_ASPECT``:1
  either way and at most ``RING_WIDTH`` of the slide width (``slide_w``; the
  renderer passes the deck's, 13.333 in = 16:9 by default). 4:1, not 3:1: this
  deck's stat targets ("15–33", "~1 min", "30/33", "14/15") measure 3.1-3.9:1
  and ring well; the swooshes were 42:1 and 57:1.
- Zoom source (``resolve(...)["source"]``, inches): the region snapped out to
  every text line it touches (union with each overlapping line box, so no
  word is cut); a region that touches no line (a photo) is padded by 4 % of
  the target's size each side, clamped to the target. No region: the target.
- ``warnings`` (never errors, so ``playable`` ignores them): dead air = the
  first cue starts more than ``DEAD_AIR_S`` after the first word of the first
  sentence that refers to a pointable unit or unit-less element (the why
  rule's word match). An empty sheet gets no warning.
"""
from __future__ import annotations

import math

from src.stt import _GENERIC_STOPWORDS
from tools.video.geometry import overlaps, union
from tools.video.script import norm

EFFECTS = ("focus", "underline", "ring", "glide", "zoom")
FOCI = ("focus", "glide")
MARKS = ("underline", "ring")
MAX_MARKS = 2
ZOOM_S = 3.0
RING_ASPECT, RING_WIDTH = 4.0, 0.25
SLIDE_W = 13.333            # inches, a 16:9 deck
ZOOM_PAD = 0.04
DEAD_AIR_S = 3.0
WORD_S, GAP_S = 0.4, 0.35   # nominal clock when no timings are given

TRANSITIONS = frozenset({"next", "then", "finally", "after", "afterwards", "second", "third",
                         "fourth", "fifth", "last", "lastly", "now", "onto"})
STOPWORDS = _GENERIC_STOPWORDS | {"s", "t", "d", "m", "ll", "re", "ve", "here", "there", "now",
                                  "just", "also", "let", "how", "what", "which", "who", "all",
                                  "these", "those", "us", "about", "up", "very", "really"}


def stem(word: str) -> str:
    w = word.lower()
    if len(w) > 4:
        for suf in ("ing", "ed", "s"):
            if w.endswith(suf) and not (suf == "s" and w.endswith("ss")):
                w = w[:-len(suf)]
                break
    return w[:-1] if w.endswith("e") and len(w) > 3 else w


def content(text: str) -> set[str]:
    """Stemmed content words of ``text`` (numbers as words, stopwords dropped)."""
    return {stem(t) for t in norm(text).replace("'", " ").split() if t not in STOPWORDS}


# ---------- slide lookups ----------

def _box(b) -> tuple:
    return (b["x"], b["y"], b["w"], b["h"])


def _units(slide) -> dict:
    return (slide.get("map") or {}).get("units") or {}


def _members(target, slide) -> list[dict]:
    if not isinstance(target, str):
        return []
    els = {e["id"]: e for e in slide.get("elements", [])}
    ids = _units(slide).get(target, [target])
    return [els[i] for i in ids if i in els]


def _unit_of(target, slide):
    if target in _units(slide):
        return target
    members = _members(target, slide)
    return members[0].get("map_unit") if members else None


def _lines(members) -> list[tuple]:
    return [(t, b) for e in members for t, b in zip(e.get("line_texts") or [], e.get("line_bboxes") or [])]


def _target_words(target, slide) -> set[str]:
    members = _members(target, slide)
    info = (slide.get("map") or {}).get("info") or {}
    unit = info.get(_unit_of(target, slide)) or {}
    texts = [e.get("text") or " ".join(e.get("line_texts") or []) for e in members]
    return content(" ".join(texts + list(unit.get("talk") or []) + list(unit.get("terms") or [])))


def _flow(slide, a, b) -> bool:
    return any(r.get("type") == "flow" and r.get("from") == a and r.get("to") == b
               for r in (slide.get("map") or {}).get("relations") or [])


# ---------- resolve ----------

def resolve(cue: dict, slide: dict, sentences) -> dict:
    """Target ids, union bbox (inches) and, for underline, the line to mark."""
    members = _members(cue["target"], slide)
    if not members:
        raise KeyError(f"unknown target {cue['target']!r}")
    lines = _lines(members)
    bbox = union([_box(e["bbox"]) for e in members])
    line = cue.get("line")
    if line is None and cue["effect"] == "underline" and lines:
        said = content(" ".join(sentences[cue["at"]["s"]]))
        shared = [len(content(text) & said) for text, _ in lines]
        line = shared.index(max(shared))        # ties -> first line
    return {"ids": [e["id"] for e in members],
            "bbox": bbox,
            "line": line,
            "line_bbox": _box(lines[line][1]) if line is not None else None,
            "source": _zoom_source(bbox, cue.get("region"), lines) if cue["effect"] == "zoom" else None}


def _zoom_source(bbox, region, lines):
    if not region:
        return bbox
    x, y, w, h = bbox
    r = (x + region[0] * w, y + region[1] * h, region[2] * w, region[3] * h)
    hit = [_box(b) for _, b in lines if overlaps(_box(b), r)]
    if hit:
        return union([r, *hit])
    px, py = ZOOM_PAD * w, ZOOM_PAD * h
    x0, y0 = max(x, r[0] - px), max(y, r[1] - py)
    x1, y1 = min(x + w, r[0] + r[2] + px), min(y + h, r[1] + r[3] + py)
    return (x0, y0, x1 - x0, y1 - y0)


# ---------- default ends ----------

def _intervals(cues, timings) -> list[tuple]:
    def t(pos):
        return timings[pos["s"]][pos["w"]]["start"]

    t0 = [t(c["at"]) for c in cues]
    order = sorted(range(len(cues)), key=lambda i: (t0[i], i))
    out = [None] * len(cues)
    for rank, i in enumerate(order):
        c, later = cues[i], order[rank + 1:]
        if c["effect"] in FOCI:
            end = next((t0[j] for j in later if cues[j]["effect"] in FOCI), math.inf)
            if "until" in c:
                end = min(end, t(c["until"]))
        elif "until" in c:
            end = t(c["until"])
        elif c["effect"] in MARKS:
            end = next((t0[j] for j in later if cues[j]["effect"] == "focus" and t0[j] > t0[i]), math.inf)
        else:   # zoom
            end = t0[i] + ZOOM_S
        out[i] = (t0[i], end)
    return out


def effective_until(sheet: dict, timings) -> list[tuple]:
    """``(start, end)`` seconds per cue, in sheet order, with the spec's defaults."""
    return _intervals(sheet["cues"], timings)


def _nominal(sentences) -> list[list[dict]]:
    out, t = [], 0.0
    for ws in sentences:
        row = []
        for w in ws:
            row.append({"word": w, "start": t, "end": t + WORD_S})
            t += WORD_S
        out.append(row)
        t += GAP_S
    return out


# ---------- validate ----------

def _pos_ok(p, sentences) -> bool:
    if not isinstance(p, dict) or not all(type(p.get(k)) is int for k in ("s", "w")):
        return False
    return 0 <= p["s"] < len(sentences) and 0 <= p["w"] < len(sentences[p["s"]])


def _region_ok(r) -> bool:
    if not (isinstance(r, list) and len(r) == 4
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in r)):
        return False
    x, y, w, h = r
    return x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= 1 and y + h <= 1


def _check(c, slide, sentences, slide_w=SLIDE_W) -> str | None:
    """First structural problem of one cue, or None."""
    if not isinstance(c, dict):
        return "not a cue object"
    eff = c.get("effect")
    if eff not in EFFECTS:
        return f"unknown effect {eff!r}"
    members = _members(c.get("target"), slide)
    if not members:
        return f"unknown target {c.get('target')!r}"
    if not any(e.get("pointable") for e in members):
        return f"target {c['target']!r} is not pointable"
    if c.get("spotlight") and eff != "focus":
        return "spotlight only accompanies a focus"
    for key in ("at", "until"):
        if (key == "at" or key in c) and not _pos_ok(c.get(key), sentences):
            return f"{key} {c.get(key)!r} is out of range of the script words"
    if "until" in c and (c["until"]["s"], c["until"]["w"]) <= (c["at"]["s"], c["at"]["w"]):
        return "until must come after at"
    n = len(_lines(members))
    if c.get("line") is not None and not (type(c["line"]) is int and 0 <= c["line"] < n):
        return f"line {c['line']!r} is out of range (target has {n} lines)"
    if eff == "underline" and n == 0:
        return f"underline target {c['target']!r} has no text lines"
    if eff == "ring":
        _, _, w, h = union([_box(e["bbox"]) for e in members])
        aspect = max(w, h) / max(min(w, h), 1e-9)
        if round(aspect, 3) > RING_ASPECT or w > RING_WIDTH * slide_w:
            return (f"ring needs a compact target — use underline ({c['target']!r} is {aspect:.1f}:1 and "
                    f"{100 * w / slide_w:.0f} % of the slide wide; max {RING_ASPECT:g}:1, {100 * RING_WIDTH:g} %)")
    if "region" in c and eff != "zoom":
        return "region only applies to a zoom"
    if "region" in c and not _region_ok(c["region"]):
        return f"region {c['region']!r} must be [x, y, w, h] fractions inside the target"
    return None


def validate(sheet, slide: dict, sentences, timings=None, slide_w=SLIDE_W) -> list[str]:
    """Errors in the cue sheet (empty list = valid)."""
    cues = sheet.get("cues") if isinstance(sheet, dict) else None
    if not isinstance(cues, list):
        return ["sheet: 'cues' must be a list"]
    if timings is not None and [len(r) for r in timings] != [len(s) for s in sentences]:
        return ["sheet: timings do not match the script words"]
    errors = []
    ok = []
    for i, c in enumerate(cues):
        problem = _check(c, slide, sentences, slide_w)
        if problem:
            errors.append((i, problem))
        else:
            ok.append(i)

    good = [cues[i] for i in ok]
    iv = _intervals(good, timings or _nominal(sentences))
    focus = None   # (target, end) of the latest focus/glide
    marks, zooms = [], []   # accepted (start, end) intervals
    for k in sorted(range(len(good)), key=lambda k: (iv[k][0], k)):
        c, i, (t0, t1) = good[k], ok[k], iv[k]
        current = focus[0] if focus and focus[1] >= t0 else None
        if c["effect"] == "glide" and current is None:
            errors.append((i, "glide needs a current focus to move from"))
        elif c["effect"] == "glide" and current == c["target"]:
            errors.append((i, f"glide target {current!r} already has the focus"))
        sent = sentences[c["at"]["s"]]
        if not content(" ".join(sent)) & _target_words(c["target"], slide):
            flow = current is not None and _flow(slide, _unit_of(current, slide), _unit_of(c["target"], slide))
            if not (flow and TRANSITIONS & set(norm(" ".join(sent)).split())):
                errors.append((i, f"why: sentence {c['at']['s']} shares no content word with {c['target']!r}"
                                  f" and is not a transition to a flow successor of the focus"))
        if c["effect"] in MARKS:
            live = 1 + sum(a <= t0 < b for a, b in marks)
            if live > MAX_MARKS:
                errors.append((i, f"concurrency: {live} marks (underline/ring) active at once, max {MAX_MARKS}"))
            else:
                marks.append((t0, t1))
        if c["effect"] == "zoom":
            if any(a <= t0 < b for a, b in zooms):
                errors.append((i, "zoom overlaps another zoom"))
            else:
                zooms.append((t0, t1))
        if c["effect"] in FOCI:
            focus = (c["target"], t1)
    return [f"cue {i}: {msg}" for i, msg in sorted(errors, key=lambda e: e[0])]


def warnings(sheet, slide: dict, sentences, timings=None) -> list[str]:
    """Advice that does not make the sheet invalid (spec §4): dead air at the start."""
    cues = sheet.get("cues") if isinstance(sheet, dict) else None
    if not isinstance(cues, list):
        return []
    rows = timings or _nominal(sentences)
    starts = [rows[c["at"]["s"]][c["at"]["w"]]["start"] for c in cues if _check(c, slide, sentences) is None]
    if not starts:
        return []
    els = slide.get("elements", [])
    targets = [u for u in _units(slide) if any(e.get("pointable") for e in _members(u, slide))]
    targets += [e["id"] for e in els if e.get("pointable") and e.get("map_unit") not in _units(slide)]
    for s, ws in enumerate(sentences):
        said = content(" ".join(ws))
        hit = next((t for t in targets if said & _target_words(t, slide)), None)
        if hit is None or not ws:
            continue
        late = min(starts) - rows[s][0]["start"]
        if late > DEAD_AIR_S:
            return [f"warning: dead air: the first cue starts {late:.1f} s after sentence {s} begins, and it "
                    f"refers to {hit!r}; open on the subject"]
        return []
    return []
