"""Agent packets: the JSON a drafter agent (spec §2) or director agent (spec §5)
reads, written by ``python3 -m tools.video packets-draft|packets-direct <deck>``.
The agents run from ``.claude/workflows/video-draft.js`` / ``video-direct.js``.

- ``out/video/<deck>/packets/draft/slide_NNN.json``: ``png``, ``message``,
  ``units``, ``relations``, ``elements``, ``prev_message``, ``next_message``.
- ``out/video/<deck>/packets/direct/slide_NNN.json``: ``png``, ``map``
  (``message``, ``units``, ``relations``, ``order``), ``elements`` (+ ``lines``
  ``[[i, text], ...]`` and ``value``), ``sentences`` ``[[[w, word], ...], ...]``
  via ``script.words``, ``vocabulary`` (``VOCAB``) and ``hash``.

Decisions where the plan left details open:
- Workflow scripts cannot read files, so the CLI decides what the agents do and
  prints it as a final ``todo: 1 3`` (or ``todo: none``) line that the
  workflow's scout agent relays: draft = slides with no ``script/<deck>/
  slide_NNN.md``; direct = slides whose cue sheet is missing, unreadable or has a
  ``"hash"`` other than the packet's.
- Both packets list only pointable elements (the validator rejects every other
  target); ``role`` is the map role, ``bbox`` is ``[x, y, w, h]`` inches rounded
  to 3 decimals. A unit is ``{members, summary, talk, terms}`` in both.
- The direct hash is sha1 of the packet minus ``png``: the content the director
  sees, not where the checkout lives (a worktree must not make every sheet
  stale). Comment lines in the script do not change it.
- Direct packets are written only for scripted slides with at least one
  sentence; others are reported and skipped. No "apply" step: agents write the
  script / cue files themselves and ``validate`` checks the sheets.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.video import ROOT, paths
from tools.video.script import sentences, words

# Spec §5 guidance, verbatim; video-direct.js carries the same text (a test checks).
GUIDANCE = ("focus on what each sentence is about, with spotlight on sentences that single one thing "
            "out. Underline the specific line when a sentence quotes or paraphrases one line of a "
            "multi-line unit. Ring a stat or number on the word where it is spoken. Glide when the "
            "narration walks a flow. Zoom when the sentence describes a detail in a photo or text too "
            "small to read. Leave slide intros/outros and pure explanation unpointed. Prefer fewer, "
            "well-timed effects to many.")

VOCAB = {  # spec §4 (schema, defaults, validator rules) + §5 (guidance)
    "version": 1,
    "sheet": {"slide": "<slide number>", "version": 1, "hash": "<this packet's hash>", "cues": ["<cue>", "..."]},
    "example": [
        {"effect": "focus", "target": "u3", "at": {"s": 2, "w": 3}, "spotlight": True},
        {"effect": "underline", "target": "s4_e11", "at": {"s": 2, "w": 7}, "line": 0},
        {"effect": "glide", "target": "u4", "at": {"s": 3, "w": 2}},
        {"effect": "ring", "target": "s4_e19", "at": {"s": 5, "w": 9}},
        {"effect": "zoom", "target": "s5_e5", "at": {"s": 4, "w": 1}, "region": [0.1, 0.4, 0.5, 0.3],
         "until": {"s": 4, "w": 12}},
    ],
    "fields": {
        "effect": "focus | glide | underline | ring | zoom",
        "target": "a map unit id (u3) or a pointable element id (s4_e11) on this slide",
        "at": "{s, w}: sentence index and word index, 0-based, as numbered in `sentences`; "
              "the effect starts at that word's start time",
        "until": "optional {s, w} after `at`; without it the effect ends by the defaults below",
        "line": "underline only, optional: index into the target's `lines` (a unit's lines are its "
                "members' lines in member order); without it the underline goes under the line that "
                "shares the most content words with the anchor sentence (ties: first line)",
        "spotlight": "focus only, optional: true dims everything outside the focus",
        "region": "zoom only, optional: [x, y, w, h] fractions of the target's bbox; "
                  "without it the zoom shows the whole target",
    },
    "effects": {
        "focus": "corner brackets snap onto the target; holds until the next focus or glide",
        "glide": "the focus brackets move from the current focus to the target",
        "underline": "a line drawn left to right under one text line; holds until the next focus",
        "ring": "an ellipse drawn around the target; holds until the next focus",
        "zoom": "a 2.2x magnifier lens of the target (or its region) in the slide's largest empty "
                "area, with a leader line to the source; lasts 3.0 s",
    },
    "rules": [
        "every target exists on this slide and is pointable",
        "`line` is within the target's lines",
        "every at/until points at an existing script word, and until comes after at",
        "glide goes from the current focus to a different target",
        "at most 1 focus, 2 marks (underline/ring) and 1 zoom are active at once",
        "spotlight only accompanies a focus",
        "why rule: the anchor sentence must refer to the target: it shares at least one content "
        "word with the target's text or its unit's talk or terms (stemmed, stopwords removed), OR "
        "the target is the next unit in a flow relation from the current focus and the sentence "
        "has a transition word (next, then, finally, after, second, third, ...)",
    ],
    "guidance": GUIDANCE,
}


def _units(m: dict) -> dict:
    info = m.get("info") or {}
    return {u: {"members": ids, "summary": (info.get(u) or {}).get("summary", ""),
                "talk": (info.get(u) or {}).get("talk", []), "terms": (info.get(u) or {}).get("terms", [])}
            for u, ids in (m.get("units") or {}).items()}


def _elements(slide: dict, full: bool) -> list[dict]:
    out = []
    for e in slide.get("elements", []):
        if not e.get("pointable"):
            continue
        d = {"id": e["id"], "role": e.get("map_role") or e.get("type"), "text": e.get("text") or "",
             "bbox": [round(e["bbox"][k], 3) for k in "xywh"]}
        if full:
            d["lines"] = [[i, t] for i, t in enumerate(e.get("line_texts") or [])]
            d["value"] = e.get("value")
        out.append(d)
    return out


def _slides(P, only):
    slides = sorted(json.loads(P.index.read_text())["slides"], key=lambda s: s["slide_number"])
    return [(k, s) for k, s in enumerate(slides) if only is None or s["slide_number"] == only], slides


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False))


def _todo(todo: list[int]) -> list[int]:
    print("todo: " + (" ".join(map(str, todo)) or "none"))
    return todo


def draft(deck: str, only: int | None = None, root: Path = ROOT) -> list[int]:
    """Write the draft packets; returns the slides with no script yet."""
    P = paths(deck, root)
    picked, slides = _slides(P, only)
    todo = []
    for k, s in picked:
        n, m = s["slide_number"], s.get("map") or {}
        msg = [(slides[j].get("map") or {}).get("message") if 0 <= j < len(slides) else None for j in (k - 1, k + 1)]
        out = P.packets / "draft" / f"slide_{n:03d}.json"
        _write(out, {"deck": deck, "slide": n, "png": str(P.slides / f"slide_{n:03d}.png"),
                     "message": m.get("message"), "units": _units(m), "relations": m.get("relations") or [],
                     "elements": _elements(s, full=False), "prev_message": msg[0], "next_message": msg[1]})
        print(f"slide {n}: {out}")
        if not (P.script / f"slide_{n:03d}.md").exists():
            todo.append(n)
    return _todo(todo)


def _sheet_hash(f: Path):
    try:
        sheet = json.loads(f.read_text())
    except (OSError, ValueError):
        return None
    return sheet.get("hash") if isinstance(sheet, dict) else None


def direct(deck: str, only: int | None = None, root: Path = ROOT) -> list[int]:
    """Write the direct packets; returns the slides whose cue sheet hash differs."""
    P = paths(deck, root)
    todo = []
    for _, s in _slides(P, only)[0]:
        n, m = s["slide_number"], s.get("map") or {}
        md = P.script / f"slide_{n:03d}.md"
        sents = sentences(md) if md.exists() else []
        if not sents:
            print(f"slide {n}: {'empty script' if md.exists() else 'no script'}, no packet")
            continue
        pkt = {"deck": deck, "slide": n,
               "map": {"message": m.get("message"), "units": _units(m), "relations": m.get("relations") or [],
                       "order": m.get("order") or []},
               "elements": _elements(s, full=True),
               "sentences": [[[w, t] for w, t in enumerate(words(x))] for x in sents],
               "vocabulary": VOCAB}
        h = hashlib.sha1(json.dumps(pkt, sort_keys=True).encode()).hexdigest()
        out = P.packets / "direct" / f"slide_{n:03d}.json"
        _write(out, {**pkt, "png": str(P.slides / f"slide_{n:03d}.png"), "hash": h})
        stale = _sheet_hash(P.cues / f"slide_{n:03d}.json") != h
        print(f"slide {n}: {'stale' if stale else 'fresh'} {out}")
        if stale:
            todo.append(n)
    return _todo(todo)
