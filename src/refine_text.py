"""Refine text-element bboxes using the soffice PDF text layer (offline, exact).

The python-pptx parser gives each shape's container box, not where the glyphs
land inside it. This transform opens the cached PDF (same one the renderer
produces), reads per-glyph coordinates with pypdfium2 (tight glyph boxes — the
exact ink extent), assembles them into words, assigns words to elements by
locating each element's text in the page's content stream (geometric containment
only as the fallback, filtered to words that spell the element's text), and rewrites each text element's `bbox` to the tight glyph
union plus per-line `line_bboxes`. PDF points / 72 = inches — the units the parser
already emits. pypdfium2's word extraction is ~29x faster than pdfminer's and its
tight glyph boxes land directly on the rasterised glyphs (see SHIFT_FRAC below).
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pypdfium2 as pdfium

from src.render import deck_pdf

PT_PER_INCH = 72.0

# PDFium reports whitespace as zero-/degenerate-box chars; treat these as word
# separators rather than glyphs.
_WHITESPACE = {" ", "\t", "\r", "\n", " ", "​"}

# Visual or geometry-tuned elements are left untouched (chart sub-regions are
# tuned by chart_geom; images/arrows carry no PDF text worth snapping to).
EXCLUDED_TYPES = {"chart", "chart_category", "image", "picture", "arrow"}


def words_to_inches(words: list[dict], sx: float, sy: float) -> list[dict]:
    """Convert pdfplumber word dicts (points) to inch-space word dicts.

    Input words have x0/x1/top/bottom in points. Output dicts have
    text/x0/y0/x1/y1 in inches (top-left origin).
    """
    out = []
    for i, w in enumerate(words):
        out.append({
            "i": i,  # PDF content-stream position: a shape's glyphs are one contiguous run
            "text": w["text"],
            "x0": w["x0"] / PT_PER_INCH * sx,
            "y0": w["top"] / PT_PER_INCH * sy,
            "x1": w["x1"] / PT_PER_INCH * sx,
            "y1": w["bottom"] / PT_PER_INCH * sy,
        })
    return out


def _center(w: dict) -> tuple[float, float]:
    return ((w["x0"] + w["x1"]) / 2.0, (w["y0"] + w["y1"]) / 2.0)


def _box_area(b: dict) -> float:
    return max(0.0, b["w"]) * max(0.0, b["h"])


def _contains(b: dict, cx: float, cy: float) -> bool:
    return (b["x"] <= cx <= b["x"] + b["w"]) and (b["y"] <= cy <= b["y"] + b["h"])


def _norm(s: str) -> str:
    """Case-folded alphanumerics only — what PDF glyphs and parsed text agree on
    (bullets, autonumber dots, quotes and spacing differ between the two)."""
    return "".join(c for c in unicodedata.normalize("NFKC", s).casefold() if c.isalnum())


def assign_words(elements: list[tuple[str, dict]], words: list[dict],
                 texts: dict[str, str] | None = None) -> dict:
    """Map element id -> list of inch-words whose center it owns.

    `elements` is a list of (element_id, shape_bbox). Each word is assigned to
    the smallest-area element whose box contains the word center; words inside no
    box are dropped. With `texts` (id -> element text), containers whose own text
    contains the word win over smaller ones that do not.
    """
    assigned: dict[str, list[dict]] = {eid: [] for eid, _ in elements}
    norm = {eid: _norm(t) for eid, t in (texts or {}).items()}
    for w in words:
        cx, cy = _center(w)
        tok = _norm(w["text"])
        best_key = None
        best_id = None
        for eid, bbox in elements:
            if _contains(bbox, cx, cy):
                mismatch = bool(tok) and eid in norm and tok not in norm[eid]
                key = (mismatch, _box_area(bbox))
                if best_key is None or key < best_key:
                    best_key = key
                    best_id = eid
        if best_id is not None:
            assigned[best_id].append(w)
    return assigned


def _gap(b: dict, words: list[dict]) -> float:
    """Distance (inches) from box `b` to the union of `words` (0 if they touch)."""
    x0, y0 = min(w["x0"] for w in words), min(w["y0"] for w in words)
    x1, y1 = max(w["x1"] for w in words), max(w["y1"] for w in words)
    dx = max(0.0, x0 - (b["x"] + b["w"]), b["x"] - x1)
    dy = max(0.0, y0 - (b["y"] + b["h"]), b["y"] - y1)
    return (dx * dx + dy * dy) ** 0.5


# inches: a free occurrence wholly outside the parser's box still competes within
# this radius (render line spacing drifts ~1.4 in from the parser's estimate down a
# long code listing); 1-3 char texts ("2", "F4") must land inside their box. Not a
# hard limit: a text left unplaced with exactly one FREE occurrence on the page is
# rescued wherever it is (see align_words).
ALIGN_MAX_GAP = 2.0
BESIDE_GAP = 0.5  # inches: bullet indent / punctuation spacing on one line


def _beside(p: dict, q: dict) -> bool:
    """Same line (vertical overlap) and horizontally within BESIDE_GAP."""
    return (min(p["y1"], q["y1"]) > max(p["y0"], q["y0"])
            and max(p["x0"] - q["x1"], q["x0"] - p["x1"]) <= BESIDE_GAP)


def align_words(elements: list[tuple[str, dict, str]], words: list[dict]) -> dict:
    """Map element id -> the stream run of words that spells its text.

    The PDF draws each shape's text as one contiguous run of the content stream,
    so an element's glyphs are found by locating its (normalised) text in the
    page's word stream — independent of where the parser thinks the box is
    (overflowing text, boxes offset from the render, full-slide label boxes).
    A match must start and end on word boundaries. A text with no exact occurrence
    is located by the free run that spells it with parts missing (the PDF leaves
    equation-box math and symbol-font runs blank; see `_spelled_blocks`), the
    survivors in order and covering at least MIN_COVER of the text. Placement
    runs in two passes over the elements, longest text first so short ones can't
    steal words:
    1. proximity — among the free occurrences the one most inside the element's
       box wins; one wholly outside it must lie within ALIGN_MAX_GAP, and 1-3
       char texts must land inside their box;
    2. page-unique rescue — a long (>= 4 char) text still unplaced with exactly
       one FREE occurrence is taken wherever the stream puts it (the parser's
       box is a guess that drifts without bound down a long placeholder:
       inherited font sizes, autofit shrink), unless a word of it sits inside
       the box of another unplaced element whose text contains it — that
       element claims it by containment.
    Elements whose text isn't found (autonumbering, glyph drift) are left to
    geometric `assign_words`.
    """
    norms = [_norm(w["text"]) for w in words]
    stream = "".join(norms)
    owner: list[int] = []
    starts, ends, pos = set(), set(), 0
    for k, n in enumerate(norms):
        if n:
            starts.add(pos)
            pos += len(n)
            ends.add(pos)
            owner += [k] * len(n)
    taken = [False] * len(words)
    spans: dict[str, list] = {}
    order = sorted(elements, key=lambda e: -len(_norm(e[2])))

    def owned_elsewhere(eid: str, spelled: str, span: list[dict]) -> bool:
        """A word of `span` lies in the box of another unplaced element whose text
        contains what the span spells."""
        return any(_contains(obox, *_center(w))
                   for oid, obox, otext in order
                   if oid != eid and oid not in spans and spelled in _norm(otext)
                   for w in span)

    for rescue in (False, True):
        for eid, bbox, text in order:
            if eid in spans:
                continue
            target = _norm(text)
            long = len(target) >= 4
            found = []  # FREE word-aligned occurrences: (inside fraction, gap, a, b)
            at = stream.find(target) if target else -1
            while at != -1:
                end = at + len(target)
                if at in starts and end in ends:
                    a, b = owner[at], owner[end - 1]
                    if not any(taken[a:b + 1]):
                        span = words[a:b + 1]
                        inside = sum(_contains(bbox, *_center(w)) for w in span) / len(span)
                        found.append((inside, _gap(bbox, span), a, b))
                at = stream.find(target, at + 1)
            if not found and long and stream.find(target) == -1:
                # no exact occurrence anywhere: the PDF may have left parts of the
                # text blank (equation boxes, symbol fonts) — locate the free run
                # that spells the rest of it in order
                masked = "".join("\0" * len(n) if taken[k] else n for k, n in enumerate(norms))
                for s, e in _gap_runs(masked, target):
                    if s not in starts or e not in ends:
                        continue
                    a, b = owner[s], owner[e - 1]
                    blocks = _spelled_blocks(stream[s:e], target)
                    if (blocks and sum(m.size for m in blocks) >= MIN_COVER * len(target)
                            and not any(taken[a:b + 1])):
                        span = words[a:b + 1]
                        inside = sum(_contains(bbox, *_center(w)) for w in span) / len(span)
                        found.append((inside, _gap(bbox, span), a, b))
            if rescue:
                # exactly one FREE occurrence on the page: the stream alone locates
                # it — unless another unplaced element's box holds a word of it
                near = []
                if long and len(found) == 1:
                    _, _, a, b = found[0]
                    if not owned_elsewhere(eid, "".join(norms[a:b + 1]), words[a:b + 1]):
                        near = found
            else:
                near = [f for f in found if f[0] > 0 or (long and f[1] <= ALIGN_MAX_GAP)]
            if near:
                _, _, a, b = min(near, key=lambda f: (-f[0], f[1]))
                taken[a:b + 1] = [True] * (b + 1 - a)
                spans[eid] = [a, b, text]
    # Grow spans over punctuation-only words beside them on the same line:
    # trailing ones only if the text has those chars ("!", ";"), so the next
    # shape keeps its bullet; then leading ones (bullets, "<").
    def free_punct(k: int, nb: int) -> bool:
        return (0 <= k < len(words) and not norms[k] and not taken[k]
                and _beside(words[k], words[nb]))

    for span in spans.values():
        while (free_punct(span[1] + 1, span[1])
               and all(c in span[2] for c in words[span[1] + 1]["text"])):
            span[1] += 1
            taken[span[1]] = True
    for span in spans.values():
        while free_punct(span[0] - 1, span[0]):
            span[0] -= 1
            taken[span[0]] = True
    return {eid: words[a:b + 1] for eid, (a, b, _) in spans.items()}


# A refined box must account for at least this fraction of the element's own text
# (alphanumeric chars); below it the glyphs found are strays (another shape's
# overflow, a lone full stop) and the shape box is the honest answer.
MIN_COVER = 0.5
MIN_LINE_H = 0.03  # inches (~2 pt): no real text line is shorter
# A glyph run is the element's when it spells at least this fraction of itself
# from the element's text, in order (the rest: autonumber/ligature drift).
MATCH_FRAC = 0.8
MIN_BLOCK = 2   # matching blocks shorter than this are coincidence
ANCHOR = 4      # and one block must be at least this long
GAP_SLACK = 2   # stream chars a stray glyph may leave between a run's surviving segments


def _spelled_blocks(run: str, target: str) -> list:
    """In-order matching blocks (run offset, target offset, size) by which `run`
    spells `target`: one block, or several when the PDF left parts of the text
    blank (equation boxes — soffice draws the words, not the math; symbol fonts;
    fields). Blocks under MIN_BLOCK chars are noise, one must reach ANCHOR, and
    together they must cover MATCH_FRAC of the run; otherwise [] — a stray."""
    blocks = [b for b in SequenceMatcher(None, run, target, autojunk=False).get_matching_blocks()
              if b.size >= MIN_BLOCK]
    if (not blocks or max(b.size for b in blocks) < ANCHOR
            or sum(b.size for b in blocks) < MATCH_FRAC * len(run)):
        return []
    return blocks


def _gap_runs(stream: str, target: str) -> list[tuple[int, int]]:
    """Stream spans (start, end) that may spell `target` with parts missing: a
    shape's glyphs are contiguous and blank math leaves none, so the surviving
    segments abut in the stream (GAP_SLACK chars for a stray glyph between)."""
    groups: list[list] = []
    for b in SequenceMatcher(None, stream, target, autojunk=False).get_matching_blocks():
        if b.size < MIN_BLOCK:
            continue
        if groups and b.a - (groups[-1][-1].a + groups[-1][-1].size) <= GAP_SLACK:
            groups[-1].append(b)
        else:
            groups.append([b])
    return [(g[0].a, g[-1].a + g[-1].size) for g in groups]


def _runs(words: list[dict]) -> list[list[dict]]:
    """Split stream-ordered words into runs of consecutive stream positions
    (hand-built words without a position are one run)."""
    if not words or "i" not in words[0]:
        return [list(words)] if words else []
    runs: list[list[dict]] = []
    for w in words:
        if runs and w["i"] == runs[-1][-1]["i"] + 1:
            runs[-1].append(w)
        else:
            runs.append([w])
    return runs


def select_text_words(text: str, words: list[dict]) -> tuple[list[dict], float]:
    """Keep the words that spell the element's own `text`; return (kept, coverage).

    Containment alone lets an element absorb any glyph inside its box — a
    full-slide label swallows every other shape's overflow. A shape's glyphs are a
    contiguous run of the PDF content stream, so each run of assigned words must
    match a not-yet-covered span of the element's text (exactly; or >= MATCH_FRAC
    of the run via the longest common block for autonumber/ligature drift; or via
    its in-order matching blocks when the PDF left parts of the text blank — see
    `_spelled_blocks`); runs that don't — or that only repeat text already
    covered — are strays. `coverage` is the fraction of the text's alphanumerics
    matched. Text with no alphanumerics keeps everything.
    """
    target = _norm(text)
    if not target:
        return list(words), 1.0
    covered = [False] * len(target)
    kept: list[dict] = []
    for run in sorted(_runs(words), key=lambda r: -len(_norm("".join(w["text"] for w in r)))):
        r = _norm("".join(w["text"] for w in run))
        if not r:
            continue
        span = None
        pos = target.find(r)
        while pos != -1:
            if not any(covered[pos:pos + len(r)]):
                span = (pos, len(r))
                break
            pos = target.find(r, pos + 1)
        spans = [span] if span else []
        if not spans and len(r) >= ANCHOR:
            m = SequenceMatcher(None, r, target, autojunk=False).find_longest_match(
                0, len(r), 0, len(target))
            if m.size >= MATCH_FRAC * len(r) and not any(covered[m.b:m.b + m.size]):
                spans = [(m.b, m.size)]
            else:  # the run may be the text with parts the PDF left blank
                blocks = _spelled_blocks(r, target)
                if blocks and not any(any(covered[b.b:b.b + b.size]) for b in blocks):
                    spans = [(b.b, b.size) for b in blocks]
        if not spans:
            continue
        for pos, n in spans:
            covered[pos:pos + n] = [True] * n
        kept.extend(run)
    order = {id(w): n for n, w in enumerate(words)}
    kept.sort(key=lambda w: order[id(w)])
    return kept, sum(covered) / len(target)


def cluster_lines(words: list[dict], y_tol: float = 0.03) -> list[list[dict]]:
    """Group inch-words into lines (top->bottom), words within a line left->right.

    A word joins a line when its vertical span overlaps the line's span, or their
    vertical centers are within `y_tol` inches.
    """
    if not words:
        return []
    ws = sorted(words, key=lambda w: ((w["y0"] + w["y1"]) / 2.0, w["x0"]))
    lines: list[list[dict]] = []
    for w in ws:
        wh = w["y1"] - w["y0"]
        cy = (w["y0"] + w["y1"]) / 2.0
        placed = False
        for line in lines:
            ly0 = min(x["y0"] for x in line)
            ly1 = max(x["y1"] for x in line)
            lh = ly1 - ly0
            lcy = (ly0 + ly1) / 2.0
            overlap = min(w["y1"], ly1) - max(w["y0"], ly0)
            min_h = min(wh, lh)
            strong_overlap = min_h > 0 and overlap > 0.5 * min_h
            if strong_overlap or abs(cy - lcy) <= y_tol:
                line.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])
    for line in lines:
        line.sort(key=lambda w: w["x0"])
    lines.sort(key=lambda line: min(w["y0"] for w in line))
    return lines


def union_bbox(words: list[dict]) -> dict | None:
    """Tight inch bbox enclosing the given inch-words, or None if empty."""
    if not words:
        return None
    x0 = min(w["x0"] for w in words)
    y0 = min(w["y0"] for w in words)
    x1 = max(w["x1"] for w in words)
    y1 = max(w["y1"] for w in words)
    return {
        "x": round(x0, 4), "y": round(y0, 4),
        "w": round(x1 - x0, 4), "h": round(y1 - y0, 4),
        "units": "inches",
    }


# With pdfminer/pdfplumber's font-metric boxes this was 0.22 — those boxes carried
# leading above the caps, so each line had to be dropped ~0.22 of its height to
# land on the rasterised glyphs. pypdfium2's TIGHT glyph boxes already ARE the ink
# extent, so no correction is needed. Re-calibrated at 150 dpi across every refined
# line on the deck: tight boxes hug the glyphs (median slack ~0.012in vs ~0.04in for
# the old font-metric boxes, equivalent clipping). Kept as a tunable knob.
SHIFT_FRAC = 0.0


def _baseline_correct(box: dict | None, shift: float | None = None) -> dict | None:
    """Drop a line box onto the rasterised glyphs by `shift` of its own height.

    `shift` defaults to SHIFT_FRAC (0.0 for pypdfium2 tight boxes); pass an
    explicit value to exercise the size-proportional correction.
    """
    if box is None:
        return None
    if shift is None:
        shift = SHIFT_FRAC
    return {**box, "y": round(box["y"] + shift * box["h"], 4)}


def _union_of_boxes(boxes: list[dict | None]) -> dict | None:
    """Bounding box of several inch boxes (used to rebuild a multi-line element box)."""
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    x0 = min(b["x"] for b in boxes)
    y0 = min(b["y"] for b in boxes)
    x1 = max(b["x"] + b["w"] for b in boxes)
    y1 = max(b["y"] + b["h"] for b in boxes)
    return {"x": round(x0, 4), "y": round(y0, 4),
            "w": round(x1 - x0, 4), "h": round(y1 - y0, 4), "units": "inches"}


def _is_refinable(e: dict) -> bool:
    return (
        e.get("type") not in EXCLUDED_TYPES
        and not e.get("needs_vision", False)
        and bool((e.get("text") or "").strip())
    )


def _apply_refinement(e: dict, words: list[dict]) -> None:
    """Mutate element `e` in place: preserve shape_bbox, set bbox/line_bboxes."""
    e["shape_bbox"] = dict(e["bbox"])
    words, coverage = select_text_words(e.get("text") or "", words)
    lines = cluster_lines(words) if coverage >= MIN_COVER else []
    line_bboxes = [_baseline_correct(union_bbox(line)) for line in lines]
    if not lines or (_norm(e.get("text") or "")
                     and max(b["h"] for b in line_bboxes) < MIN_LINE_H):
        e["line_bboxes"] = []
        e["line_texts"] = []
        e["bbox_source"] = "shape"
        return
    # Correct each line by its own height (font size), then rebuild the element
    # box from the corrected lines so multi-line elements shift correctly.
    e["line_bboxes"] = line_bboxes
    # Per-line text, parallel to line_bboxes — the handle line-level highlighting
    # uses to pick WHICH wrapped line the speaker is on (a soft-wrapped paragraph
    # stays one element, so this is the only per-line text we keep).
    e["line_texts"] = _assign_text_to_lines(e.get("text") or "", lines)
    e["bbox"] = _union_of_boxes(e["line_bboxes"])
    e["bbox_source"] = "pdf_text"


def _assign_text_to_lines(text: str, lines: list[list[dict]]) -> list[str]:
    """Reconstruct clean per-line text from the element's clean `text`.

    pypdfium2's tight glyph boxes split on a fixed 2pt gap, so the assembled
    glyph "words" are often character-fragmented ("Th i s" for "This") — fine for
    the line *geometry* (a line's union box is identical however glyphs chunk)
    but useless for word matching. Instead, split the element's CLEAN text across
    lines by each line's glyph CHARACTER COUNT: walk the clean words in order,
    filling each line's char budget before advancing. Deterministic; recovers the
    true visual wrap (verified: "This is a single long" / "paragraph that must"…).

    Falls back to the space-joined glyph words if the clean text is absent or its
    length diverges from the glyph total (ligatures/markers) — then line-level
    selection simply no-ops to the union bbox rather than mis-highlighting.
    """
    counts = [sum(len(w["text"]) for w in ln) for ln in lines]
    words = text.split()
    glyph_total = sum(counts)
    if not words or glyph_total == 0 or abs(len(text.replace(" ", "")) - glyph_total) > 2:
        return [" ".join(w["text"] for w in ln) for ln in lines]
    result: list[list[str]] = [[] for _ in lines]
    li = 0
    used = 0
    for word in words:
        if li < len(counts) - 1 and used >= counts[li]:
            li += 1
            used = 0
        result[li].append(word)
        used += len(word)
    return [" ".join(r) for r in result]


def _assemble_words(textpage, page_height: float,
                    x_tol: float = 2.0, y_tol: float = 3.0) -> list[dict]:
    """Group PDFium glyph boxes into words (point coords, y from page TOP).

    Returns dicts {text, x0, x1, top, bottom} matching pdfplumber's extract_words
    output shape. Uses TIGHT glyph boxes (loose=False), so a word's union is the
    exact ink extent — there is no font-metric leading to baseline-correct away.
    A new word starts on a whitespace char, a line change (top jumps > y_tol), or
    a horizontal gap > x_tol.
    """
    words: list[dict] = []
    cur: dict | None = None
    prev_x1: float | None = None
    for i in range(textpage.count_chars()):
        ch = textpage.get_text_range(i, 1)
        if ch in _WHITESPACE:
            if cur is not None:
                words.append(cur)
                cur, prev_x1 = None, None
            continue
        left, bottom, right, top = textpage.get_charbox(i, loose=False)
        w_top = page_height - top        # PDF bottom-left origin -> top-left origin
        w_bottom = page_height - bottom
        if cur is not None and (
            abs(w_top - cur["top"]) > y_tol
            or (prev_x1 is not None and left - prev_x1 > x_tol)
        ):
            words.append(cur)
            cur = None
        if cur is None:
            cur = {"text": ch, "x0": left, "x1": right, "top": w_top, "bottom": w_bottom}
        else:
            cur["text"] += ch
            cur["x0"] = min(cur["x0"], left)
            cur["x1"] = max(cur["x1"], right)
            cur["top"] = min(cur["top"], w_top)
            cur["bottom"] = max(cur["bottom"], w_bottom)
        prev_x1 = right
    if cur is not None:
        words.append(cur)
    return words


def _iter_pdf_pages(pdf_path):
    """Yield (width_pt, height_pt, words) per PDF page (words as _assemble_words)."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for page in pdf:
            width, height = page.get_size()
            textpage = page.get_textpage()
            yield width, height, _assemble_words(textpage, height)
    finally:
        pdf.close()


def refine_index(index: dict, pdf_path) -> dict:
    """Refine every text element's bbox in `index` (mutated and returned)."""
    slide_size = index["slide_size"]
    pages = list(_iter_pdf_pages(pdf_path))
    if len(pages) != len(index["slides"]):
        raise RuntimeError(
            f"PDF has {len(pages)} pages but index has {len(index['slides'])} slides"
        )
    for slide, (page_w, page_h, page_words) in zip(index["slides"], pages):
        sx = slide_size["w"] / (page_w / PT_PER_INCH)
        sy = slide_size["h"] / (page_h / PT_PER_INCH)
        words = words_to_inches(page_words, sx, sy)
        refinable = [e for e in slide["elements"] if _is_refinable(e)]
        # stream alignment first; containment only for what it couldn't place
        assigned = align_words([(e["id"], e["bbox"], e["text"]) for e in refinable], words)
        claimed = {id(w) for ws in assigned.values() for w in ws}
        rest = [e for e in refinable if e["id"] not in assigned]
        assigned.update(assign_words([(e["id"], e["bbox"]) for e in rest],
                                     [w for w in words if id(w) not in claimed],
                                     {e["id"]: e["text"] for e in rest}))
        for e in slide["elements"]:
            if _is_refinable(e):
                _apply_refinement(e, assigned.get(e["id"], []))
    return index


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Refine text element bboxes via the PDF text layer."
    )
    ap.add_argument("--input", default="out/index.json")
    ap.add_argument("--out", default="out/index.refined.json")
    ap.add_argument("--deck", default=None,
                    help="path to .pptx (default: index['deck'])")
    ap.add_argument("--pdf-dir", default="out",
                    help="where to cache the soffice PDF")
    ap.add_argument("--slides", default="out/slides",
                    help="rendered slide PNGs, used to measure column-chart bars")
    args = ap.parse_args(argv)

    index = json.loads(Path(args.input).read_text())
    deck = args.deck or index["deck"]
    pdf_path = deck_pdf(deck, args.pdf_dir, text_layer=True)
    refined = refine_index(index, pdf_path)

    # measure column-chart bars from the rendered slides (falls back to the
    # chart_geom heuristic per-chart when a PNG is missing or unmeasurable)
    if Path(args.slides).exists():
        from .chart_detect import refine_chart_bboxes
        refine_chart_bboxes(refined, args.slides)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(refined, indent=2))

    n = sum(1 for s in refined["slides"] for e in s["elements"]
            if e.get("bbox_source") == "pdf_text")
    total = sum(1 for s in refined["slides"] for e in s["elements"]
                if _is_refinable(e))
    nb = sum(1 for s in refined["slides"] for e in s["elements"]
             if e.get("bbox_source") == "chart_detect")
    print(f"refined {n}/{total} text elements, {nb} chart bars → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
