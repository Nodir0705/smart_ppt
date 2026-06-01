"""Refine text-element bboxes using the soffice PDF text layer (offline, exact).

The python-pptx parser gives each shape's container box, not where the glyphs
land inside it. This transform opens the cached PDF (same one the renderer
produces), reads per-word coordinates with pdfplumber, assigns words to elements
by geometric containment, and rewrites each text element's `bbox` to the tight
glyph union plus per-line `line_bboxes`. PDF points / 72 = inches — the units the
parser already emits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pdfplumber

from src.render import deck_pdf

PT_PER_INCH = 72.0

# Visual or geometry-tuned elements are left untouched (chart sub-regions are
# tuned by chart_geom; images/arrows carry no PDF text worth snapping to).
EXCLUDED_TYPES = {"chart", "chart_category", "image", "picture", "arrow"}


def words_to_inches(words: list[dict], sx: float, sy: float) -> list[dict]:
    """Convert pdfplumber word dicts (points) to inch-space word dicts.

    Input words have x0/x1/top/bottom in points. Output dicts have
    text/x0/y0/x1/y1 in inches (top-left origin).
    """
    out = []
    for w in words:
        out.append({
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


def assign_words(elements: list[tuple[str, dict]], words: list[dict]) -> dict:
    """Map element id -> list of inch-words whose center it owns.

    `elements` is a list of (element_id, shape_bbox). Each word is assigned to
    the smallest-area element whose box contains the word center; words inside no
    box are dropped.
    """
    assigned: dict[str, list[dict]] = {eid: [] for eid, _ in elements}
    for w in words:
        cx, cy = _center(w)
        best_id = None
        best_area = None
        for eid, bbox in elements:
            if _contains(bbox, cx, cy):
                area = _box_area(bbox)
                if best_area is None or area < best_area:
                    best_area = area
                    best_id = eid
        if best_id is not None:
            assigned[best_id].append(w)
    return assigned


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


def _is_refinable(e: dict) -> bool:
    return (
        e.get("type") not in EXCLUDED_TYPES
        and not e.get("needs_vision", False)
        and bool((e.get("text") or "").strip())
    )


def _apply_refinement(e: dict, words: list[dict]) -> None:
    """Mutate element `e` in place: preserve shape_bbox, set bbox/line_bboxes."""
    e["shape_bbox"] = dict(e["bbox"])
    if not words:
        e["line_bboxes"] = []
        e["bbox_source"] = "shape"
        return
    lines = cluster_lines(words)
    e["line_bboxes"] = [union_bbox(line) for line in lines]
    e["bbox"] = union_bbox(words)
    e["bbox_source"] = "pdf_text"


def refine_index(index: dict, pdf_path) -> dict:
    """Refine every text element's bbox in `index` (mutated and returned)."""
    slide_size = index["slide_size"]
    with pdfplumber.open(str(pdf_path)) as pdf:
        pages = pdf.pages
        if len(pages) != len(index["slides"]):
            raise RuntimeError(
                f"PDF has {len(pages)} pages but index has "
                f"{len(index['slides'])} slides"
            )
        for slide, page in zip(index["slides"], pages):
            sx = slide_size["w"] / (page.width / PT_PER_INCH)
            sy = slide_size["h"] / (page.height / PT_PER_INCH)
            words = words_to_inches(page.extract_words(), sx, sy)
            refinable = [(e["id"], e["bbox"]) for e in slide["elements"]
                         if _is_refinable(e)]
            assigned = assign_words(refinable, words)
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
    args = ap.parse_args(argv)

    index = json.loads(Path(args.input).read_text())
    deck = args.deck or index["deck"]
    pdf_path = deck_pdf(deck, args.pdf_dir)
    refined = refine_index(index, pdf_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(refined, indent=2))

    n = sum(1 for s in refined["slides"] for e in s["elements"]
            if e.get("bbox_source") == "pdf_text")
    total = sum(1 for s in refined["slides"] for e in s["elements"]
                if _is_refinable(e))
    print(f"refined {n}/{total} text elements → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
