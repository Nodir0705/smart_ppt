"""Parser coverage of a built deck: what the rendered slide shows that the index
does not number, and what the index numbers that the render does not back.

    python3 -m tools.coverage out/bench/<deck>/default [--out <dir>]

Inputs (all already built, nothing is rendered here): ``index.scoped.json``,
``slides/slide_NNN.png`` and the deck's soffice PDF in <deck_dir>. Output
(default ``<deck_dir>/coverage/``):

* ``slide_NNN.png`` — the render with every element's region drawn and labelled
  with its short id (``e13``): blue = text types, magenta = image, grey = arrow,
  orange = ``needs_vision``. ``line_bboxes`` are drawn when present, else ``bbox``.
* ``report.json`` / ``report.md`` — per slide (a PDF word is one of pdfium's
  whitespace tokens, tight glyph box, in slide inches):
  - unclaimed lines: PDF words (>= 2 alphanumerics) whose centre lies in no
    element region (union of ``line_bboxes`` else ``bbox``, each padded by 0.03 in),
    clustered into lines; a line whose y0 is below 92 % of the slide height is
    flagged ``footer``;
  - empty text elements: text-type elements (anything but image/arrow/chart/
    chart_category) whose region holds no PDF word centre at all;
  - counts: elements, PDF words, claimed words, unclaimed words (non-footer
    lines only), unclaimed lines (all, footer included), empty text elements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont

from src.refine_text import (_WHITESPACE, PT_PER_INCH, _norm, cluster_lines, union_bbox,
                             words_to_inches)

PAD_IN = 0.03
FOOTER_FRAC = 0.92
MIN_ALNUM = 2
NON_TEXT_TYPES = {"image", "arrow", "chart", "chart_category"}

COLOURS = {"text": (30, 90, 255), "image": (200, 0, 180), "arrow": (140, 140, 140),
           "vision": (255, 140, 0)}
# U+FFFE is pdfium's marker for a hyphenated line break ("per\ufffeexercise"): no
# whitespace follows it, so without this the two halves fuse into a two-line token.
_SEPARATORS = _WHITESPACE | {"\ufffe"}


def regions(e: dict) -> list[dict]:
    """The boxes a word centre must fall in to count as this element's, padded."""
    boxes = e.get("line_bboxes") or [e["bbox"]]
    return [{"x": b["x"] - PAD_IN, "y": b["y"] - PAD_IN,
             "w": b["w"] + 2 * PAD_IN, "h": b["h"] + 2 * PAD_IN} for b in boxes]


def _inside(b: dict, cx: float, cy: float) -> bool:
    return b["x"] <= cx <= b["x"] + b["w"] and b["y"] <= cy <= b["y"] + b["h"]


def _page_words(textpage, page_height: float) -> list[dict]:
    """pdfium's whitespace tokens with their tight glyph-union box (points, y from top).

    refine_text._assemble_words also splits on 2 pt / 3 pt glyph gaps — right for
    snapping line boxes, but it cuts large-font words into fragments, which is
    noise for a word-centre test and garbles the report text.
    """
    words: list[dict] = []
    cur: dict | None = None
    for i in range(textpage.count_chars()):
        ch = textpage.get_text_range(i, 1)
        if ch in _SEPARATORS or ch.isspace():
            if cur is not None:
                words.append(cur)
            cur = None
            continue
        left, bottom, right, top = textpage.get_charbox(i, loose=False)
        if right <= left or top <= bottom:  # degenerate box: no ink to measure
            continue
        box = {"text": ch, "x0": left, "x1": right,
               "top": page_height - top, "bottom": page_height - bottom}
        if cur is None:
            cur = box
        else:
            cur["text"] += ch
            cur["x0"], cur["x1"] = min(cur["x0"], left), max(cur["x1"], right)
            cur["top"], cur["bottom"] = min(cur["top"], box["top"]), max(cur["bottom"], box["bottom"])
    if cur is not None:
        words.append(cur)
    return words


def _pdf_pages(pdf_path: Path) -> list[tuple[float, float, list[dict]]]:
    """(width_pt, height_pt, words) per page."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return [(*page.get_size(), _page_words(page.get_textpage(), page.get_size()[1]))
                for page in pdf]
    finally:
        pdf.close()


def slide_coverage(slide: dict, words: list[dict], slide_size: dict) -> dict:
    """Coverage row for one slide from its elements and its inch-space PDF words."""
    regs = [(e, regions(e)) for e in slide["elements"]]
    backed: set[str] = set()
    unclaimed: list[dict] = []
    claimed = 0
    for w in words:
        cx, cy = (w["x0"] + w["x1"]) / 2, (w["y0"] + w["y1"]) / 2
        owners = [e["id"] for e, rs in regs if any(_inside(b, cx, cy) for b in rs)]
        if owners:
            claimed += 1
            backed.update(owners)
        elif len(_norm(w["text"])) >= MIN_ALNUM:
            unclaimed.append(w)
    footer_y = FOOTER_FRAC * slide_size["h"]
    lines = []
    for ln in cluster_lines(unclaimed):
        b = union_bbox(ln)
        lines.append({"text": " ".join(w["text"] for w in ln),
                      "bbox": {k: round(b[k], 2) for k in ("x", "y", "w", "h")},
                      "words": len(ln), "footer": b["y"] >= footer_y})
    return {
        "slide": slide["slide_number"],
        "elements": len(slide["elements"]),
        "words": len(words),
        "claimed": claimed,
        "unclaimed_words": sum(ln["words"] for ln in lines if not ln["footer"]),
        "unclaimed_lines": lines,
        "empty_text": [e["id"] for e in slide["elements"]
                       if e["type"] not in NON_TEXT_TYPES and e["id"] not in backed],
    }


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


def _colour(e: dict) -> tuple[int, int, int]:
    if e.get("needs_vision"):
        return COLOURS["vision"]
    return COLOURS.get(e["type"], COLOURS["text"])


def draw_overlay(img: Image.Image, elements: list[dict], slide_size: dict) -> Image.Image:
    """Set-of-mark view: every element's region outlined and tagged with its id."""
    out = img.convert("RGB")
    draw = ImageDraw.Draw(out)
    sx, sy = out.width / slide_size["w"], out.height / slide_size["h"]
    font = _font(max(10, out.height // 60))
    for e in elements:
        colour = _colour(e)
        width = 1 if e["type"] == "arrow" else 3
        boxes = e.get("line_bboxes") or [e["bbox"]]
        for b in boxes:
            x0, y0 = b["x"] * sx, b["y"] * sy
            draw.rectangle([x0, y0, x0 + b["w"] * sx, y0 + b["h"] * sy],
                           outline=colour, width=width)
        label = e["id"].split("_", 1)[-1]
        b = boxes[0]
        tx0, ty0, tx1, ty1 = draw.textbbox((0, 0), label, font=font)
        tw, th = tx1 - tx0 + 4, ty1 - ty0 + 4
        lx = min(max(0, b["x"] * sx), out.width - tw)
        ly = min(max(0, b["y"] * sy - th), out.height - th)
        draw.rectangle([lx, ly, lx + tw, ly + th], fill=colour)
        draw.text((lx + 2 - tx0, ly + 2 - ty0), label, fill="white", font=font)
    return out


def _md(deck: str, rows: list[dict], totals: dict) -> str:
    cols = ["slide", "elements", "words", "unclaimed_words", "unclaimed_lines", "empty_text"]
    head = "| slide | elements | words | unclaimed words | unclaimed lines | empty text elements |"
    out = [f"# Parser coverage — {deck}", "", head, "|" + "---:|" * len(cols)]
    for r in rows + [dict(totals, slide="total")]:
        cells = [r[c] if not isinstance(r[c], list) else len(r[c]) for c in cols]
        out.append("| " + " | ".join(str(c) for c in cells) + " |")
    for r in rows:
        out += ["", f"## Slide {r['slide']}", ""]
        if not r["unclaimed_lines"] and not r["empty_text"]:
            out.append("clean")
        for ln in r["unclaimed_lines"]:
            b, tag = ln["bbox"], " `footer`" if ln["footer"] else ""
            out.append(f"- ({b['x']:.2f}, {b['y']:.2f}, {b['w']:.2f}x{b['h']:.2f}) "
                       f"{ln['text'][:60]}{tag}")
        if r["empty_text"]:
            out.append(f"- empty text elements: {', '.join(r['empty_text'])}")
    return "\n".join(out) + "\n"


def run(deck_dir: Path, out_dir: Path) -> dict:
    index = json.loads((deck_dir / "index.scoped.json").read_text())
    pdfs = sorted(deck_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"no PDF in {deck_dir}")
    pages = _pdf_pages(pdfs[0])
    if len(pages) != len(index["slides"]):
        raise RuntimeError(f"PDF has {len(pages)} pages but index has {len(index['slides'])} slides")
    size = index["slide_size"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for slide, (pw, ph, page_words) in zip(index["slides"], pages):
        words = words_to_inches(page_words, size["w"] / (pw / PT_PER_INCH),
                                size["h"] / (ph / PT_PER_INCH))
        rows.append(slide_coverage(slide, words, size))
        n = slide["slide_number"]
        with Image.open(deck_dir / "slides" / f"slide_{n:03d}.png") as img:
            draw_overlay(img, slide["elements"], size).save(out_dir / f"slide_{n:03d}.png")
    totals = {k: sum(r[k] for r in rows) for k in ("elements", "words", "claimed", "unclaimed_words")}
    totals["unclaimed_lines"] = sum(len(r["unclaimed_lines"]) for r in rows)
    totals["empty_text"] = sum(len(r["empty_text"]) for r in rows)
    report = {"deck": index.get("deck"), "slides": rows, "totals": totals}
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    (out_dir / "report.md").write_text(_md(index.get("deck", deck_dir.name), rows, totals))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("deck_dir", type=Path, help="out/bench/<deck>/<variant>")
    ap.add_argument("--out", type=Path, default=None, help="default <deck_dir>/coverage")
    args = ap.parse_args(argv)
    report = run(args.deck_dir, args.out or args.deck_dir / "coverage")
    t = report["totals"]
    print(f"{len(report['slides'])} slides, {t['words']} words, {t['claimed']} claimed, "
          f"{t['unclaimed_words']} unclaimed (non-footer) in {t['unclaimed_lines']} lines, "
          f"{t['empty_text']} empty text elements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
