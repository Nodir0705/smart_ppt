"""Stage 1 — Parse a .pptx into a flat element index.

See CLAUDE.md "Stage 1" and the data contract section for the schema this
module emits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Iterable

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Emu


EMU_PER_INCH = 914400
ARROW_DIM_THRESHOLD_IN = 0.15  # any dimension below this with empty text → arrow
DIAGRAM_LABEL_MIN_PAIRS = 3    # ≥ N merged box-pairs on a slide → diagram_label, else stat


# ---------- data classes ----------

@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float
    units: str = "inches"

    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def contains_point(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h


@dataclass
class Element:
    id: str
    type: str
    text: str | None
    bbox: BBox
    trigger_phrases: list[str] = field(default_factory=list)
    needs_vision: bool = False
    parent_id: str | None = None


# ---------- helpers ----------

def _emu_to_in(emu: int | None) -> float:
    if emu is None:
        return 0.0
    return float(emu) / EMU_PER_INCH


def _shape_bbox(shape) -> BBox:
    return BBox(
        x=_emu_to_in(shape.left),
        y=_emu_to_in(shape.top),
        w=_emu_to_in(shape.width),
        h=_emu_to_in(shape.height),
    )


def _shape_text(shape) -> str:
    if not shape.has_text_frame:
        return ""
    return (shape.text_frame.text or "").strip()


def _is_zero_dim(bbox: BBox) -> bool:
    return bbox.w < ARROW_DIM_THRESHOLD_IN or bbox.h < ARROW_DIM_THRESHOLD_IN


def _norm(s: str) -> str:
    """Lowercased, whitespace-collapsed, punctuation-stripped (light)."""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _strip_punct(s: str) -> str:
    return re.sub(r"[^\w\s<>+%./-]", "", s)


def _phrases_for_text(text: str) -> list[str]:
    """Trigger phrases for a chunk of text.

    Keep it cheap and additive. Stage 3 semantic similarity does the heavy
    lifting; we just need a few useful surface forms.
    """
    if not text:
        return []
    out: list[str] = []
    full = _norm(text)
    if full:
        out.append(full)
    bare = _norm(_strip_punct(text))
    if bare and bare != full:
        out.append(bare)
    # individual non-empty lines (useful when a single shape holds multiple paragraphs)
    for line in text.splitlines():
        ln = _norm(line)
        if ln and ln not in out:
            out.append(ln)
    return out


def _paragraphs(shape) -> list[str]:
    if not shape.has_text_frame:
        return []
    paras: list[str] = []
    for p in shape.text_frame.paragraphs:
        txt = "".join(run.text for run in p.runs).strip()
        if txt:
            paras.append(txt)
    return paras


# Default if the deck inherits font size from a master (python-pptx returns None).
_DEFAULT_FONT_SIZE_IN = 0.222  # ~ 16 pt
_DEFAULT_LINE_SPACING = 1.2


def _paragraph_metrics(shape) -> list[dict]:
    """Per non-empty paragraph: text, line height (in), space-after (in).

    Uses each paragraph's own font size when present; falls back to the
    first non-None run font size; finally to 16 pt. Line height = font * 1.2
    unless `line_spacing` overrides (float multiplier or absolute Length).
    """
    if not shape.has_text_frame:
        return []
    out: list[dict] = []
    for p in shape.text_frame.paragraphs:
        text = "".join(r.text for r in p.runs).strip()
        if not text:
            continue
        size = p.font.size
        if size is None:
            for r in p.runs:
                if r.font.size is not None:
                    size = r.font.size
                    break
        size_in = size.inches if size is not None else _DEFAULT_FONT_SIZE_IN
        ls = p.line_spacing
        if ls is None:
            line_h = size_in * _DEFAULT_LINE_SPACING
        elif isinstance(ls, (int, float)):
            line_h = size_in * ls
        else:  # absolute Length
            try:
                line_h = ls.inches
            except AttributeError:
                line_h = size_in * _DEFAULT_LINE_SPACING
        sa = p.space_after
        sa_in = sa.inches if sa is not None else 0.0
        out.append({"text": text, "line_h": line_h, "space_after": sa_in})
    return out


def _bullet_bboxes(shape) -> list[tuple[str, BBox]]:
    """Per-paragraph (text, bbox) for a multi-line text shape.

    Walks paragraphs with their actual line heights and `space_after` rather
    than splitting the frame into equal strips — so each highlight box hugs
    the line it represents and consecutive bullets don't overlap.
    """
    metrics = _paragraph_metrics(shape)
    if not metrics:
        return []
    from pptx.enum.text import MSO_ANCHOR

    tf = shape.text_frame
    margin_top = (tf.margin_top.inches if tf.margin_top is not None else 0.0)
    margin_bottom = (tf.margin_bottom.inches if tf.margin_bottom is not None else 0.0)
    frame = _shape_bbox(shape)
    inner_h = max(0.0, frame.h - margin_top - margin_bottom)

    content_h = sum(m["line_h"] for m in metrics)
    if len(metrics) > 1:
        content_h += sum(m["space_after"] for m in metrics[:-1])

    va = tf.vertical_anchor
    if va == MSO_ANCHOR.BOTTOM:
        y0 = frame.y + frame.h - margin_bottom - content_h
    elif va == MSO_ANCHOR.MIDDLE:
        y0 = frame.y + margin_top + max(0.0, (inner_h - content_h) / 2.0)
    else:  # TOP or unspecified
        y0 = frame.y + margin_top

    out: list[tuple[str, BBox]] = []
    y = y0
    for m in metrics:
        out.append((m["text"], BBox(x=frame.x, y=y, w=frame.w, h=m["line_h"])))
        y += m["line_h"] + m["space_after"]
    return out


# ---------- pair-merging (empty container + text inside) ----------

def _find_container_text_pairs(shapes_meta: list[dict]) -> list[tuple[int, int]]:
    """Pair an empty-text shape with a text shape whose centre lies inside it.

    Returns list of (container_idx, text_idx). Each index appears at most
    once. Containers must have non-trivial area; text shapes must have text.
    """
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for ci, c in enumerate(shapes_meta):
        if ci in used:
            continue
        if c["text"]:
            continue  # container must be empty-text
        if c["is_picture"] or c["is_table"] or c["is_chart"]:
            continue
        if c["bbox"].w < 0.5 or c["bbox"].h < 0.5:
            continue  # arrows / tiny stuff
        for ti, t in enumerate(shapes_meta):
            if ti == ci or ti in used:
                continue
            if not t["text"]:
                continue
            if t["is_picture"] or t["is_table"] or t["is_chart"]:
                continue
            tcx, tcy = t["bbox"].center()
            if c["bbox"].contains_point(tcx, tcy):
                pairs.append((ci, ti))
                used.add(ci)
                used.add(ti)
                break
    return pairs


# ---------- per-shape extraction ----------

def _expand_table(shape, slide_no: int, start_idx: int) -> list[Element]:
    """Emit one table_cell element per cell, with computed per-cell bbox."""
    table = shape.table
    bbox = _shape_bbox(shape)
    col_widths = [_emu_to_in(col.width) for col in table.columns]
    row_heights = [_emu_to_in(row.height) for row in table.rows]
    # sanity: if widths sum to ~ bbox width, great; otherwise just spread evenly
    if abs(sum(col_widths) - bbox.w) > 0.1 or any(w <= 0 for w in col_widths):
        col_widths = [bbox.w / len(table.columns)] * len(table.columns)
    if abs(sum(row_heights) - bbox.h) > 0.1 or any(h <= 0 for h in row_heights):
        row_heights = [bbox.h / len(table.rows)] * len(table.rows)
    out: list[Element] = []
    idx = start_idx
    for ri, row in enumerate(table.rows):
        for ci, cell in enumerate(row.cells):
            cell_x = bbox.x + sum(col_widths[:ci])
            cell_y = bbox.y + sum(row_heights[:ri])
            text = (cell.text or "").strip()
            out.append(Element(
                id=f"s{slide_no}_e{idx}",
                type="table_cell",
                text=text or None,
                bbox=BBox(x=cell_x, y=cell_y, w=col_widths[ci], h=row_heights[ri]),
                trigger_phrases=_phrases_for_text(text),
            ))
            idx += 1
    return out


def _expand_chart(shape, slide_no: int, start_idx: int) -> list[Element]:
    """Emit one `chart` element + one `chart_category` per category label.

    Each chart_category gets its own sub-bbox (bar strip / pie sector / line
    point strip) via `chart_geom.subregions_for_chart`, so saying "the STT
    bar" highlights just the STT column instead of the whole chart.
    """
    from .chart_geom import subregions_for_chart

    bbox = _shape_bbox(shape)
    chart = shape.chart
    chart_type = str(chart.chart_type)
    categories: list[str] = []
    values: list[float] = []
    try:
        categories = [str(c) for c in chart.plots[0].categories]
        if chart.plots[0].series:
            values = [float(v or 0) for v in chart.plots[0].series[0].values]
    except Exception:
        categories = []
        values = []
    from .chart_geom import classify_chart_type
    sub_bboxes = subregions_for_chart(bbox, chart_type, len(categories), values)
    ct_simple = classify_chart_type(chart_type)
    # what to call each item: bar | slice | point
    noun = {"pie": "slice", "line": "point", "scatter": "point"}.get(ct_simple, "bar")

    out: list[Element] = []
    idx = start_idx
    chart_el = Element(
        id=f"s{slide_no}_e{idx}",
        type="chart",
        text=chart_type,
        bbox=bbox,
        trigger_phrases=_phrases_for_text(chart_type) + ["chart"],
    )
    out.append(chart_el)
    idx += 1
    for i, cat in enumerate(categories):
        cat_bbox = sub_bboxes[i] if i < len(sub_bboxes) else bbox
        # Each category gets its plain text triggers plus a "the {cat} {noun}"
        # pattern so "the STT bar" resolves to the STT column, not the whole chart.
        cat_phrases = _phrases_for_text(cat)
        labelled = f"the {cat.lower()} {noun}"
        if labelled not in cat_phrases:
            cat_phrases.append(labelled)
        out.append(Element(
            id=f"s{slide_no}_e{idx}",
            type="chart_category",
            text=cat,
            bbox=cat_bbox,
            trigger_phrases=cat_phrases,
            parent_id=chart_el.id,
        ))
        idx += 1
    return out


def _classify_text_shape(shape, slide_w: float, slide_h: float) -> str:
    """Title / bullet / body / caption from position and content."""
    bbox = _shape_bbox(shape)
    paras = _paragraphs(shape)
    if bbox.y < 1.2 and bbox.w > slide_w * 0.5 and len(paras) <= 2:
        return "title"
    if len(paras) >= 2:
        return "bullets"
    if bbox.y + bbox.h > slide_h - 0.8 and bbox.w > slide_w * 0.5:
        return "caption"
    return "body"


# ---------- main per-slide parse ----------

def _parse_slide(slide, slide_no: int, slide_w: float, slide_h: float) -> list[Element]:
    # 1) inventory shapes
    shapes_meta: list[dict] = []
    for shape in slide.shapes:
        shapes_meta.append({
            "shape": shape,
            "bbox": _shape_bbox(shape),
            "text": _shape_text(shape),
            "is_picture": shape.shape_type == MSO_SHAPE_TYPE.PICTURE,
            "is_table": getattr(shape, "has_table", False),
            "is_chart": getattr(shape, "has_chart", False),
        })

    # 2) merge container+text pairs (diagram labels or stat callouts)
    pairs = _find_container_text_pairs(shapes_meta)
    paired_idx: dict[int, tuple[int, int]] = {}  # any-idx → (container_idx, text_idx)
    for ci, ti in pairs:
        paired_idx[ci] = (ci, ti)
        paired_idx[ti] = (ci, ti)
    merged_type = "diagram_label" if len(pairs) >= DIAGRAM_LABEL_MIN_PAIRS else "stat"

    # 3) emit elements
    elements: list[Element] = []

    def next_id() -> str:
        return f"s{slide_no}_e{len(elements)}"

    consumed: set[int] = set()
    # 3a) emit merged pairs first (in original container order for stable ids)
    seen_pairs: set[tuple[int, int]] = set()
    for i, meta in enumerate(shapes_meta):
        if i not in paired_idx:
            continue
        pair = paired_idx[i]
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        ci, ti = pair
        consumed.add(ci)
        consumed.add(ti)
        container = shapes_meta[ci]
        text_shape = shapes_meta[ti]
        text = text_shape["text"]
        elements.append(Element(
            id=next_id(),
            type=merged_type,
            text=text,
            bbox=container["bbox"],  # the BOX bbox, per acceptance test
            trigger_phrases=_phrases_for_text(text),
        ))

    # 3b) tables, charts, pictures, arrows, plain text
    for i, meta in enumerate(shapes_meta):
        if i in consumed:
            continue
        shape = meta["shape"]
        bbox = meta["bbox"]
        text = meta["text"]
        if meta["is_table"]:
            elements.extend(_expand_table(shape, slide_no, len(elements)))
            continue
        if meta["is_chart"]:
            elements.extend(_expand_chart(shape, slide_no, len(elements)))
            continue
        if meta["is_picture"]:
            elements.append(Element(
                id=next_id(),
                type="image",
                text=None,
                bbox=bbox,
                trigger_phrases=[],
                needs_vision=True,
            ))
            continue
        if not text and _is_zero_dim(bbox):
            elements.append(Element(
                id=next_id(),
                type="arrow",
                text=None,
                bbox=bbox,
                trigger_phrases=[],
            ))
            continue
        if not text:
            # empty shape that wasn't paired and isn't an arrow — skip (purely decorative)
            continue
        kind = _classify_text_shape(shape, slide_w, slide_h)
        if kind == "bullets":
            # one element per paragraph, with each bbox sized to the actual line
            for line, line_bbox in _bullet_bboxes(shape):
                elements.append(Element(
                    id=next_id(),
                    type="bullet",
                    text=line,
                    bbox=line_bbox,
                    trigger_phrases=_phrases_for_text(line),
                ))
        else:
            elements.append(Element(
                id=next_id(),
                type=kind,
                text=text,
                bbox=bbox,
                trigger_phrases=_phrases_for_text(text),
            ))

    return elements


# ---------- top-level ----------

def parse_deck(path: str) -> dict:
    """Parse a .pptx and return the index dict (see CLAUDE.md data contract)."""
    prs = Presentation(path)
    slide_w = _emu_to_in(prs.slide_width)
    slide_h = _emu_to_in(prs.slide_height)
    out_slides = []
    for i, slide in enumerate(prs.slides, start=1):
        elements = _parse_slide(slide, i, slide_w, slide_h)
        out_slides.append({
            "slide_number": i,
            "elements": [_element_to_dict(e) for e in elements],
        })
    return {
        "deck": path,
        "slide_size": {"w": slide_w, "h": slide_h, "units": "inches"},
        "slides": out_slides,
    }


def _element_to_dict(e: Element) -> dict:
    return {
        "id": e.id,
        "type": e.type,
        "text": e.text,
        "bbox": asdict(e.bbox),
        "trigger_phrases": e.trigger_phrases,
        "needs_vision": e.needs_vision,
        "parent_id": e.parent_id,
    }
