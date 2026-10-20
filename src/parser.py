"""Stage 1 — Parse a .pptx into a flat element index.

See CLAUDE.md "Stage 1" and the data contract section for the schema this
module emits.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field, asdict

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE, PP_PLACEHOLDER
from pptx.oxml.shapes.groupshape import CT_GroupShape

_log = logging.getLogger(__name__)

EMU_PER_INCH = 914400
ARROW_DIM_THRESHOLD_IN = 0.15  # thinner than this in one dimension = line-thin
ARROW_MIN_LEN_IN = 0.5         # ... and at least this long = a drawn line (not a dot)
OFF_CANVAS_TOL_IN = 0.05       # a shape this far outside the slide is wholly off-canvas
OCCLUSION_TOL_IN = 0.02        # a later opaque shape reaching this close to a leaf's edges covers it
MAX_TEXTS_IN_LABEL_BOX = 2     # a label box holds a label and at most a sub-label (KPI number + caption)


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

    def is_outside(self, w: float, h: float, tol: float) -> bool:
        """Wholly outside the [0, w] x [0, h] canvas (beyond `tol`)."""
        return (self.x + self.w < -tol or self.y + self.h < -tol
                or self.x > w + tol or self.y > h + tol)

    def covers(self, other: "BBox", tol: float) -> bool:
        """`other` lies wholly inside this box (to within `tol`)."""
        return (self.x - tol <= other.x and self.y - tol <= other.y
                and other.x + other.w <= self.x + self.w + tol
                and other.y + other.h <= self.y + self.h + tol)


@dataclass
class Element:
    id: str
    type: str
    text: str | None
    bbox: BBox
    trigger_phrases: list[str] = field(default_factory=list)
    needs_vision: bool = False
    parent_id: str | None = None
    value: float | None = None   # numeric value (chart_category bars/slices)
    series: str | None = None    # series name (multi-series chart categories)
    multi_series: bool = False   # chart element flag → chart_detect skips it
    chart_title: str | None = None       # chart's own title (group_scope labels)
    series_names: list[str] | None = None  # chart series names (group_scope labels)


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


# ---------- group flattening ----------
#
# python-pptx walks `slide.shapes` only, so shapes nested inside a GROUP (every
# SmartArt diagram, and any author-grouped cluster) are never seen and vanish
# from the index. We recurse groups and map each child from the group's child
# coordinate space (chOff/chExt) into slide EMU via an affine transform, composed
# through nesting. A `_Transform` is (ax, ay, bx, by): slide = a + b * raw_emu.

_IDENTITY = (0.0, 0.0, 1.0, 1.0)
_A_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


def _group_xfrm(group_shape):
    """Return (off, ext, chOff, chExt) as (x, y) EMU tuples for a group's child
    coordinate transform. The group's own xfrm is the one carrying chOff."""
    xfrms = group_shape._element.findall(".//" + _A_NS + "xfrm")
    xfrm = next((x for x in xfrms if x.find(_A_NS + "chOff") is not None), None)
    if xfrm is None:  # malformed group — treat as identity (no scaling)
        return (0, 0), (1, 1), (0, 0), (1, 1)

    def _xy(tag, ax, ay):
        e = xfrm.find(_A_NS + tag)
        return (int(e.get(ax)), int(e.get(ay)))

    return (_xy("off", "x", "y"), _xy("ext", "cx", "cy"),
            _xy("chOff", "x", "y"), _xy("chExt", "cx", "cy"))


def _compose(parent, group_shape):
    """Compose the parent→slide transform with a group's child→group transform."""
    off, ext, choff, chext = _group_xfrm(group_shape)
    pax, pay, pbx, pby = parent
    # the group's box mapped into slide space by the parent transform
    g_x, g_y = pax + pbx * off[0], pay + pby * off[1]
    g_w, g_h = pbx * ext[0], pby * ext[1]
    sx = g_w / chext[0] if chext[0] else 1.0
    sy = g_h / chext[1] if chext[1] else 1.0
    return (g_x - choff[0] * sx, g_y - choff[1] * sy, sx, sy)


_MC_NS = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"
_P_NS = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
# python-pptx has no shape class for `p:contentPart` (pen ink): handing one to
# the shape factory raises and the whole slide came back empty. Not a shape
# here — an ink `mc:AlternateContent` resolves to its Fallback picture (what
# the renderer draws) and a bare contentPart in the tree is skipped.
_SHAPE_TAGS = tuple(t for t in CT_GroupShape._shape_tags if not t.endswith("}contentPart"))


def _is_hidden(elm) -> bool:
    """`p:cNvPr/@hidden` (Selection Pane "eye off"): the shape — or, on a
    group, its whole subtree — is not on the rendered slide."""
    cnv = elm.find("./*/" + _P_NS + "cNvPr")
    return cnv is not None and cnv.get("hidden") in ("1", "true")


def _shape_elms(sp_tree):
    """Shape elements of a `p:spTree`/`p:grpSp` in document (= z-) order.

    python-pptx's own walk skips `mc:AlternateContent`, which PowerPoint wraps
    around every text box holding an equation (Choice = the live box, Fallback =
    a picture of it), so those boxes vanished whole. Resolve each to the shapes
    of its first alternative that holds any — the Choices first, then Fallback.
    """
    for elm in sp_tree.iterchildren():
        if elm.tag in _SHAPE_TAGS:
            yield elm
        elif elm.tag == _MC_NS + "AlternateContent":
            for alt in elm.iterchildren():
                inner = [c for c in alt.iterchildren() if c.tag in _SHAPE_TAGS]
                if inner:
                    yield from inner
                    break


def _iter_flat(shapes, transform=_IDENTITY):
    """Yield (leaf_shape, transform) for every shape, recursing into groups so
    nested shapes surface with a transform that maps them to slide coordinates."""
    for elm in _shape_elms(shapes._spTree):
        if _is_hidden(elm):
            continue
        shape = shapes._shape_factory(elm)
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from _iter_flat(shape.shapes, _compose(transform, shape))
        else:
            yield shape, transform


def _rotated_aabb(x: float, y: float, w: float, h: float, deg: float) -> BBox:
    """Axis-aligned bounding box of a w×h rect rotated `deg`° about its centre —
    so a rotated shape's highlight covers its actual rendered extent, not the
    unrotated box."""
    import math
    cx, cy = x + w / 2.0, y + h / 2.0
    th = math.radians(deg)
    c, s = abs(math.cos(th)), abs(math.sin(th))
    nw = w * c + h * s
    nh = w * s + h * c
    return BBox(x=cx - nw / 2.0, y=cy - nh / 2.0, w=nw, h=nh)


def _transformed_bbox(shape, transform) -> BBox:
    ax, ay, bx, by = transform
    x = _emu_to_in(ax + bx * shape.left)
    y = _emu_to_in(ay + by * shape.top)
    w = _emu_to_in(bx * shape.width)
    h = _emu_to_in(by * shape.height)
    rot = getattr(shape, "rotation", 0.0) or 0.0
    if rot % 360 == 0:
        return BBox(x=x, y=y, w=w, h=h)
    return _rotated_aabb(x, y, w, h, rot)


# ---------- paragraph text ----------
#
# PowerPoint stores an equation as <a14:m><m:oMath>…<m:t>𝐿</m:t>…</m:oMath></a14:m>
# between a paragraph's ordinary runs. python-pptx's `paragraph.text` skips it
# ("Let  be languages") and `paragraph.runs` also skips <a:br/>, gluing a soft
# line break's two lines ("TAs:Fadi"). One reader over the raw `a:p` children,
# in document order, so every text consumer agrees.

_A14_M = "{http://schemas.microsoft.com/office/drawing/2010/main}m"
_M_T = "{http://schemas.openxmlformats.org/officeDocument/2006/math}t"


def _paragraph_text(p) -> str:
    """Text of an `a:p` element: `a:r`/`a:fld` verbatim, `a:br` as a newline (a
    soft line break is a line boundary — its lines are the chunks a presenter
    says, and `_phrases_for_text` keeps each as a phrase), and `a14:m` math as
    its `m:t` texts NFKC-normalised — math-italic letters (𝐿, U+1D43F…) to plain
    ones, U+00A0 to a space — so it is speakable."""
    parts: list[str] = []
    for child in p:
        if child.tag in (_A_NS + "r", _A_NS + "fld"):
            parts.append(child.findtext(_A_NS + "t") or "")
        elif child.tag == _A_NS + "br":
            parts.append("\n")
        elif child.tag == _A14_M:
            parts.append(unicodedata.normalize(
                "NFKC", "".join(t.text or "" for t in child.iter(_M_T))))
    return "".join(parts)


def _frame_text(text_frame) -> str:
    """Text of a text frame (shape or table cell): paragraphs joined with "\\n"."""
    return "\n".join(_paragraph_text(p._p) for p in text_frame.paragraphs).strip()


def _shape_text(shape) -> str:
    if not shape.has_text_frame:
        return ""
    return _frame_text(shape.text_frame)


_IMG_EXT_RE = re.compile(r"\.(png|jpe?g|gif|bmp|svg|tiff?|webp|emf|wmf)$", re.I)
_GENERIC_NAME_RE = re.compile(r"^(picture|image|graphic|object|content placeholder)\s*\d*$", re.I)
# PowerPoint tags every description it generates itself with this sentence.
# ponytail: English UI only; a localised PowerPoint writes the marker in its
# own language — add those strings here when a deck shows one.
_AUTO_ALT_MARKER_RE = re.compile(r"ai-generated content may be incorrect", re.I)


def _clean_alt_text(raw: str) -> str:
    """Keep author-meaningful alt text; drop junk that PowerPoint/exporters put
    in `descr`/`title` — file paths, bare image filenames, generic auto-names
    like 'Picture 1', and PowerPoint's own generated captions ("A close-up of
    a graph", words the slide never contains) — which would only add noise
    triggers. The picture then takes the OCR / needs_vision path."""
    s = (raw or "").strip()
    if not s:
        return ""
    if _AUTO_ALT_MARKER_RE.search(s):  # PowerPoint-generated description
        return ""
    if "/" in s or "\\" in s:          # a file path, not a description
        return ""
    if _IMG_EXT_RE.search(s):          # a bare image filename
        return ""
    if _GENERIC_NAME_RE.match(s):      # auto-generated placeholder name
        return ""
    return s


def _alt_text(shape) -> str:
    """Author-written alt text for a shape (its `cNvPr` `descr`, else `title`).

    For pictures this is the only deterministic, offline handle on otherwise
    text-less raster content. Namespace-agnostic: matches the `cNvPr` element by
    local name so it works across picture/shape/group non-visual property types.
    """
    for cnv in shape._element.iter():
        if cnv.tag.endswith("}cNvPr"):
            return _clean_alt_text(cnv.get("descr") or cnv.get("title") or "")
    return ""


def _ocr_picture(shape) -> str:
    """OCR a picture's raster (deterministic, offline) when it has no alt text.

    Returns "" for unsupported/linked pictures or when OCR yields no confident
    text — the caller then leaves the image needs_vision (no false pointable).
    """
    try:
        # a Movie has no `.image`; its raster is the poster frame
        blob = (getattr(shape, "image", None) or shape.poster_frame).blob
    except Exception:
        return ""  # linked or unsupported picture exposes no blob
    from .ocr import ocr_image_text
    return _clean_alt_text(ocr_image_text(blob))


def _is_zero_dim(bbox: BBox) -> bool:
    return bbox.w < ARROW_DIM_THRESHOLD_IN or bbox.h < ARROW_DIM_THRESHOLD_IN


# `a:prstGeom/@prst` values that draw a line or an arrow.
_LINE_PRESETS = frozenset(
    "line straightConnector1 bentConnector2 bentConnector3 bentConnector4 bentConnector5 "
    "curvedConnector2 curvedConnector3 curvedConnector4 curvedConnector5 "
    "rightArrow leftArrow upArrow downArrow leftRightArrow upDownArrow bentArrow uturnArrow "
    "curvedRightArrow curvedLeftArrow curvedUpArrow curvedDownArrow notchedRightArrow "
    "stripedRightArrow swooshArrow quadArrow leftRightUpArrow leftUpArrow bentUpArrow "
    "circularArrow leftCircularArrow leftRightCircularArrow".split())


def _is_line_like(shape, bbox: BBox) -> bool:
    """An empty shape that draws a line or arrow: a connector, a line/arrow
    preset, a zero-dim freeform, or a thin (< 0.15 in) filled bar at least
    0.5 in long (a divider line). Dots, small circles and zero-area boxes of
    any size are decorative, not arrows."""
    el = shape._element
    if el.tag == _P_NS + "cxnSp":
        return True
    geom = el.find(_P_NS + "spPr/" + _A_NS + "prstGeom")
    if geom is not None and geom.get("prst") in _LINE_PRESETS:
        return True
    if el.find(_P_NS + "spPr/" + _A_NS + "custGeom") is not None:
        return _is_zero_dim(bbox)
    return (_is_zero_dim(bbox) and max(bbox.w, bbox.h) >= ARROW_MIN_LEN_IN
            and _fill(shape) is not None)


# ---------- visibility (what the rendered slide shows) ----------

def _fill(shape):
    """The element painting a `p:sp`'s interior: its own `spPr` fill
    (solid/gradient/picture/pattern), else its theme `fillRef` when idx >= 1.
    None for `noFill`, or no fill and no style (transparent)."""
    sp_pr = shape._element.find(_P_NS + "spPr")
    if sp_pr is None:
        return None
    for tag in ("solidFill", "gradFill", "blipFill", "pattFill"):
        f = sp_pr.find(_A_NS + tag)
        if f is not None:
            return f
    if sp_pr.find(_A_NS + "noFill") is not None:
        return None
    ref = shape._element.find(_P_NS + "style/" + _A_NS + "fillRef")
    idx = ref.get("idx", "0") if ref is not None else "0"
    return ref if idx.isdigit() and int(idx) >= 1 else None


def _is_opaque(shape) -> bool:
    """A leaf that hides whatever lies wholly under it: a picture (ponytail:
    PNG alpha not inspected), or a `p:sp` filled with no `a:alpha` < 100 %.
    Connectors, graphic frames, transparent shapes never occlude."""
    tag = shape._element.tag
    if tag == _P_NS + "pic":
        return True
    if tag != _P_NS + "sp":
        return False
    f = _fill(shape)
    return f is not None and not any(
        (a.get("val") or "").isdigit() and int(a.get("val")) < 100000
        for a in f.iter(_A_NS + "alpha"))


def _drop_occluded(shapes_meta: list[dict], slide_no: int) -> list[dict]:
    """Keep only leaves not wholly covered by a LATER (= above, z-order is
    document order) opaque leaf — what the viewer shows, so a diagram under a
    reveal cover is neither text nor a container."""
    opaque = [_is_opaque(m["shape"]) for m in shapes_meta]
    kept: list[dict] = []
    for i, m in enumerate(shapes_meta):
        # ponytail: O(n^2) per slide; slides hold tens of shapes
        if any(opaque[j] and shapes_meta[j]["bbox"].covers(m["bbox"], OCCLUSION_TOL_IN)
               for j in range(i + 1, len(shapes_meta))):
            _log.debug("slide %d: %r %r hidden under a later opaque shape; dropped",
                       slide_no, m["shape"].name, m["text"])
            continue
        kept.append(m)
    return kept


def _norm(s: str) -> str:
    """Lowercased, whitespace-collapsed, punctuation-stripped (light)."""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _strip_punct(s: str) -> str:
    return re.sub(r"[^\w\s<>+%./-]", "", s)


# A "[symbol] number unit" stat callout, e.g. "< 1 ms", "20 %", "3.5 GB".
# Mirrors group_scope.stat_phrases so a pair classified "stat" here gets spoken
# stat phrases there. Used to type each merged pair by its OWN text instead of a
# slide-wide pair-count guess.
_STAT_TEXT_RE = re.compile(r"^\s*[<>≤≥~]?\s*\d+(?:\.\d+)?\s*[a-zA-Z%]+\s*$")


def _is_stat_text(text: str) -> bool:
    return bool(_STAT_TEXT_RE.match(text or ""))


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
        txt = _paragraph_text(p._p).strip()
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
    unless `line_spacing` overrides (float multiplier or absolute Length),
    times the paragraph's line count (each soft line break `a:br` adds one).
    """
    if not shape.has_text_frame:
        return []
    out: list[dict] = []
    for p in shape.text_frame.paragraphs:
        text = _paragraph_text(p._p).strip()
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
        line_h *= 1 + text.count("\n")  # a soft line break (a:br) adds a line
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
    A box holding more than MAX_TEXTS_IN_LABEL_BOX text centres is a
    background/cover/panel, not a label box — it pairs with nothing (a
    full-slide rectangle used to pair with the slide title).
    """
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for ci, c in enumerate(shapes_meta):
        if c["text"]:
            continue  # container must be empty-text
        if c["is_picture"] or c["is_table"] or c["is_chart"]:
            continue
        if c["bbox"].w < 0.5 or c["bbox"].h < 0.5:
            continue  # arrows / tiny stuff
        inside = [ti for ti, t in enumerate(shapes_meta)
                  if t["text"] and not (t["is_picture"] or t["is_table"] or t["is_chart"])
                  and c["bbox"].contains_point(*t["bbox"].center())]
        if len(inside) > MAX_TEXTS_IN_LABEL_BOX:
            continue
        for ti in inside:
            if ti not in used:
                pairs.append((ci, ti))
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
            # the same reader as text boxes: python-pptx `cell.text` skips
            # `a14:m` equations (a δ transition table read 9 of 12 cells empty)
            text = _frame_text(cell.text_frame)
            out.append(Element(
                id=f"s{slide_no}_e{idx}",
                type="table_cell",
                text=text or None,
                bbox=BBox(x=cell_x, y=cell_y, w=col_widths[ci], h=row_heights[ri]),
                trigger_phrases=_phrases_for_text(text),
            ))
            idx += 1
    return out


def _chart_meta(chart) -> dict:
    """The chart's own title and series names, for deck-agnostic group labels.

    Reads `chart_title.text_frame` only when it already exists (`has_text_frame`)
    — touching `.text_frame` otherwise would create one. A malformed chart part
    yields empty metadata rather than aborting the slide."""
    title = None
    names: list[str] = []
    try:
        if chart.has_title and chart.chart_title.has_text_frame:
            title = chart.chart_title.text_frame.text.strip() or None
        names = [str(se.name) for plot in chart.plots for se in plot.series]
    except Exception:
        pass
    return {"chart_title": title, "series_names": names}


def _expand_chart(shape, slide_no: int, start_idx: int) -> list[Element]:
    """Emit one `chart` element + one `chart_category` per category label.

    Each chart_category gets its own sub-bbox (bar strip / pie sector / line
    point strip) via `chart_geom.subregions_for_chart`, so saying "the STT
    bar" highlights just the STT column instead of the whole chart.
    """
    from .chart_geom import classify_chart_type, subregions_for_chart

    bbox = _shape_bbox(shape)
    chart = shape.chart
    chart_type = str(chart.chart_type)
    ct_simple = classify_chart_type(chart_type)
    meta = _chart_meta(chart)

    # Radar/bubble/stock/surface and combo (multi-plot) charts have no sound
    # per-category bar/slice geometry — emit one needs_vision chart element
    # rather than fabricating misleading sub-regions.
    if ct_simple == "unsupported" or len(chart.plots) > 1:
        return [Element(
            id=f"s{slide_no}_e{start_idx}",
            type="chart",
            text=chart_type,
            bbox=bbox,
            trigger_phrases=_phrases_for_text(chart_type) + ["chart"],
            needs_vision=True,
            **meta,
        )]

    categories: list[str] = []
    series_list: list[tuple[str, list[float]]] = []  # (name, values) per series
    try:
        categories = [str(c) for c in chart.plots[0].categories]
        for se in chart.plots[0].series:
            series_list.append((se.name, [float(v or 0) for v in se.values]))
    except Exception:
        categories = []
        series_list = []

    # Multi-series bar: one DISTINCT sub-region per (category, series) —
    # side-by-side sub-columns when clustered, stacked vertical segments when
    # stacked. Both give every series its own pointable geometry.
    if ct_simple == "bar" and len(series_list) > 1 and categories:
        from .chart_geom import grouped_bar_subregions, stacked_bar_subregions
        series_values = [sv for _, sv in series_list]
        if "STACKED" in chart_type.upper():
            sub = stacked_bar_subregions(bbox, series_values)
        else:
            sub = grouped_bar_subregions(bbox, series_values)
        return _expand_multiseries_chart(slide_no, start_idx, bbox, chart_type,
                                         categories, series_list, sub, meta)

    values = series_list[0][1] if series_list else []
    sub_bboxes = subregions_for_chart(bbox, chart_type, len(categories), values)
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
        **meta,
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
            value=values[i] if i < len(values) else None,
        ))
        idx += 1
    return out


def _expand_multiseries_chart(slide_no, start_idx, bbox, chart_type, categories,
                              series_list, sub, meta):
    """Multi-series bar: 1 chart + one chart_category per (category, series),
    each with its own distinct sub-region bbox (`sub`, in category-major order)
    and series-named triggers so "the revenue bar in Q2" resolves to it."""
    chart_el = Element(
        id=f"s{slide_no}_e{start_idx}",
        type="chart",
        text=chart_type,
        bbox=bbox,
        trigger_phrases=_phrases_for_text(chart_type) + ["chart"],
        multi_series=True,
        **meta,
    )
    out: list[Element] = [chart_el]
    idx = start_idx + 1
    k = 0
    for ci, cat in enumerate(categories):
        for sj, (sname, sv) in enumerate(series_list):
            sn = (sname or f"series {sj + 1}").strip()
            phrases = _phrases_for_text(cat)
            for p in (f"the {cat.lower()} bar in {sn.lower()}",
                      f"the {sn.lower()} {cat.lower()} bar",
                      f"{sn.lower()} {cat.lower()}"):
                if p not in phrases:
                    phrases.append(p)
            out.append(Element(
                id=f"s{slide_no}_e{idx}",
                type="chart_category",
                text=f"{cat} — {sn}",
                bbox=sub[k] if k < len(sub) else bbox,
                trigger_phrases=phrases,
                parent_id=chart_el.id,
                value=sv[ci] if ci < len(sv) else None,
                series=sn,
            ))
            idx += 1
            k += 1
    return out


def _is_title_placeholder(shape) -> bool:
    """True when the shape is an authoritative title placeholder, regardless of
    where it sits — handles centred title-slide layouts the y<1.2 heuristic
    misses (and any repositioned title)."""
    if not getattr(shape, "is_placeholder", False):
        return False
    try:
        return shape.placeholder_format.type in (
            PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE,
        )
    except (KeyError, ValueError, AttributeError):
        return False


def _classify_text_shape(shape, slide_w: float, slide_h: float) -> str:
    """Title / bullet / body / caption from position and content."""
    # Authoritative placeholder type wins; the geometry heuristic is the fallback
    # for plain text boxes (which carry no placeholder metadata).
    if _is_title_placeholder(shape):
        return "title"
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

def _parse_slide(slide, slide_no: int, slide_w: float, slide_h: float,
                 *, ocr: bool = False) -> list[Element]:
    # 1) inventory shapes (groups flattened to their leaf shapes, with each
    #    leaf's bbox mapped from group child-space into slide coordinates)
    shapes_meta: list[dict] = []
    for shape, tf in _iter_flat(slide.shapes):
        bbox = _transformed_bbox(shape, tf)
        if bbox.is_outside(slide_w, slide_h, OFF_CANVAS_TOL_IN):
            _log.debug("slide %d: %r off-canvas at %s; skipped", slide_no, shape.name, bbox)
            continue
        shapes_meta.append({
            "shape": shape,
            "bbox": bbox,
            "text": _shape_text(shape),
            # every `p:pic` — python-pptx types one in a layout placeholder
            # PLACEHOLDER and an embedded video MEDIA, not PICTURE
            "is_picture": shape._element.tag == _P_NS + "pic",
            "is_table": getattr(shape, "has_table", False),
            "is_chart": getattr(shape, "has_chart", False),
        })

    shapes_meta = _drop_occluded(shapes_meta, slide_no)

    # 2) merge container+text pairs (diagram labels or stat callouts)
    pairs = _find_container_text_pairs(shapes_meta)
    paired_idx: dict[int, tuple[int, int]] = {}  # any-idx → (container_idx, text_idx)
    for ci, ti in pairs:
        paired_idx[ci] = (ci, ti)
        paired_idx[ti] = (ci, ti)

    # 3) emit elements
    elements: list[Element] = []

    def next_id() -> str:
        return f"s{slide_no}_e{len(elements)}"

    consumed: set[int] = set()
    # 3a) emit merged pairs first (in original container order for stable ids)
    seen_pairs: set[tuple[int, int]] = set()
    for i, _meta in enumerate(shapes_meta):
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
        # Classify each pair by its OWN text: a "[symbol] number unit" callout is
        # a stat (gets spoken stat phrases downstream); anything else is a diagram
        # label. Replaces the slide-wide ">=3 pairs" guess that mislabelled both
        # KPI slides (3 numeric callouts -> diagram_label) and 2-node diagrams.
        merged_type = "stat" if _is_stat_text(text) else "diagram_label"
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
            # Author alt text makes an otherwise text-less image pointable
            # (deterministic, offline). When absent and OCR is enabled, read text
            # baked into the raster (tesseract). Else fall back to needs_vision.
            label = _alt_text(shape)
            if not label and ocr:
                label = _ocr_picture(shape)
            elements.append(Element(
                id=next_id(),
                type="image",
                text=label or None,
                bbox=bbox,
                trigger_phrases=_phrases_for_text(label),
                needs_vision=not label,
            ))
            continue
        if not text and _is_line_like(shape, bbox):
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

def parse_deck(path: str, *, ocr: bool = False) -> dict:
    """Parse a .pptx and return the index dict (see CLAUDE.md data contract).

    `ocr=True` reads text baked into alt-text-less pictures via tesseract
    (offline, deterministic per tesseract+traineddata). Default off so the pure
    parse path stays tool-free; build_index enables it when tesseract is present.
    """
    if ocr:
        from .ocr import tesseract_available
        if not tesseract_available():
            _log.warning("ocr=True but tesseract not on PATH; images stay needs_vision")
            ocr = False
    prs = Presentation(path)
    slide_w = _emu_to_in(prs.slide_width)
    slide_h = _emu_to_in(prs.slide_height)
    out_slides = []
    for i, slide in enumerate(prs.slides, start=1):
        try:
            elements = _parse_slide(slide, i, slide_w, slide_h, ocr=ocr)
        except Exception:
            # One malformed slide must not abort a whole (e.g. 100-slide) deck.
            # Emit an empty slide so numbering — and the refine page-count
            # invariant — stay intact, and surface the failure as a warning.
            _log.warning("slide %d failed to parse; emitting empty slide", i,
                         exc_info=True)
            elements = []
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
    d = {
        "id": e.id,
        "type": e.type,
        "text": e.text,
        "bbox": asdict(e.bbox),
        "trigger_phrases": e.trigger_phrases,
        "needs_vision": e.needs_vision,
        "parent_id": e.parent_id,
        "value": e.value,
    }
    # additive, multi-series only — keeps single-series output byte-identical
    if e.series is not None:
        d["series"] = e.series
    if e.multi_series:
        d["multi_series"] = True
    # additive, chart elements only — non-chart output stays byte-identical
    if e.chart_title is not None:
        d["chart_title"] = e.chart_title
    if e.series_names is not None:
        d["series_names"] = e.series_names
    return d
