"""Measure column-chart bar rectangles from the rendered slide PNG.

`chart_geom` *estimates* bar positions from heuristic insets because python-pptx
exposes no plot-area geometry. This module instead MEASURES the bars directly from
the rasterised slide — the same "measure, don't guess" approach used for text
bboxes — so the highlight box matches the actual bar's width and value-proportional
height for any column-chart layout, offline and deterministically.

Algorithm (per column chart):
  1. Crop the chart bbox from the slide image.
  2. Mask "bar" pixels: saturated (coloured), not white/black/grey axis or text.
  3. Group into vertical bands; keep dense (filled) bands — drops sparse gridlines
     and axis labels that happen to be coloured.
  4. baseline = the shared bottom of the bars (the x-axis).
  5. Map detected bands to categories by matching measured heights to known values,
     so even sub-pixel bars (value≈0) get the right category slot and a
     value-proportional height. Returns one box per category, or None to signal
     "fall back to the heuristic" when the chart can't be measured confidently.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .parser import BBox

# tuning (pixel-space, robust across DPI because they're ratios or small absolutes)
_SAT_MIN = 40          # min (max-min) channel spread to count as a coloured bar/line pixel
_VAL_MIN = 45          # drop near-black axis lines/text (white is already excluded by saturation)
_FILL_MIN = 0.55       # a real bar fills most rows top→baseline; text/gridlines don't
_MIN_BAND_PX = 4       # ignore 1–3px speckle bands


def _bar_mask(crop: np.ndarray) -> np.ndarray:
    """Boolean mask of saturated (coloured) pixels — bars, not grey axis/text/bg."""
    c = crop.astype(np.int16)
    mx = c.max(axis=2)
    mn = c.min(axis=2)
    sat = mx - mn
    return (sat >= _SAT_MIN) & (mx >= _VAL_MIN)


def _bands(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous column ranges [start, end) that contain coloured pixels."""
    colcount = mask.sum(axis=0)
    out: list[tuple[int, int]] = []
    start = None
    for x in range(mask.shape[1]):
        if colcount[x] > 0 and start is None:
            start = x
        elif colcount[x] == 0 and start is not None:
            out.append((start, x))
            start = None
    if start is not None:
        out.append((start, mask.shape[1]))
    return out


def detect_bars(image: np.ndarray, chart_px: tuple[int, int, int, int],
                values: list[float]) -> list[BBox] | None:
    """Detect one bar box (in IMAGE pixels) per category, or None to fall back.

    `image` is the full RGB slide (H, W, 3). `chart_px` is the chart bbox as
    integer pixels (x0, y0, x1, y1). `values` are the per-category numeric values
    in category order. Returns inch-free pixel BBoxes (x, y, w, h) in image space.
    """
    if not values or max(values) <= 0:
        return None
    x0, y0, x1, y1 = chart_px
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    mask = _bar_mask(crop)

    # candidate bands with a high fill ratio (top→bottom solidly coloured)
    cands = []  # (cx_px, w_px, top_px, bottom_px)
    for bs, be in _bands(mask):
        if be - bs < _MIN_BAND_PX:
            continue
        sub = mask[:, bs:be]
        rows = np.where(sub.any(axis=1))[0]
        if len(rows) == 0:
            continue
        top, bot = int(rows[0]), int(rows[-1])
        # fill ratio: fraction of the band's vertical span that is coloured
        fill = sub[top:bot + 1].mean()
        if fill < _FILL_MIN:
            continue
        cands.append((bs + (be - bs) / 2.0, be - bs, top, bot))
    if len(cands) < 1:
        return None

    # baseline = shared bottom of the bars (bars rest on the x-axis)
    baseline = max(c[3] for c in cands)
    # value→pixel scale from the tallest detected bar = max(values)
    tallest = max(cands, key=lambda c: baseline - c[2])
    tallest_h = baseline - tallest[2]
    vmax = max(values)
    if tallest_h <= 0:
        return None
    scale = tallest_h / vmax  # pixels per value unit

    # assign each detected band to a category by matching measured height to value
    remaining = list(range(len(values)))
    assigned: list[tuple[int, float]] = []  # (category_index, cx_px)
    for cx, _w, top, _bot in cands:
        est = (baseline - top) / scale
        j = min(remaining, key=lambda k: abs(values[k] - est)) if remaining else None
        if j is None:
            continue
        remaining.remove(j)
        assigned.append((j, cx))
    if len(assigned) < 1:
        return None

    # fit the evenly-spaced category grid: cx = origin + step * index
    width_px = float(np.median([c[1] for c in cands]))
    if len(assigned) >= 2:
        idx = np.array([a[0] for a in assigned], dtype=float)
        cxs = np.array([a[1] for a in assigned], dtype=float)
        A = np.vstack([idx, np.ones_like(idx)]).T
        step, origin = np.linalg.lstsq(A, cxs, rcond=None)[0]
    else:
        # single bar: anchor the grid on it, infer step from the plot width
        j0, cx0 = assigned[0]
        step = (x1 - x0) / max(len(values), 1)
        origin = cx0 - step * j0

    min_h = max(_MIN_BAND_PX, 0.02 * tallest_h)  # keep tiny bars visible
    out: list[BBox] = []
    for i, v in enumerate(values):
        cx = origin + step * i
        h = max(v * scale, min_h)
        out.append(BBox(
            x=float(x0 + cx - width_px / 2.0),
            y=float(y0 + baseline - h),
            w=float(width_px),
            h=float(h),
        ))
    return out


def detect_line_points(image: np.ndarray, chart_px: tuple[int, int, int, int],
                       n_categories: int, legend_clip_frac: float = 0.12
                       ) -> list[BBox] | None:
    """Detect one box per category for a line/scatter chart (in IMAGE pixels).

    Each box is centred on the category's marker column and spans the vertical
    extent of ALL series' data points there — i.e. from the top line's marker to
    the bottom line's marker at that x. Colour-agnostic: it locates markers by
    coloured-pixel density, so it works for any series palette. Returns None to
    fall back to the heuristic when the chart can't be measured.

    `legend_clip_frac` drops the bottom strip of the chart (where the legend's
    series-coloured swatches live) so they aren't mistaken for data points.
    """
    if n_categories <= 0:
        return None
    x0, y0, x1, y1 = chart_px
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    mask = _bar_mask(crop)
    if legend_clip_frac > 0:
        mask[int((1.0 - legend_clip_frac) * mask.shape[0]):, :] = False

    xcols = np.where(mask.any(axis=0))[0]
    if len(xcols) < 2:
        return None
    left, right = int(xcols[0]), int(xcols[-1])
    if right <= left:
        return None
    step = (right - left) / max(n_categories - 1, 1)
    box_half = max(_MIN_BAND_PX, int(0.13 * step))     # drawn box width
    search_half = max(box_half, int(0.25 * step))      # wider y-extent search
    refine = max(_MIN_BAND_PX, int(0.20 * step))       # x snap radius
    colden = mask.sum(axis=0)

    out: list[BBox] = []
    for i in range(n_categories):
        x_est = left + round(i * step)
        # snap to the densest column nearby — the marker (line segments between
        # markers anti-alias below threshold, so the marker is the density peak)
        lo, hi = max(0, x_est - refine), min(mask.shape[1], x_est + refine + 1)
        seg = colden[lo:hi]
        xc = lo + int(np.argmax(seg)) if seg.size and seg.max() > 0 else x_est
        # search a WIDER window for the vertical extent so a sibling series' marker
        # a few px off in x (and the sloped line between points) is still captured…
        slo, shi = max(0, xc - search_half), xc + search_half + 1
        rows = np.where(mask[:, slo:shi].any(axis=1))[0]
        if len(rows) == 0:
            return None  # a missing point means the measurement is unreliable
        out.append(BBox(
            x=float(x0 + xc - box_half),       # …but draw the box at the narrow width
            y=float(y0 + int(rows[0])),
            w=float(2 * box_half),
            h=float(int(rows[-1]) - int(rows[0])),
        ))
    return out


# ---------- index-level application ----------

def refine_chart_bboxes(index: dict, slides_dir: str | Path) -> dict:
    """Override column-chart `chart_category` bboxes with PNG-measured bars.

    For each bar chart, measure its bars from the rendered slide PNG and rewrite
    each category element's bbox (in inches). Pie/line charts and any chart that
    can't be measured keep the heuristic bbox from the parser. Mutates and returns
    `index`. Requires Pillow; a missing/unreadable PNG just skips that slide.
    """
    from PIL import Image
    from .chart_geom import classify_chart_type

    slides_dir = Path(slides_dir)
    slide_w = index["slide_size"]["w"]
    for slide in index["slides"]:
        sno = slide["slide_number"]
        png = slides_dir / f"slide_{sno:03d}.png"
        # group category children under their parent chart, keyed by chart kind
        charts = {e["id"]: e for e in slide["elements"] if e["type"] == "chart"}
        children: dict[str, list[dict]] = {}
        for e in slide["elements"]:
            if e["type"] == "chart_category" and e.get("parent_id") in charts:
                children.setdefault(e["parent_id"], []).append(e)
        if not children or not png.exists():
            continue
        try:
            image = np.asarray(Image.open(png).convert("RGB"))
        except Exception:
            continue
        dpi = image.shape[1] / slide_w
        for chart_id, cats in children.items():
            kind = classify_chart_type(charts[chart_id].get("text") or "")
            cb = charts[chart_id]["bbox"]
            chart_px = (int(cb["x"] * dpi), int(cb["y"] * dpi),
                        int((cb["x"] + cb["w"]) * dpi), int((cb["y"] + cb["h"]) * dpi))
            if kind == "bar":
                values = [c.get("value") or 0.0 for c in cats]
                boxes = detect_bars(image, chart_px, values)
            elif kind in ("line", "scatter"):
                boxes = detect_line_points(image, chart_px, len(cats))
            else:
                boxes = None  # pie etc. keep the heuristic
            if boxes is None or len(boxes) != len(cats):
                continue  # keep heuristic bbox
            for cat, box in zip(cats, boxes):
                cat["bbox"] = {
                    "x": round(box.x / dpi, 4), "y": round(box.y / dpi, 4),
                    "w": round(box.w / dpi, 4), "h": round(box.h / dpi, 4),
                    "units": "inches",
                }
                cat["bbox_source"] = "chart_detect"
    return index
