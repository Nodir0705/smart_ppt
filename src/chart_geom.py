"""Per-category sub-region geometry for charts (option 2).

python-pptx gives us the chart's outer bbox and its categories + values,
but NOT the plot-area geometry (where the bars/slices/points actually
live). This module estimates those positions from the chart bbox using
heuristic insets, then computes one sub-bbox per category.

Result: a `chart_category` element gets its own bbox (e.g., just the STT
column), instead of inheriting the whole chart's bbox.

The insets are tuned for LibreOffice/PowerPoint default rendering. They
will be slightly off for highly customised chart layouts (legends moved,
no title, etc.), but the resulting highlight still lands in the right
region for the test deck and most defaults.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .parser import BBox


# ---------- chart-type classification ----------

def classify_chart_type(chart_type_repr: str) -> str:
    """Map python-pptx chart_type strings to: bar, pie, line, scatter, or
    'unsupported' (radar/bubble/stock/surface — no sound per-category geometry,
    so the caller marks the chart needs_vision instead of fabricating bars)."""
    s = chart_type_repr.upper()
    if "PIE" in s or "DOUGHNUT" in s:
        return "pie"
    if "LINE" in s:
        return "line"
    if "BAR" in s or "COLUMN" in s:
        return "bar"
    if "SCATTER" in s or "BUBBLE" in s:
        return "scatter" if "SCATTER" in s else "unsupported"
    if "AREA" in s:
        return "line"
    if any(k in s for k in ("RADAR", "STOCK", "SURFACE")):
        return "unsupported"
    return "bar"  # safest default for unrecognised column/bar-like types


# ---------- plot-area inset estimation ----------

@dataclass(frozen=True)
class PlotInset:
    top: float
    bottom: float
    left: float
    right: float


# Fractions of chart bbox reserved for title (top), axis labels (bottom/left), padding (right).
# These are only the FALLBACK estimate: for column charts the bbox is normally
# measured from the rendered PNG by chart_detect; this heuristic is used when that
# measurement isn't available (no render) or fails, and for pie/line charts.
_DEFAULT_INSETS: dict[str, PlotInset] = {
    "bar":     PlotInset(top=0.18, bottom=0.10, left=0.10, right=0.03),
    "line":    PlotInset(top=0.15, bottom=0.18, left=0.13, right=0.10),
    "scatter": PlotInset(top=0.15, bottom=0.18, left=0.13, right=0.10),
    "pie":     PlotInset(top=0.15, bottom=0.05, left=0.05, right=0.05),
}


def plot_area(chart_bbox: BBox, chart_type: str) -> BBox:
    """Inset the chart bbox to get the inner plot area where data lives."""
    inset = _DEFAULT_INSETS.get(chart_type, _DEFAULT_INSETS["bar"])
    return BBox(
        x=chart_bbox.x + chart_bbox.w * inset.left,
        y=chart_bbox.y + chart_bbox.h * inset.top,
        w=chart_bbox.w * (1.0 - inset.left - inset.right),
        h=chart_bbox.h * (1.0 - inset.top - inset.bottom),
    )


# ---------- bar / column ----------

def bar_subregions(chart_bbox: BBox, n_categories: int,
                   values: list[float] | None = None,
                   min_h_frac: float = 0.04) -> list[BBox]:
    """One sub-bbox per category column.

    When `values` is given, each box is VALUE-PROPORTIONAL and anchored to the
    x-axis baseline: its height is `(value / max_value) * plot_height`, so the
    tallest bar fills the plot and short bars get short boxes (instead of every
    bar inheriting the full plot height). `min_h_frac` floors the height so a
    near-zero bar still gets a visible highlight at its baseline position.

    With no `values` (or a non-positive max), it falls back to full-height
    strips — the original behaviour — so callers without data are unaffected.
    """
    if n_categories <= 0:
        return []
    plot = plot_area(chart_bbox, "bar")
    step = plot.w / n_categories
    baseline_y = plot.y + plot.h  # bottom of the plot area = the x-axis
    vmax = max(values) if values else 0.0
    out: list[BBox] = []
    for i in range(n_categories):
        x = plot.x + i * step
        if vmax > 0 and values is not None and i < len(values):
            frac = max(values[i], 0.0) / vmax
            bar_h = max(frac * plot.h, min_h_frac * plot.h)
        else:
            bar_h = plot.h  # no value -> full-height strip (back-compat)
        out.append(BBox(x=x, y=baseline_y - bar_h, w=step, h=bar_h))
    return out


def grouped_bar_subregions(chart_bbox: BBox,
                           series_values: list[list[float]],
                           min_h_frac: float = 0.04) -> list[BBox]:
    """Distinct sub-column per (category, series) for a clustered bar chart.

    Within each category's column slot (width `step`), the `n_series` bars sit
    side by side, each `step / n_series` wide and value-proportional to the
    GLOBAL max (so heights are comparable across the whole chart). Returned in
    category-major order: cat0[s0, s1, …], cat1[s0, s1, …], … — matching how a
    clustered column chart renders. Empty when there are no series/categories.
    """
    n_series = len(series_values)
    n_categories = len(series_values[0]) if n_series else 0
    if n_series == 0 or n_categories == 0:
        return []
    plot = plot_area(chart_bbox, "bar")
    step = plot.w / n_categories
    sub_w = step / n_series
    baseline_y = plot.y + plot.h
    vmax = max((v for sv in series_values for v in sv), default=0.0)
    out: list[BBox] = []
    for ci in range(n_categories):
        for sj in range(n_series):
            val = series_values[sj][ci] if ci < len(series_values[sj]) else 0.0
            if vmax > 0:
                bar_h = max(max(val, 0.0) / vmax * plot.h, min_h_frac * plot.h)
            else:
                bar_h = plot.h
            x = plot.x + ci * step + sj * sub_w
            out.append(BBox(x=x, y=baseline_y - bar_h, w=sub_w, h=bar_h))
    return out


def stacked_bar_subregions(chart_bbox: BBox,
                           series_values: list[list[float]],
                           min_h_frac: float = 0.02) -> list[BBox]:
    """Vertical segment per (category, series) for a STACKED bar chart.

    Each category is one full-width column whose total height is proportional to
    that category's stack total vs the largest stack. Within it, series stack
    from the baseline up (series 0 at the bottom), each segment proportional to
    its share of the stack. Category-major order: cat0[s0, s1, …], cat1[…], … —
    so a given series sits at the same stack position across categories.
    """
    n_series = len(series_values)
    n_categories = len(series_values[0]) if n_series else 0
    if n_series == 0 or n_categories == 0:
        return []
    plot = plot_area(chart_bbox, "bar")
    step = plot.w / n_categories
    baseline_y = plot.y + plot.h
    totals = [sum(max(series_values[s][ci], 0.0) for s in range(n_series))
              for ci in range(n_categories)]
    gmax = max(totals, default=0.0)
    out: list[BBox] = []
    for ci in range(n_categories):
        col_h = (totals[ci] / gmax * plot.h) if gmax > 0 else plot.h
        below = 0.0  # cumulative height of segments already placed (from baseline)
        for sj in range(n_series):
            val = max(series_values[sj][ci], 0.0) if ci < len(series_values[sj]) else 0.0
            if totals[ci] > 0:
                seg_h = max(val / totals[ci] * col_h, min_h_frac * plot.h)
            else:
                seg_h = col_h / n_series
            y_top = baseline_y - below - seg_h
            out.append(BBox(x=plot.x + ci * step, y=y_top, w=step, h=seg_h))
            below += seg_h
    return out


# ---------- line ----------

def line_subregions(chart_bbox: BBox, n_categories: int,
                    point_box_in: float = 0.6) -> list[BBox]:
    """A vertical strip per category, narrowed to roughly point-marker width.

    For a line chart with multiple series, "the 0.5 point" should highlight
    a vertical slice containing both series' values at that x — that's what
    a presenter actually means. We don't try to be y-specific; a strip is
    visually intuitive and works for any number of series.
    """
    if n_categories <= 0:
        return []
    plot = plot_area(chart_bbox, "line")
    if n_categories == 1:
        # single point — center it
        w = min(plot.w, point_box_in)
        return [BBox(x=plot.x + (plot.w - w) / 2, y=plot.y, w=w, h=plot.h)]
    step = plot.w / (n_categories - 1)
    strip_w = min(step * 0.6, point_box_in)
    out: list[BBox] = []
    for i in range(n_categories):
        cx = plot.x + i * step
        out.append(BBox(x=cx - strip_w / 2, y=plot.y, w=strip_w, h=plot.h))
    return out


# ---------- pie ----------

def _arc_bbox(cx: float, cy: float, r: float,
              theta_start: float, theta_end: float) -> BBox:
    """Axis-aligned bbox of a circular SECTOR (slice including the center point).

    Angles are in radians, math convention (0 = East, CCW positive). Image
    coordinates have y pointing down, so a point at math angle θ is at
    (cx + r·cos θ, cy − r·sin θ).
    """
    # arrange so a_low ≤ a_high
    a_low = min(theta_start, theta_end)
    a_high = max(theta_start, theta_end)
    points: list[tuple[float, float]] = [(cx, cy)]  # center
    for θ in (theta_start, theta_end):
        points.append((cx + r * math.cos(θ), cy - r * math.sin(θ)))
    # cardinal angles inside [a_low, a_high] modulo 2π — these are the cos/sin extrema
    for base in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2):
        for k in (-1, 0, 1):
            θ = base + 2 * math.pi * k
            if a_low <= θ <= a_high:
                points.append((cx + r * math.cos(θ), cy - r * math.sin(θ)))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return BBox(x=min(xs), y=min(ys), w=max(xs) - min(xs), h=max(ys) - min(ys))


def pie_subregions(chart_bbox: BBox, values: list[float]) -> list[BBox]:
    """Bounding box per slice. Slices that sit next to each other will have
    overlapping bboxes (that's geometrically correct — adjacent sectors share
    a radial edge). For v1 we accept the overlap; tightening to a non-overlapping
    visual would need a polygon highlight, which the Tk Canvas doesn't do
    portably.
    """
    if not values:
        return []
    plot = plot_area(chart_bbox, "pie")
    cx = plot.x + plot.w / 2
    cy = plot.y + plot.h / 2
    r = min(plot.w, plot.h) / 2

    total = sum(values) if sum(values) > 0 else 1.0
    out: list[BBox] = []
    cum = 0.0
    for v in values:
        f_start = cum / total
        f_end = (cum + v) / total
        cum += v
        # pie convention: 0% at 12 o'clock (math angle π/2), proceeding clockwise
        # math angle decreases as the pie's percentage increases
        theta_start = math.pi / 2 - f_start * 2 * math.pi
        theta_end = math.pi / 2 - f_end * 2 * math.pi
        out.append(_arc_bbox(cx, cy, r, theta_start, theta_end))
    return out


# ---------- top-level dispatch ----------

def subregions_for_chart(chart_bbox: BBox, chart_type: str,
                         n_categories: int, values: list[float] | None = None) -> list[BBox]:
    """Return one sub-bbox per category. `values` is only used for pies."""
    ct = classify_chart_type(chart_type)
    if ct == "pie":
        return pie_subregions(chart_bbox, values or [1.0] * n_categories)
    if ct in ("line", "scatter"):
        return line_subregions(chart_bbox, n_categories)
    return bar_subregions(chart_bbox, n_categories, values)
