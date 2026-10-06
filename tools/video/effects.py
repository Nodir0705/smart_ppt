"""HUD effects: timeline state and PIL drawing (pure, deterministic).

``Effect(kind, t0, t1, box, extra)`` is one resolved cue: ``kind`` is the cue's
``effect`` name (focus | glide | underline | ring | zoom), ``t0``/``t1`` are
slide-audio seconds (active on the half-open window t0 <= t < t1, so a focus
ending at the next focus's start never overlaps it), and ``box`` is in frame
px. ``extra`` keys:
- focus/glide: ``spotlight`` (bool). The spotlight is a flag on the bracket
  effect, not its own effect, so it follows the brackets through a glide.
- glide: ``from`` -- the box the brackets leave.
- zoom: ``lens`` -- the lens box, precomputed with ``lens_box`` (placing it
  needs the slide's occupied boxes, which ``draw`` does not have). ``box`` is
  the zoom source (``cues.resolve(...)["source"]``, padded like any box).
- underline: ``box`` is the unpadded line box; the stroke sits 3 px below it.

Building Effects from a cue sheet (``cues.resolve``/``effective_until`` +
timings + ``Letterbox``) is left to ``render.py``, which has all of those
inputs; this module stays independent of the cue schema.

``state_at`` returns a tuple of hashable ``Active`` items (the frame memo key).
``draw`` never mutates its input; it draws spotlight -> underline -> ring ->
brackets -> zoom, and the zoom lens crops the pre-effect frame so it is never
dimmed. The spotlight has no fade (none is specified); it multiplies outside
pixels by ``dim`` (spec: "dims to 55 % brightness").

Polish (2026-10-06, after the first frame audit). Pixel sizes in ``HUD`` are
at 1080p and scale with the frame height (``REF_H``).
- Every accent stroke = soft glow (the stroke's halo shape, Gaussian-blurred,
  pasted in accent at ``glow_alpha``; cached per stroke geometry) -> 2 px dark
  halo -> accent. Strokes are primitives (``("line", pts, w)`` /
  ``("arc", bbox, start, end, w)``) so the glow and the strokes share them.
- ``back`` (1 + 2.2(p-1)^3 + 1.2(p-1)^2, ~5 % overshoot) eases the focus snap
  (108 % -> 100 %, dipping to ~99.6 %) and the ring. A ring sweep past 360 deg
  draws the same pixels as 360 deg (checked in PIL), so the ring's overshoot
  shows as the same 108 % -> 100 % snap of its radii while it sweeps.
- Bracket arm = max(0.18 x min(w, h), 18 px): the spec'd upper clamp
  0.4 x min(w, h) only applies when it is >= 18 px, and 0.18 x m <= 0.4 x m,
  so it never binds. On a one-line box the vertical arms meet ("[ ]").
- Ring = an ellipse ``ring`` (1.3) x the padded box, clamped so ring + halo
  stay in the frame. A circumscribing ellipse is 1.41x; 1.12x (the brief)
  and 1.2x cut the digits of "15–33" / "~1 min" / "30/33", 1.3x clears them.
  Wide targets are refused by the validator (``cues``), so this is the
  safety net, not the fix.
- Zoom lens (``lens_box``): hard boxes (the source and every focus/glide box
  live during the zoom) are never covered; the title and then the other soft
  boxes (pointable elements) are avoided while possible. All boxes grow by a
  12 px gap. Try 2.2x, then 2.1x ... 1.6x, and take the first scale with a
  spot that touches no box, nearest the source (lens centre to source
  centre; ties: top, left). If none fits at 1.6x, the 1.6x lens goes where it
  covers, in this order: the least hard area (0 whenever possible), the least
  title area, the least other content (decoration and empty space are free),
  then nearest the source. (Area alone parked slide 3's lens on the title.) The search is exact: candidate x/y are the frame
  edges, the ideal (centred on the source) spot and every box edge +- the
  lens size; overlap areas are integer numpy products, so it is
  deterministic.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from src.geom import ease_in_out_cubic, lerp_bbox

HUD = dict(accent=(56, 212, 255), halo=(10, 14, 20), dim=0.55, feather=6, bracket_len=0.18, bracket_min=18,
           line_w=4, ring=1.3, glow=6, glow_alpha=0.55)
REF_H = 1080                # HUD pixel sizes (bracket_min, glow, LENS_GAP) are at this frame height

# animation length (s): focus snap, glide, underline draw, ring sweep, zoom fade in/out
DUR = {"focus": 0.18, "glide": 0.40, "underline": 0.35, "ring": 0.40, "zoom": 0.20}
SNAP_FROM = 1.08
ZOOM, ZOOM_MIN = 2.2, 1.6
LENS_GAP = 12


class Effect(NamedTuple):
    kind: str
    t0: float
    t1: float
    box: tuple
    extra: dict | None = None


class Active(NamedTuple):
    kind: str
    box: tuple          # current (animated) box, frame px
    p: float            # animation progress 0..1 (zoom: alpha)
    spotlight: bool = False
    lens: tuple | None = None


def _scale(box, s):
    x, y, w, h = box
    return (x + w / 2 - w * s / 2, y + h / 2 - h * s / 2, w * s, h * s)


def back(p: float) -> float:
    """Ease out with a ~5 % overshoot: back(0) = 0, back(1) = 1."""
    q = p - 1.0
    return 1.0 + 2.2 * q ** 3 + 1.2 * q ** 2


def _snap(p: float) -> float:
    return SNAP_FROM - (SNAP_FROM - 1.0) * back(p)


def state_at(t: float, plan) -> tuple:
    out = []
    for e in plan:
        if not e.t0 <= t < e.t1:
            continue
        ex = e.extra or {}
        p = min(1.0, (t - e.t0) / DUR[e.kind])
        box = tuple(e.box)
        if e.kind == "zoom":
            p = min(p, (e.t1 - t) / DUR["zoom"])
        elif p < 1.0 and e.kind == "focus":
            box = _scale(box, _snap(p))
        elif p < 1.0 and e.kind == "glide":
            box = lerp_bbox(tuple(ex["from"]), box, ease_in_out_cubic(p))
        lens = tuple(round(v) for v in ex["lens"]) if "lens" in ex else None
        out.append(Active(e.kind, box, p, bool(ex.get("spotlight")), lens))
    return tuple(out)


def arm_len(box, frame_h, style=HUD) -> float:
    return max(style["bracket_len"] * min(box[2], box[3]), style["bracket_min"] * frame_h / REF_H)


def ring_ellipse(box, frame_size, p, style=HUD) -> tuple:
    """The ring's ellipse bbox (x0, y0, x1, y1) at progress ``p``: ``ring`` x the box, snapping
    in from 108 %, shrunk where it would leave the frame (3 px margin for the halo)."""
    x, y, w, h = box
    W, H = frame_size
    k = style["ring"] / 2 * _snap(p)
    cx, cy = x + w / 2, y + h / 2
    rx = max(1.0, min(w * k, cx - 3, W - 3 - cx))
    ry = max(1.0, min(h * k, cy - 3, H - 3 - cy))
    return (cx - rx, cy - ry, cx + rx, cy + ry)


def _grow(b, g):
    return (round(b[0]) - g, round(b[1]) - g, round(b[2]) + 2 * g, round(b[3]) + 2 * g)


def _cover(boxes, xs, ys, lw, lh):
    """Overlap area of a lw x lh lens at every (xs[i], ys[j]) with ``boxes``, summed: (len(xs), len(ys))."""
    if not len(boxes):
        return np.zeros((len(xs), len(ys)), dtype=np.int64)
    b = np.asarray(boxes, dtype=np.int64)
    ox = np.clip(np.minimum(xs[:, None] + lw, b[None, :, 0] + b[None, :, 2]) - np.maximum(xs[:, None], b[None, :, 0]), 0, None)
    oy = np.clip(np.minimum(ys[:, None] + lh, b[None, :, 1] + b[None, :, 3]) - np.maximum(ys[:, None], b[None, :, 1]), 0, None)
    return ox @ oy.T


def _spot(src, frame_size, tiers, lw, lh, g):
    """Best (area per tier..., dist, y, x) for a lw x lh lens; see the module docstring."""
    W, H = frame_size
    edges = [b for t in tiers for b in t]
    c2x, c2y = 2 * round(src[0]) + round(src[2]), 2 * round(src[1]) + round(src[3])
    axes = []
    for lo, hi, size, ideal, k in ((g, W - g - lw, lw, (c2x - lw) // 2, 0), (g, H - g - lh, lh, (c2y - lh) // 2, 1)):
        hi = max(lo, hi)
        c = {lo, hi, ideal, *(e[k] - size for e in edges), *(e[k] + e[k + 2] for e in edges),
             *(e[k] for e in edges), *(e[k] + e[k + 2] - size for e in edges)}
        axes.append(np.array(sorted(v for v in c if lo <= v <= hi), dtype=np.int64))
    xs, ys = axes
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    dist = (2 * X + lw - c2x) ** 2 + (2 * Y + lh - c2y) ** 2
    cost = (*(_cover(t, xs, ys, lw, lh).ravel() for t in tiers), dist.ravel(), Y.ravel(), X.ravel())
    i = np.lexsort(cost[::-1])[0]
    return tuple(int(c[i]) for c in cost)


def lens_box(src, frame_size, soft, hard=(), title=(), scale=ZOOM, min_scale=ZOOM_MIN) -> tuple:
    """Lens (x, y, w, h) for a zoom on ``src``: never over ``src`` or ``hard`` (the live focus),
    over ``title`` and then ``soft`` (pointable boxes) only when no scale down to ``min_scale`` fits."""
    W, H = frame_size
    g = round(LENS_GAP * H / REF_H)
    tiers = [[_grow(b, g) for b in t] for t in ((src, *hard), title, soft)]
    for i in range(round((scale - min_scale) / 0.1) + 1):
        k = scale - 0.1 * i
        w, h = max(1.0, src[2] * k), max(1.0, src[3] * k)
        f = min(1.0, (W - 2 * g) / w, (H - 2 * g) / h)
        lw, lh = max(1, round(w * f)), max(1, round(h * f))
        best = _spot(src, frame_size, tiers, lw, lh, g)
        if not any(best[:3]):
            break
    return (best[-1], best[-2], lw, lh)


# ---- drawing ----------------------------------------------------------------

def _paint(img, prims, style):
    """Glow under all of ``prims``, then each one's dark halo and accent stroke."""
    r = max(1, round(style["glow"] * img.height / REF_H))
    g = _glow_mask(prims, img.size, r, style["glow_alpha"])
    if g:
        img.paste(style["accent"], g[0], g[1])
    d = ImageDraw.Draw(img)
    for kind, geo, *rest in prims:
        if kind == "arc":
            (x0, y0, x1, y1), (a0, a1, w) = geo, rest   # arc width grows inward: halo box 2 px larger
            d.arc([x0 - 2, y0 - 2, x1 + 2, y1 + 2], a0, a1, fill=style["halo"], width=w + 4)
            d.arc([x0, y0, x1, y1], a0, a1, fill=style["accent"], width=w)
        else:
            d.line(geo, fill=style["halo"], width=rest[0] + 4, joint="curve")
            d.line(geo, fill=style["accent"], width=rest[0], joint="curve")


@lru_cache(maxsize=64)
def _glow_mask(prims, size, r, alpha):
    """((x0, y0, x1, y1), L mask) of the blurred halo shapes, cropped to them + 3r."""
    pts = [q for kind, geo, *_ in prims for q in (geo if kind == "line" else (geo[:2], geo[2:]))]
    m = 3 * r + 4
    x0, y0 = max(0, math.floor(min(q[0] for q in pts) - m)), max(0, math.floor(min(q[1] for q in pts) - m))
    x1 = min(size[0], math.ceil(max(q[0] for q in pts) + m))
    y1 = min(size[1], math.ceil(max(q[1] for q in pts) + m))
    if x1 <= x0 or y1 <= y0:
        return None
    mask = Image.new("L", (x1 - x0, y1 - y0), 0)
    d = ImageDraw.Draw(mask)
    for kind, geo, *rest in prims:
        if kind == "arc":
            (a, b, c, e), (s0, s1, w) = geo, rest
            d.arc([a - x0 - 2, b - y0 - 2, c - x0 + 2, e - y0 + 2], s0, s1, fill=255, width=w + 4)
        else:
            d.line([(px - x0, py - y0) for px, py in geo], fill=255, width=rest[0] + 4, joint="curve")
    lut = [round(v * alpha) for v in range(256)]
    return (x0, y0, x1, y1), mask.filter(ImageFilter.GaussianBlur(r)).point(lut)


def _rect_pts(box):
    x, y, w, h = box
    return ((x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y))


@lru_cache(maxsize=16)
def _spot_mask(box, size, feather):
    x, y, w, h = box
    r = feather // 2
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rectangle([x - r, y - r, x + w + r, y + h + r], fill=255)
    return m.filter(ImageFilter.GaussianBlur(r))


def _spotlight(img, box, style):
    lut = [round(v * style["dim"]) for v in range(256)] * 3
    mask = _spot_mask(tuple(round(v) for v in box), img.size, style["feather"])
    return Image.composite(img, img.point(lut), mask)


def _brackets(img, src, a, style):
    x, y, w, h = a.box
    L = arm_len(a.box, img.height, style)
    x1, y1 = x + w, y + h
    _paint(img, tuple(("line", pts, style["line_w"]) for pts in (
        ((x, y + L), (x, y), (x + L, y)), ((x1 - L, y), (x1, y), (x1, y + L)),
        ((x1, y1 - L), (x1, y1), (x1 - L, y1)), ((x + L, y1), (x, y1), (x, y1 - L)))), style)
    return img


def _underline(img, src, a, style):
    if a.p > 0:
        x, y, w, h = a.box
        _paint(img, (("line", ((x, y + h + 3), (x + w * a.p, y + h + 3)), style["line_w"]),), style)
    return img


def _ring(img, src, a, style):
    if a.p > 0:
        end = -90 + 360 * min(1.0, back(a.p))
        _paint(img, (("arc", ring_ellipse(a.box, img.size, a.p, style), -90, end, style["line_w"]),), style)
    return img


def _nearest(px, py, box):
    x, y, w, h = box
    return (min(max(px, x), x + w), min(max(py, y), y + h))


def _zoom(img, src, a, style):
    x, y, w, h = a.box
    lx, ly, lw, lh = a.lens
    over = img.copy()
    crop = src.crop((round(x), round(y), round(x + w), round(y + h))).resize((lw, lh), Image.LANCZOS)
    over.paste(crop, (lx, ly))
    leader = (_nearest(lx + lw / 2, ly + lh / 2, a.box), _nearest(x + w / 2, y + h / 2, a.lens))
    _paint(over, (("line", leader, 2), ("line", _rect_pts(a.box), 2), ("line", _rect_pts(a.lens), style["line_w"])),
           style)
    return over if a.p >= 1.0 else Image.blend(img, over, a.p)


_DRAW = {"underline": (1, _underline), "ring": (2, _ring), "focus": (3, _brackets),
         "glide": (3, _brackets), "zoom": (4, _zoom)}


def draw(frame: Image.Image, state, style=HUD) -> Image.Image:
    out = frame.copy()
    for a in state:
        if a.spotlight:
            out = _spotlight(out, a.box, style)
    for a in sorted(state, key=lambda a: _DRAW[a.kind][0]):
        out = _DRAW[a.kind][1](out, frame, a, style)
    return out
