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
  needs the slide's occupied boxes, which ``draw`` does not have).
- underline: ``box`` is the unpadded line box; the stroke sits 3 px below it.

Building Effects from a cue sheet (``cues.resolve``/``effective_until`` +
timings + ``Letterbox``) is left to ``render.py``, which has all of those
inputs; this module stays independent of the cue schema.

``state_at`` returns a tuple of hashable ``Active`` items (the frame memo key).
``draw`` never mutates its input; it draws spotlight -> underline -> ring ->
brackets -> zoom, and the zoom lens crops the pre-effect frame so it is never
dimmed. The spotlight has no fade (none is specified); it multiplies outside
pixels by ``dim`` (spec: "dims to 55 % brightness").
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import NamedTuple

from PIL import Image, ImageDraw, ImageFilter

from src.geom import ease_in_out_cubic, lerp_bbox
from tools.video.geometry import empty_region

HUD = dict(accent=(56, 212, 255), halo=(10, 14, 20), dim=0.55, feather=6, bracket_len=0.18, line_w=4)

# animation length (s): focus snap, glide, underline draw, ring sweep, zoom fade in/out
DUR = {"focus": 0.18, "glide": 0.40, "underline": 0.35, "ring": 0.40, "zoom": 0.20}
SNAP_FROM = 1.08
ZOOM = 2.2


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
            box = _scale(box, SNAP_FROM - (SNAP_FROM - 1.0) * p)
        elif p < 1.0 and e.kind == "glide":
            box = lerp_bbox(tuple(ex["from"]), box, ease_in_out_cubic(p))
        lens = tuple(round(v) for v in ex["lens"]) if "lens" in ex else None
        out.append(Active(e.kind, box, p, bool(ex.get("spotlight")), lens))
    return tuple(out)


def lens_box(src, frame_size, occupied, scale=ZOOM) -> tuple:
    """Lens for a zoom on ``src``: ``scale``x its size (shrunk to fit the frame),
    placed by ``empty_region``. Pass the slide's boxes (and ``src``) as occupied."""
    W, H = frame_size
    w, h = max(1, src[2] * scale), max(1, src[3] * scale)
    k = min(1.0, W / w, H / h)
    lw, lh = round(w * k), round(h * k)
    return (*empty_region(frame_size, occupied, lw, lh), lw, lh)


# ---- drawing ----------------------------------------------------------------

def _stroke(d, pts, style, width=None):
    """A polyline in accent colour over a 2 px dark halo."""
    w = width or style["line_w"]
    d.line(pts, fill=style["halo"], width=w + 4, joint="curve")
    d.line(pts, fill=style["accent"], width=w, joint="curve")


def _rect_pts(box):
    x, y, w, h = box
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]


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
    L = style["bracket_len"] * min(w, h)
    x1, y1 = x + w, y + h
    d = ImageDraw.Draw(img)
    for pts in ([(x, y + L), (x, y), (x + L, y)], [(x1 - L, y), (x1, y), (x1, y + L)],
                [(x1, y1 - L), (x1, y1), (x1 - L, y1)], [(x + L, y1), (x, y1), (x, y1 - L)]):
        _stroke(d, pts, style)
    return img


def _underline(img, src, a, style):
    if a.p > 0:
        x, y, w, h = a.box
        _stroke(ImageDraw.Draw(img), [(x, y + h + 3), (x + w * a.p, y + h + 3)], style)
    return img


def _ring(img, src, a, style):
    if a.p > 0:
        x, y, w, h = a.box
        cx, cy = x + w / 2, y + h / 2
        rx, ry = w / math.sqrt(2) + style["line_w"], h / math.sqrt(2) + style["line_w"]
        end = -90 + 360 * a.p
        d = ImageDraw.Draw(img)   # arc width grows inward: halo box 2 px larger, 4 px wider
        d.arc([cx - rx - 2, cy - ry - 2, cx + rx + 2, cy + ry + 2], -90, end,
              fill=style["halo"], width=style["line_w"] + 4)
        d.arc([cx - rx, cy - ry, cx + rx, cy + ry], -90, end, fill=style["accent"], width=style["line_w"])
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
    d = ImageDraw.Draw(over)
    leader = [_nearest(lx + lw / 2, ly + lh / 2, a.box), _nearest(x + w / 2, y + h / 2, a.lens)]
    _stroke(d, leader, style, width=2)
    _stroke(d, _rect_pts(a.box), style, width=2)
    _stroke(d, _rect_pts(a.lens), style)
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
