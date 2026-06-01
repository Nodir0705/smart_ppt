"""Pure geometry helpers for the overlay — kept out of viewer.py so they can
be unit-tested without a display.

All bboxes are 4-tuples (x, y, w, h). Coordinate spaces:

  inches      — element bbox as it sits in the deck (Stage 1 output)
  image_px    — pixels on the rendered slide PNG (from src/render.py)
  canvas_px   — pixels in the Tk Canvas (image, scaled to fit, may be letter-boxed)
"""

from __future__ import annotations

from dataclasses import dataclass


Bbox = tuple[float, float, float, float]  # (x, y, w, h)


def inches_to_image_px(
    bbox_in: Bbox,
    slide_w_in: float,
    slide_h_in: float,
    image_w_px: int,
    image_h_px: int,
) -> Bbox:
    """Map a deck-inch bbox onto the rendered slide image (in pixels)."""
    sx = image_w_px / slide_w_in
    sy = image_h_px / slide_h_in
    x, y, w, h = bbox_in
    return (x * sx, y * sy, w * sx, h * sy)


@dataclass
class FitTransform:
    """How an image of size (img_w, img_h) is letterboxed onto a (canvas_w, canvas_h) area.

    Preserves aspect ratio; centers within the canvas.
    """
    scale: float
    offset_x: float
    offset_y: float
    draw_w: float
    draw_h: float

    @classmethod
    def fit(cls, img_w: int, img_h: int, canvas_w: int, canvas_h: int) -> "FitTransform":
        if img_w <= 0 or img_h <= 0 or canvas_w <= 0 or canvas_h <= 0:
            return cls(scale=1.0, offset_x=0, offset_y=0, draw_w=0, draw_h=0)
        scale = min(canvas_w / img_w, canvas_h / img_h)
        draw_w = img_w * scale
        draw_h = img_h * scale
        offset_x = (canvas_w - draw_w) / 2
        offset_y = (canvas_h - draw_h) / 2
        return cls(scale=scale, offset_x=offset_x, offset_y=offset_y, draw_w=draw_w, draw_h=draw_h)

    def image_bbox_to_canvas(self, bbox_image_px: Bbox) -> Bbox:
        x, y, w, h = bbox_image_px
        return (
            x * self.scale + self.offset_x,
            y * self.scale + self.offset_y,
            w * self.scale,
            h * self.scale,
        )


def lerp_bbox(a: Bbox, b: Bbox, t: float) -> Bbox:
    """Linear interpolation between two bboxes, t in [0, 1]."""
    t = max(0.0, min(1.0, t))
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return (
        ax + (bx - ax) * t,
        ay + (by - ay) * t,
        aw + (bw - aw) * t,
        ah + (bh - ah) * t,
    )


def ease_in_out_cubic(t: float) -> float:
    """Smooth ease curve, t in [0, 1] → [0, 1]."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4 * t * t * t
    f = 2 * t - 2
    return 1 + f * f * f / 2
