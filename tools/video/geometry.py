"""Frame geometry for the HUD renderer (pure, deterministic).

- ``Letterbox`` maps deck inches -> slide PNG px -> frame px (aspect kept,
  centred on a near-black background), reusing ``src.geom``.
- ``union`` is the bounding box of several (x, y, w, h) boxes.

Decisions where the plan left details open:
- ``box_px`` accepts a tuple or the index's ``{x, y, w, h}`` dict and returns
  rounded integer frame pixels, so boxes are stable cache/memo keys.
- The zoom lens is placed by ``effects.lens_box`` (an exact search, which
  replaced a 16x9-cell grid whose coarse cells made it overlap content).
"""
from __future__ import annotations

from PIL import Image

from src.geom import FitTransform, inches_to_image_px

Box = tuple  # (x, y, w, h)

BG = (8, 10, 14)


class Letterbox:
    def __init__(self, slide_w_in, slide_h_in, png_w, png_h, frame_w, frame_h, bg=BG):
        self.slide = (slide_w_in, slide_h_in)
        self.png = (png_w, png_h)
        self.size = (frame_w, frame_h)
        self.bg = bg
        self.fit = FitTransform.fit(png_w, png_h, frame_w, frame_h)

    def box_px(self, bbox_in, pad_px=6) -> Box:
        if isinstance(bbox_in, dict):
            bbox_in = (bbox_in["x"], bbox_in["y"], bbox_in["w"], bbox_in["h"])
        x, y, w, h = self.fit.image_bbox_to_canvas(inches_to_image_px(bbox_in, *self.slide, *self.png))
        return (round(x - pad_px), round(y - pad_px), round(w + 2 * pad_px), round(h + 2 * pad_px))

    def place(self, png: Image.Image) -> Image.Image:
        size = (round(self.fit.draw_w), round(self.fit.draw_h))
        img = png.convert("RGB")
        if img.size != size:
            img = img.resize(size, Image.LANCZOS)
        frame = Image.new("RGB", self.size, self.bg)
        frame.paste(img, (round(self.fit.offset_x), round(self.fit.offset_y)))
        return frame


def union(boxes) -> Box:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


def overlaps(a, b) -> bool:
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]
