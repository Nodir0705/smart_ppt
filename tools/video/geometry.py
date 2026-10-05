"""Frame geometry for the HUD renderer (pure, deterministic).

- ``Letterbox`` maps deck inches -> slide PNG px -> frame px (aspect kept,
  centred on a near-black background), reusing ``src.geom``.
- ``union`` is the bounding box of several (x, y, w, h) boxes.
- ``empty_region`` places the zoom lens.

Decisions where the plan left details open:
- ``box_px`` accepts a tuple or the index's ``{x, y, w, h}`` dict and returns
  rounded integer frame pixels, so boxes are stable cache/memo keys.
- ``empty_region`` marks every cell of a 16x9 grid that overlaps an occupied
  box, then ranks all-free cell rectangles by (fits want_w x want_h, area,
  right edge, -top, width) and takes the best: the largest free rectangle,
  rectangles that can hold the lens first, ties going right then top. It
  returns the top-left of a want_w x want_h box centred in that rectangle and
  clamped into the frame. With no free cell at all it returns the top-right
  corner.
"""
from __future__ import annotations

from PIL import Image

from src.geom import FitTransform, inches_to_image_px

Box = tuple  # (x, y, w, h)

BG = (8, 10, 14)
GRID = (16, 9)


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


def empty_region(frame_size, occupied_boxes, want_w, want_h) -> tuple[int, int]:
    W, H = frame_size
    nx, ny = GRID
    cw, ch = W / nx, H / ny
    free = [[not any(overlaps((i * cw, j * ch, cw, ch), b) for b in occupied_boxes)
             for i in range(nx)] for j in range(ny)]
    best = None
    for j0 in range(ny):
        for i0 in range(nx):
            max_i1 = nx - 1
            for j1 in range(j0, ny):
                i = i0
                while i <= max_i1 and free[j1][i]:
                    i += 1
                max_i1 = i - 1
                if max_i1 < i0:
                    break
                # for fixed (i0, j0, j1) the widest rectangle wins every key field
                cols, rows = max_i1 - i0 + 1, j1 - j0 + 1
                key = (cols * cw >= want_w and rows * ch >= want_h, cols * rows, max_i1, -j0, cols)
                if best is None or key > best[0]:
                    best = (key, i0, j0, cols, rows)
    if best is None:
        return (max(0, W - want_w), 0)
    _, i0, j0, cols, rows = best
    x = i0 * cw + (cols * cw - want_w) / 2
    y = j0 * ch + (rows * ch - want_h) / 2
    return (round(max(0, min(x, W - want_w))), round(max(0, min(y, H - want_h))))
