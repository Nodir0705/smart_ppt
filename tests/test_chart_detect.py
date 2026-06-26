"""Tests for measuring column-chart bars from a rendered image (chart_detect).

Uses synthetic images (teal rectangles on white) so the detector can be verified
deterministically without rendering the real deck.
"""
from __future__ import annotations

import numpy as np

from src.chart_detect import detect_axis_labels, detect_bars, detect_line_points

TEAL = (28, 114, 147)
GREY = (120, 128, 140)


def _canvas(h=400, w=600):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _bar(img, cx, half_w, baseline, height, color=TEAL):
    top = baseline - height
    img[top:baseline, cx - half_w:cx + half_w] = color


def test_recovers_four_bars():
    img = _canvas()
    baseline = 350
    centers = [100, 250, 400, 550]
    values = [100.0, 50.0, 40.0, 20.0]
    scale = 300 / max(values)  # tallest fills 300px
    for cx, v in zip(centers, values):
        _bar(img, cx, 30, baseline, int(v * scale))

    boxes = detect_bars(img, (0, 0, 600, 400), values)
    assert boxes is not None and len(boxes) == 4
    for box, cx, v in zip(boxes, centers, values):
        # centered on the bar, ~bar width, bottom on the baseline, proportional h
        assert abs((box.x + box.w / 2) - cx) <= 3
        assert abs(box.w - 60) <= 4
        assert abs((box.y + box.h) - baseline) <= 3
        assert abs(box.h - v * scale) <= 4


def test_ignores_sparse_grey_text_and_axes():
    img = _canvas()
    baseline = 350
    _bar(img, 150, 30, baseline, 300)
    _bar(img, 450, 30, baseline, 120)
    # sparse grey "axis labels" / gridline speckle — must NOT be read as bars
    img[360:380, 140:160] = GREY            # category label under bar 1
    img[200:202, 50:560] = GREY             # a thin horizontal gridline
    boxes = detect_bars(img, (0, 0, 600, 400), [100.0, 40.0])
    assert boxes is not None and len(boxes) == 2
    assert abs((boxes[0].x + boxes[0].w / 2) - 150) <= 3
    assert abs((boxes[1].x + boxes[1].w / 2) - 450) <= 3


def test_missing_short_bar_gets_interpolated_slot():
    img = _canvas()
    baseline = 350
    # 3 of 4 categories visible; index 1 (value ~0) not drawn at all
    _bar(img, 100, 30, baseline, 300)   # i0
    _bar(img, 400, 30, baseline, 120)   # i2
    _bar(img, 550, 30, baseline, 60)    # i3
    values = [100.0, 0.2, 40.0, 20.0]
    boxes = detect_bars(img, (0, 0, 600, 400), values)
    assert boxes is not None and len(boxes) == 4
    # the undetected bar is placed at the interpolated grid center (~250) with a
    # small visible (floored) height anchored to the baseline
    assert abs((boxes[1].x + boxes[1].w / 2) - 250) <= 6
    assert boxes[1].h > 0
    assert abs((boxes[1].y + boxes[1].h) - baseline) <= 3


def test_returns_none_when_unmeasurable():
    img = _canvas()
    assert detect_bars(img, (0, 0, 600, 400), []) is None          # no values
    assert detect_bars(img, (0, 0, 600, 400), [0, 0, 0]) is None   # all zero
    assert detect_bars(img, (0, 0, 600, 400), [10, 5]) is None     # blank image -> no bars


def _marker(img, cx, cy, color, r=6):
    img[cy - r:cy + r, cx - r:cx + r] = color


def _line(img, p0, p1, color):
    (x0, y0), (x1, y1) = p0, p1
    for t in np.linspace(0, 1, 400):
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        img[y - 1:y + 2, x - 1:x + 2] = color


class TestLinePoints:
    def test_two_series_box_spans_both_markers(self):
        img = _canvas()
        TOP, BOT = (28, 114, 147), (240, 110, 110)  # teal + coral, like the deck
        xs = [80, 230, 380, 530]
        top_y = [250, 180, 130, 110]   # rising series
        bot_y = [300, 330, 360, 380]   # falling series
        for i in range(len(xs)):
            if i:
                _line(img, (xs[i - 1], top_y[i - 1]), (xs[i], top_y[i]), TOP)
                _line(img, (xs[i - 1], bot_y[i - 1]), (xs[i], bot_y[i]), BOT)
        for i, x in enumerate(xs):
            _marker(img, x, top_y[i], TOP)
            _marker(img, x, bot_y[i], BOT)

        boxes = detect_line_points(img, (0, 0, 600, 400), 4, legend_clip_frac=0.0)
        assert boxes is not None and len(boxes) == 4
        for i, (box, x) in enumerate(zip(boxes, xs)):
            # centred on the marker column
            assert abs((box.x + box.w / 2) - x) <= 8
            # spans from the upper marker to the lower marker at that x
            assert box.y <= top_y[i] + 8
            assert box.y + box.h >= bot_y[i] - 8
            # not a full-height strip: bounded to the data band
            assert box.h < 400 * 0.9

    def test_none_on_blank(self):
        assert detect_line_points(_canvas(), (0, 0, 600, 400), 4) is None


class TestAxisLabels:
    def test_finds_labels_and_ignores_axis_line_and_legend(self):
        img = _canvas()
        GREY = (110, 120, 135)
        baseline = 300
        centers = [100, 250, 400, 550]
        # full-width axis line just below the baseline (must be ignored)
        img[baseline + 2:baseline + 4, 40:560] = (90, 90, 90)
        # a short grey label glyph-block under each category
        for cx in centers:
            img[baseline + 12:baseline + 28, cx - 12:cx + 12] = GREY
        # a centred "legend" further down under the middle category (must be ignored)
        img[baseline + 70:baseline + 90, 320:480] = GREY

        boxes = detect_axis_labels(img, (0, 0, 600, 400), centers, baseline, step=150)
        assert len(boxes) == 4
        for box, cx in zip(boxes, centers):
            assert box is not None
            assert abs((box.x + box.w / 2) - cx) <= 6   # centred on the label
            assert box.w <= 40                           # tight to the glyph block, not the legend
            assert box.y > baseline                      # below the baseline
            assert box.y + box.h < baseline + 60         # the label line, not the legend row
