"""Tests for measuring column-chart bars from a rendered image (chart_detect).

Uses synthetic images (teal rectangles on white) so the detector can be verified
deterministically without rendering the real deck.
"""
from __future__ import annotations

import numpy as np

from src.chart_detect import detect_bars

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
