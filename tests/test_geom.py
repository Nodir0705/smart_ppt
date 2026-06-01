"""Pure geometry tests for the Stage 4 overlay math.

These run without a display, without an embedding model — they cover the
coordinate transforms and animation easing that determine *where* the
highlight is drawn at any given moment.
"""

from __future__ import annotations

import math

import pytest

from src.geom import FitTransform, ease_in_out_cubic, inches_to_image_px, lerp_bbox


class TestInchesToImagePx:
    def test_origin_maps_to_origin(self):
        out = inches_to_image_px((0, 0, 1, 1), slide_w_in=10, slide_h_in=5, image_w_px=1000, image_h_px=500)
        assert out == (0, 0, 100, 100)

    def test_full_slide_covers_full_image(self):
        out = inches_to_image_px((0, 0, 10, 5), slide_w_in=10, slide_h_in=5, image_w_px=2000, image_h_px=1000)
        assert out == (0, 0, 2000, 1000)

    def test_test_deck_diagram_box(self):
        # slide 6, Matcher box: bbox (6.80, 2.60, 2.40, 1.40) in a 13.33 x 7.50 deck
        # rendered at 2001 x 1125 px
        out = inches_to_image_px(
            (6.80, 2.60, 2.40, 1.40),
            slide_w_in=13.33, slide_h_in=7.50,
            image_w_px=2001, image_h_px=1125,
        )
        x, y, w, h = out
        # x = 6.80 * 2001/13.33 = ~1020.6, y = 2.60 * 1125/7.50 = 390.0
        assert math.isclose(x, 6.80 * (2001 / 13.33), rel_tol=1e-9)
        assert math.isclose(y, 2.60 * (1125 / 7.50), rel_tol=1e-9)
        assert math.isclose(w, 2.40 * (2001 / 13.33), rel_tol=1e-9)
        assert math.isclose(h, 1.40 * (1125 / 7.50), rel_tol=1e-9)


class TestFitTransform:
    def test_image_fits_exactly_in_matching_canvas(self):
        fit = FitTransform.fit(1000, 500, 1000, 500)
        assert fit.scale == 1.0
        assert fit.offset_x == 0
        assert fit.offset_y == 0
        assert fit.draw_w == 1000
        assert fit.draw_h == 500

    def test_letterboxed_horizontally(self):
        # 16:9 image inside a 4:3 canvas → letterboxed top/bottom
        fit = FitTransform.fit(1600, 900, 1200, 1200)
        assert math.isclose(fit.scale, 1200 / 1600)
        assert math.isclose(fit.draw_w, 1200)
        assert math.isclose(fit.draw_h, 900 * (1200 / 1600))  # 675
        assert math.isclose(fit.offset_x, 0)
        assert math.isclose(fit.offset_y, (1200 - 675) / 2)

    def test_letterboxed_vertically(self):
        # 4:3 image inside a 16:9 canvas → letterboxed left/right
        fit = FitTransform.fit(800, 600, 1600, 900)
        assert math.isclose(fit.scale, 900 / 600)
        assert math.isclose(fit.draw_h, 900)
        assert math.isclose(fit.draw_w, 800 * 1.5)  # 1200
        assert math.isclose(fit.offset_x, (1600 - 1200) / 2)
        assert math.isclose(fit.offset_y, 0)

    def test_image_to_canvas_applies_scale_and_offset(self):
        fit = FitTransform.fit(1600, 900, 1200, 1200)  # scale 0.75, offset_y=262.5
        out = fit.image_bbox_to_canvas((100, 200, 50, 50))
        assert math.isclose(out[0], 100 * 0.75)
        assert math.isclose(out[1], 200 * 0.75 + 262.5)
        assert math.isclose(out[2], 50 * 0.75)
        assert math.isclose(out[3], 50 * 0.75)

    def test_zero_canvas_falls_back_to_identity(self):
        fit = FitTransform.fit(100, 100, 0, 0)
        assert fit.scale == 1.0
        assert fit.draw_w == 0
        assert fit.draw_h == 0


class TestLerpAndEase:
    def test_lerp_endpoints(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (100.0, 50.0, 20.0, 20.0)
        assert lerp_bbox(a, b, 0.0) == a
        assert lerp_bbox(a, b, 1.0) == b

    def test_lerp_midpoint(self):
        out = lerp_bbox((0, 0, 10, 10), (100, 50, 20, 20), 0.5)
        assert out == (50, 25, 15, 15)

    def test_lerp_clamps_t(self):
        a, b = (0, 0, 0, 0), (10, 10, 10, 10)
        assert lerp_bbox(a, b, -0.5) == a
        assert lerp_bbox(a, b, 1.5) == b

    @pytest.mark.parametrize("t,expected", [(0, 0), (1, 1), (0.5, 0.5)])
    def test_ease_endpoints_and_middle(self, t, expected):
        assert math.isclose(ease_in_out_cubic(t), expected, abs_tol=1e-9)

    def test_ease_is_monotonic(self):
        ts = [i / 20 for i in range(21)]
        vs = [ease_in_out_cubic(t) for t in ts]
        for prev, cur in zip(vs, vs[1:]):
            assert cur >= prev - 1e-9
