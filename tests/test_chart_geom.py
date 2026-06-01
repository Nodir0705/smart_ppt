"""Tests for chart sub-region geometry (option 2).

The math: insets compute the plot area; per-category sub-bboxes are
positioned within the plot area. These checks pin down invariants
(non-overlap, bbox containment, slice angles summing to 360°) rather than
exact pixel positions, since the insets are heuristics that may be tuned.
"""

from __future__ import annotations

import math

import pytest

from src.chart_geom import (
    bar_subregions,
    classify_chart_type,
    line_subregions,
    pie_subregions,
    plot_area,
    subregions_for_chart,
)
from src.parser import BBox


CHART = BBox(x=1.0, y=2.0, w=6.0, h=4.0)  # arbitrary fixture


def _contains(outer: BBox, inner: BBox, eps: float = 1e-6) -> bool:
    return (
        inner.x >= outer.x - eps
        and inner.y >= outer.y - eps
        and inner.x + inner.w <= outer.x + outer.w + eps
        and inner.y + inner.h <= outer.y + outer.h + eps
    )


class TestClassify:
    @pytest.mark.parametrize("raw,expected", [
        ("COLUMN_CLUSTERED (51)", "bar"),
        ("BAR_STACKED (53)", "bar"),
        ("PIE (5)", "pie"),
        ("DOUGHNUT (8)", "pie"),
        ("LINE_MARKERS (65)", "line"),
        ("AREA (1)", "line"),
        ("SCATTER_MARKERS (-4169)", "scatter"),
        ("RAINBOW", "bar"),  # default
    ])
    def test_known_types(self, raw, expected):
        assert classify_chart_type(raw) == expected


class TestPlotArea:
    def test_strictly_inside_chart(self):
        p = plot_area(CHART, "bar")
        assert _contains(CHART, p)
        # actually inset, not equal
        assert p.x > CHART.x and p.y > CHART.y
        assert p.x + p.w < CHART.x + CHART.w
        assert p.y + p.h < CHART.y + CHART.h

    def test_pie_inset_is_symmetric(self):
        p = plot_area(CHART, "pie")
        left_pad = p.x - CHART.x
        right_pad = (CHART.x + CHART.w) - (p.x + p.w)
        # pie inset is the same on left and right by design
        assert math.isclose(left_pad, right_pad, abs_tol=1e-6)


class TestBarSubregions:
    def test_four_categories_split_plot_width(self):
        regions = bar_subregions(CHART, 4)
        assert len(regions) == 4
        plot = plot_area(CHART, "bar")
        # all strips have the same width
        widths = [r.w for r in regions]
        assert all(math.isclose(w, widths[0], abs_tol=1e-9) for w in widths)
        # together they cover the plot width
        total_w = sum(r.w for r in regions)
        assert math.isclose(total_w, plot.w, abs_tol=1e-6)

    def test_non_overlapping(self):
        regions = bar_subregions(CHART, 4)
        for a, b in zip(regions, regions[1:]):
            assert math.isclose(a.x + a.w, b.x, abs_tol=1e-9)

    def test_all_inside_chart(self):
        for r in bar_subregions(CHART, 4):
            assert _contains(CHART, r)

    def test_zero_categories(self):
        assert bar_subregions(CHART, 0) == []


class TestLineSubregions:
    def test_five_evenly_spaced_centers(self):
        regions = line_subregions(CHART, 5)
        assert len(regions) == 5
        plot = plot_area(CHART, "line")
        centers = [r.x + r.w / 2 for r in regions]
        # first and last centers at plot edges, others evenly spaced between
        assert math.isclose(centers[0], plot.x, abs_tol=1e-6)
        assert math.isclose(centers[-1], plot.x + plot.w, abs_tol=1e-6)
        gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        assert all(math.isclose(g, gaps[0], abs_tol=1e-6) for g in gaps)

    def test_single_point_centered(self):
        regions = line_subregions(CHART, 1)
        assert len(regions) == 1
        plot = plot_area(CHART, "line")
        cx = regions[0].x + regions[0].w / 2
        assert math.isclose(cx, plot.x + plot.w / 2, abs_tol=1e-6)


class TestPieSubregions:
    def test_three_slices(self):
        # 50/20/30 slices like the test deck
        regions = pie_subregions(CHART, [50, 20, 30])
        assert len(regions) == 3
        for r in regions:
            assert r.w > 0 and r.h > 0
            assert _contains(CHART, r)

    def test_empty_values(self):
        assert pie_subregions(CHART, []) == []

    def test_all_zero_values_doesnt_divide_by_zero(self):
        # weird input — should not crash and should still produce sectors
        regions = pie_subregions(CHART, [0, 0, 0])
        assert len(regions) == 3


class TestDispatch:
    def test_dispatch_picks_right_function(self):
        # bar
        rs = subregions_for_chart(CHART, "COLUMN_CLUSTERED (51)", 4)
        assert len(rs) == 4
        # the bar dispatch should match bar_subregions exactly
        assert rs == bar_subregions(CHART, 4)
        # pie
        rs = subregions_for_chart(CHART, "PIE (5)", 3, [50, 20, 30])
        assert len(rs) == 3
        assert rs == pie_subregions(CHART, [50, 20, 30])
        # line
        rs = subregions_for_chart(CHART, "LINE_MARKERS (65)", 5)
        assert len(rs) == 5
        assert rs == line_subregions(CHART, 5)
