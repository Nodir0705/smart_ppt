"""Task 3: HUD geometry + effects (pure, deterministic)."""
from __future__ import annotations

import hashlib

import pytest
from PIL import Image

from tools.video.effects import HUD, Effect, draw, lens_box, state_at
from tools.video.geometry import Letterbox, empty_region, union


def _overlaps(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


# ---- geometry ---------------------------------------------------------------

def test_letterbox_bars_and_box():
    lb = Letterbox(13.333, 7.5, 1280, 720, 640, 480)   # 16:9 into 4:3 -> bars top/bottom
    assert lb.box_px((0, 0, 13.333, 7.5), pad_px=0) == (0, 60, 640, 360)
    assert lb.box_px({"x": 0, "y": 0, "w": 13.333, "h": 7.5}) == (-6, 54, 652, 372)
    frame = lb.place(Image.new("RGB", (1280, 720), (200, 200, 200)))
    assert frame.size == (640, 480)
    assert frame.getpixel((320, 10)) == lb.bg            # bar
    assert frame.getpixel((320, 240)) == (200, 200, 200)  # slide


def test_union():
    assert union([(10, 10, 10, 10), (30, 5, 10, 10)]) == (10, 5, 30, 15)


def test_empty_region_avoids_occupied_and_prefers_right():
    occupied = [(0, 0, 320, 360)]                      # left half full
    x, y = empty_region((640, 360), occupied, 200, 150)
    assert not _overlaps((x, y, 200, 150), occupied[0])
    assert 0 <= x <= 640 - 200 and 0 <= y <= 360 - 150
    # a centre column splits the frame into equal free halves -> right wins
    x, y = empty_region((640, 360), [(280, 0, 80, 360)], 100, 100)
    assert x >= 360
    # the scattered case: lens must not touch any occupied box when a fit exists
    occ = [(0, 0, 640, 80), (0, 0, 120, 360), (400, 200, 240, 160)]
    x, y = empty_region((640, 360), occ, 160, 100)
    assert not any(_overlaps((x, y, 160, 100), b) for b in occ)


def test_empty_region_deterministic():
    occ = [(100, 50, 200, 100)]
    assert empty_region((1920, 1080), occ, 400, 300) == empty_region((1920, 1080), occ, 400, 300)


def test_lens_box_avoids_source_and_fits_frame():
    src = (40, 40, 100, 50)
    lx, ly, lw, lh = lens_box(src, (640, 360), [src])
    assert (lw, lh) == (220, 110)                      # 2.2x
    assert not _overlaps((lx, ly, lw, lh), src)
    big = lens_box((0, 0, 600, 300), (640, 360), [])
    assert big[2] <= 640 and big[3] <= 360


# ---- timeline / state_at ----------------------------------------------------

A, B = (100, 100, 100, 50), (300, 200, 80, 40)
PLAN = [
    Effect("focus", 1.0, 3.0, A, {"spotlight": True}),
    Effect("underline", 1.5, 3.0, (110, 110, 80, 20)),
    Effect("ring", 2.0, 3.0, (150, 120, 30, 20)),
    Effect("glide", 3.0, 6.0, B, {"from": A, "spotlight": True}),
    Effect("zoom", 4.0, 5.0, B, {"lens": (400, 20, 176, 88)}),
]


def _by_kind(state):
    return {a.kind: a for a in state}


def test_state_at_active_sets():
    assert state_at(0.5, PLAN) == ()
    assert set(_by_kind(state_at(1.2, PLAN))) == {"focus"}
    assert set(_by_kind(state_at(2.5, PLAN))) == {"focus", "underline", "ring"}
    # half-open windows: at 3.0 the focus/marks are gone and the glide has begun
    assert set(_by_kind(state_at(3.0, PLAN))) == {"glide"}
    assert set(_by_kind(state_at(4.5, PLAN))) == {"glide", "zoom"}
    assert state_at(6.0, PLAN) == ()


def test_state_at_progress():
    f = _by_kind(state_at(1.09, PLAN))["focus"]        # halfway through the 180 ms snap
    assert f.p == pytest.approx(0.5)
    assert f.box[2] == pytest.approx(100 * 1.04) and f.box[3] == pytest.approx(50 * 1.04)
    cx = f.box[0] + f.box[2] / 2
    assert cx == pytest.approx(150)                    # scales about the centre
    assert f.spotlight is True
    assert _by_kind(state_at(1.5, PLAN))["focus"].box == A          # snapped
    assert _by_kind(state_at(1.675, PLAN))["underline"].p == pytest.approx(0.5)
    assert _by_kind(state_at(2.1, PLAN))["ring"].p == pytest.approx(0.25)
    g = _by_kind(state_at(3.2, PLAN))["glide"]         # halfway: eased 0.5 -> midpoint
    assert g.p == pytest.approx(0.5)
    assert g.box == pytest.approx((200, 150, 90, 45))
    assert _by_kind(state_at(3.5, PLAN))["glide"].box == B
    assert _by_kind(state_at(4.1, PLAN))["zoom"].p == pytest.approx(0.5)  # fade in
    assert _by_kind(state_at(4.5, PLAN))["zoom"].p == 1.0
    assert _by_kind(state_at(4.95, PLAN))["zoom"].p == pytest.approx(0.25)  # fade out
    assert _by_kind(state_at(4.5, PLAN))["zoom"].lens == (400, 20, 176, 88)


def test_state_is_hashable_and_stable():
    assert hash(state_at(4.5, PLAN)) == hash(state_at(4.5, PLAN))
    assert state_at(2.9, PLAN) == state_at(2.95, PLAN)  # settled -> memo hit


# ---- draw ---------------------------------------------------------------------

def _frame():
    img = Image.new("RGB", (640, 360), (128, 128, 128))
    for x in range(0, 640, 40):                        # some structure for the zoom crop
        for y in range(360):
            img.putpixel((x, y), (30, 60, 90))
    return img


def test_draw_deterministic_and_spotlight():
    frame = _frame()
    plan = [
        Effect("focus", 0.0, 9.0, (100, 100, 200, 100), {"spotlight": True}),
        Effect("underline", 0.0, 9.0, (110, 150, 150, 20)),
        Effect("ring", 0.0, 9.0, (220, 110, 40, 30)),
        Effect("zoom", 0.0, 9.0, (40, 250, 80, 40), {"lens": (420, 20, 176, 88)}),
    ]
    st = state_at(1.0, plan)
    a = draw(frame, st, HUD).tobytes()
    b = draw(frame, st, HUD).tobytes()
    assert hashlib.sha256(a).digest() == hashlib.sha256(b).digest()
    out = draw(frame, state_at(1.0, plan[:1]), HUD)
    inside, outside = out.getpixel((150, 130)), out.getpixel((500, 300))
    assert sum(inside) > sum(outside)
    assert outside == tuple(round(128 * HUD["dim"]) for _ in range(3))
    assert frame.tobytes() == _frame().tobytes()       # input not mutated


def test_draw_empty_state_unchanged():
    frame = _frame()
    assert draw(frame, state_at(1.0, []), HUD).tobytes() == frame.tobytes()
