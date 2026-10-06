"""Task 3: HUD geometry + effects (pure, deterministic)."""
from __future__ import annotations

import hashlib

import pytest
from PIL import Image

from tools.video.effects import HUD, Effect, arm_len, back, draw, lens_box, ring_ellipse, state_at
from tools.video.geometry import Letterbox, union


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


def _area(a, b):
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h)


def test_lens_fits_at_2_2_next_to_its_source():
    src = (40, 40, 100, 50)
    lens = lens_box(src, (640, 360), [src])
    assert lens[2:] == (220, 110)                      # 2.2x
    assert not _overlaps(lens, src) and lens[0] >= 0 and lens[0] + lens[2] <= 640
    assert lens_box(src, (640, 360), [src]) == lens    # deterministic
    big = lens_box((0, 0, 600, 300), (640, 360), [])
    assert big[2] <= 640 and big[3] <= 360


def test_lens_shrinks_toward_1_6_before_it_overlaps():
    soft = [(0, 0, 460, 360)]                          # free: a 180 px column on the right (4 px gaps at 360p)
    src = (20, 20, 100, 100)
    lens = lens_box(src, (640, 360), soft)
    assert lens[2:] == (170, 170)                      # 2.2 .. 1.8 do not fit, 1.7 does
    assert not any(_overlaps(lens, b) for b in soft)


def test_lens_never_covers_focus_or_source_and_covers_least_content():
    src, focus = (20, 20, 100, 50), (0, 0, 320, 360)
    soft = [(0, 0, 640, 300), (330, 300, 120, 60)]     # no free spot at any scale
    lens = lens_box(src, (640, 360), soft, [focus])
    assert lens[2:] == (160, 80)                       # 1.6x once overlap is unavoidable
    assert not _overlaps(lens, focus) and not _overlaps(lens, src)
    assert _area(lens, soft[1]) == 0                   # the bottom-right corner covers only the big box
    assert lens[1] + lens[3] == 360 - 4


def test_lens_overlap_spares_the_title_before_other_content():
    src, focus = (500, 20, 100, 50), (480, 0, 160, 120)
    title, soft = [(0, 0, 460, 90)], [(0, 150, 640, 210)]   # a 60 px band between title and photo
    # by area alone, any lens between them covers 28 rows (title or photo), and the source pulls it up
    lens = lens_box(src, (640, 360), soft, [focus], title)
    assert lens[2:] == (160, 80)
    assert not _overlaps(lens, title[0]) and not _overlaps(lens, focus) and not _overlaps(lens, src)
    assert lens[1] == 90 + 4 and _area(lens, soft[0]) > 0  # the photo is covered instead


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
    f = _by_kind(state_at(1.09, PLAN))["focus"]        # halfway through the 180 ms snap ("back" ease)
    assert f.p == pytest.approx(0.5)
    k = 1.08 - 0.08 * back(0.5)
    assert f.box[2] == pytest.approx(100 * k) and f.box[3] == pytest.approx(50 * k)
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


def test_back_ease_overshoots_then_settles():
    assert back(0.0) == pytest.approx(0.0) and back(1.0) == 1.0
    peak = max(back(i / 100) for i in range(101))
    assert 1.04 < peak < 1.06                          # ~5 % overshoot
    f = _by_kind(state_at(1.0 + 0.65 * 0.18, PLAN))["focus"]
    assert f.box[2] < 100                              # brackets dip just under 100 % before settling


def test_bracket_arms_have_a_floor_that_scales_with_the_frame():
    assert arm_len((0, 0, 400, 36), 1080) == 18        # one-line target: 0.18 * 36 = 6.5 -> 18 px floor
    assert arm_len((0, 0, 400, 300), 1080) == pytest.approx(54)   # 0.18 * 300
    assert arm_len((0, 0, 400, 36), 720) == 12         # floor scales with frame height
    assert arm_len((0, 0, 30, 30), 1080) == 18         # even past 0.4 * min side, the floor wins


def test_ring_is_compact_and_stays_in_the_frame():
    x0, y0, x1, y1 = ring_ellipse((100, 100, 100, 50), (640, 360), 1.0)
    assert (x1 - x0, y1 - y0) == pytest.approx((130, 65))          # 1.3x the padded box
    assert ((x0 + x1) / 2, (y0 + y1) / 2) == pytest.approx((150, 125))
    x0, y0, x1, y1 = ring_ellipse((560, 330, 80, 30), (640, 360), 1.0)   # in the corner
    assert x0 >= 0 and y0 >= 0 and x1 <= 640 - 3 and y1 <= 360 - 3
    x0, y0, x1, y1 = ring_ellipse((0, 100, 640, 20), (640, 360), 1.0)    # a wide box can't push it off-frame
    assert x0 >= 3 and x1 <= 637
    start = ring_ellipse((100, 100, 100, 50), (640, 360), 0.0)
    assert start[2] - start[0] == pytest.approx(130 * 1.08)         # snaps in from 108 %


def test_accent_strokes_have_a_soft_glow():
    frame = Image.new("RGB", (1280, 720), (128, 128, 128))   # glow radius scales: 4 px at 720p
    plan = [Effect("focus", 0.0, 9.0, (100, 100, 200, 100))]
    out = draw(frame, state_at(1.0, plan), HUD)
    r, g, b = out.getpixel((112, 92))                   # 8 px above the top-left arm, 4 px outside the halo
    assert b > r + 5                                    # cyan tint
    assert out.getpixel((200, 150)) == (128, 128, 128)  # the box centre is untouched
    assert draw(frame, state_at(1.0, plan), HUD).tobytes() == out.tobytes()
