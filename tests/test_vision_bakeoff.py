"""Pure helpers of the vision bake-off (tools/vision_bakeoff.py) — no models, no downloads."""
from __future__ import annotations

import pytest

from tools import vision_bakeoff as vb

SLIDE = {"w": 13.333333333333334, "h": 7.5, "units": "inches"}


def box(x, y, w, h):
    return {"x": x, "y": y, "w": w, "h": h}


def el(id_, text, b, type_="body"):
    return {"id": id_, "type": type_, "text": text, "bbox": b}


def test_px_per_inch_and_round_trip():
    sx, sy = vb.px_per_inch(2001, 1125, SLIDE)
    assert sy == pytest.approx(150.0) and sx == pytest.approx(150.075, abs=1e-3)
    crop = vb.inch_to_px_box(box(1, 2, 3, 1), (sx, sy), 2001, 1125)
    assert crop == (150, 300, 601, 450)
    x, y, w, h = vb.px_to_inch_box(*crop, (sx, sy))
    assert (x, y, w, h) == pytest.approx((1, 2, 3, 1), abs=0.01)


def test_px_per_inch_rejects_non_inch_and_skew():
    with pytest.raises(AssertionError):
        vb.px_per_inch(2001, 1125, {"w": 12192000, "h": 6858000, "units": "emu"})
    with pytest.raises(AssertionError):
        vb.px_per_inch(2001, 900, SLIDE)


def test_crop_clips_offslide_and_drops_empty():
    s = (150.0, 150.0)
    assert vb.inch_to_px_box(box(0, 0, 13.33, 10.17), s, 2000, 1125) == (0, 0, 2000, 1125)
    assert vb.inch_to_px_box(box(14, 1, 1, 1), s, 2000, 1125) is None


def test_quad_to_xyxy():
    assert vb.quad_to_xyxy([10, 5, 50, 6, 51, 20, 9, 19]) == (9, 5, 51, 20)


def test_classify_image():
    W, H = SLIDE["w"], SLIDE["h"]
    assert vb.classify_image(box(0, 0, 13.33, 10.17), W, H) == "background"
    assert vb.classify_image(box(0, 0, 12.5, 7.0), W, H) == "background"   # both >= 90%
    assert vb.classify_image(box(0, 0, 13.33, 3.0), W, H) == "content"     # wide banner, not bg
    assert vb.classify_image(box(1, 1, 0.7, 0.7), W, H) == "icon"          # 0.49 in^2
    assert vb.classify_image(box(1, 1, 1.0, 0.5), W, H) == "content"       # exactly 0.5


def test_gt_and_anchor_text_elements():
    slide = {"elements": [el("t", "Title", box(1, 1, 5, 1), "title"),
                          el("c", "Chart", box(1, 3, 5, 3), "chart"),
                          el("i", None, box(6, 3, 2, 2), "image"),
                          el("full", "BIG", box(0, 0, 13.33, 7.5), "diagram_label"),
                          el("e", "", box(1, 6, 1, 1))]}
    assert [e["id"] for e in vb.gt_text_elements(slide)] == ["t", "full"]
    assert [e["id"] for e in vb.anchor_text_elements(slide, SLIDE["w"], SLIDE["h"])] == ["t"]


def test_words_and_content_words():
    assert vb.words("Hello, World! (Foo-bar) — 3.5mm") == ["hello", "world", "foo-bar", "3.5mm"]
    assert vb.content_words("A photo of the Qualcomm chip, 845") == {"qualcomm", "chip"}
    assert vb.word_overlap(["a", "a", "b"], ["a", "b", "b"]) == 2


def test_rect_gap_iou_contains():
    assert vb.rect_gap(box(0, 0, 1, 1), box(2, 0, 1, 1)) == pytest.approx(1.0)
    assert vb.rect_gap(box(0, 0, 1, 1), box(4, 5, 1, 1)) == pytest.approx(5.0)   # 3-4-5
    assert vb.rect_gap(box(0, 0, 2, 2), box(1, 1, 2, 2)) == 0.0
    assert vb.iou(box(0, 0, 2, 2), [1, 0, 2, 2]) == pytest.approx(1 / 3)
    assert vb.iou(box(0, 0, 1, 1), box(5, 5, 1, 1)) == 0.0
    assert vb.contains_point(box(0, 0, 2, 2), (1, 1)) and not vb.contains_point(box(0, 0, 2, 2), (3, 1))


def test_nearest_text_and_clip_rank_metric():
    img = box(8, 2, 3, 2)                       # centre (9.5, 3)
    texts = [el("far", "far away", box(0, 0, 2, 1)),
             el("cap", "Snapdragon 845 SoC", box(8, 4.2, 3, 0.5)),   # centre (9.5, 4.45)
             el("mid", "middle", box(4, 2.5, 2, 1))]                 # centre (5, 3)
    target = vb.nearest_text_id(img, texts)
    assert target == "cap"
    scores = {"far": 20.1, "cap": 25.3, "mid": 25.3}
    assert vb.rank_of(target, scores) == 1      # tie broken by id: "cap" < "mid"
    assert vb.rank_of("mid", scores) == 2
    assert vb.rank_of("far", scores) == 3
    assert vb.nearest_text_id(img, []) is None


def test_caption_hit_and_overlay_and_icon_risk():
    img = box(8, 2, 3, 2)
    texts = [el("cap", "Snapdragon 845 SoC", box(8, 4.5, 3, 0.5)),     # gap 0.5in
             el("far", "Audio codec", box(0, 6.5, 2, 0.5))]             # gap > 1in
    assert vb.caption_hit("a close up of a snapdragon chip", img, texts) == (True, True)
    assert vb.caption_hit("an audio codec", img, texts) == (True, False)    # only far text matches
    assert vb.caption_hit("anything", img, texts[1:]) == (False, False)
    assert not vb.has_overlaid_text(img, texts)
    assert vb.has_overlaid_text(img, [el("in", "x", box(9, 2.5, 1, 1))])
    assert vb.icon_risk("a microphone icon", texts)          # mints "microphone"
    assert not vb.icon_risk("snapdragon logo", texts)        # word already on slide; "logo" is filler
    assert not vb.icon_risk("a green icon", [])              # nothing but filler


def test_layout_scores():
    gt = [box(1, 1, 4, 1), box(1, 3, 4, 1)]
    truth = gt + [box(7, 1, 3, 3)]
    pred = [[1.1, 1.0, 4.0, 1.0],    # matches gt[0]
            [7, 1, 3, 3],            # matches the image (truth, not text GT)
            [0, 5, 2, 2]]            # orphan
    assert vb.layout_scores(gt, truth, pred, 0.5) == (1, 1)
    assert vb.layout_scores(gt, truth, [], 0.5) == (0, 0)
