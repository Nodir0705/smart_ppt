"""Pure parts of src/vision_link (no model download: the scorer is stubbed)."""
from PIL import Image

from src.vision_link import (crop_box, decide, link_pictures, picture_candidates,
                             subject, visual_phrases)

W, H = 40 / 3, 7.5   # 16:9 in inches -> exactly 150 px/in at 2000 px


def el(id_, type_, x, y, w, h, text=None, nv=None):
    return {"id": id_, "type": type_, "text": text, "trigger_phrases": [],
            "needs_vision": (type_ == "image" and not text) if nv is None else nv,
            "bbox": {"x": x, "y": y, "w": w, "h": h, "units": "inches"}}


def slide(*els):
    return {"slide_number": 1, "elements": list(els)}


def test_candidates_drop_icon_background_panel_and_outer_frame():
    s = slide(
        el("bg", "image", 0, 0, W, H),                     # background
        el("icon", "image", 1, 1, 0.3, 0.3),               # < 0.5 in^2
        el("card", "image", 1, 3, 4, 1),                   # owns "Machine Driver"
        el("t_card", "body", 1.5, 3.2, 1, 0.4, "Machine Driver"),
        el("frame", "image", 8.4, 1.1, 2.8, 1.7),          # contains the photo
        el("photo", "image", 8.5, 1.2, 2.6, 1.5),
        el("ph2", "image", 8.8, 4.0, 2, 2),                # photo with a caption band
        el("band", "image", 8.8, 5.5, 2, 0.5),
        el("t_cap", "body", 9.0, 5.6, 1.5, 0.3, "WCD9340 Codec"),
        el("alt", "image", 4, 4, 2, 2, "Keypad Matrix"),   # has alt text already
    )
    assert [p["id"] for p in picture_candidates(s, W, H)] == ["photo", "ph2"]


def test_crop_box_maps_inches_to_clipped_pixels():
    assert crop_box({"x": 1, "y": 1, "w": 2, "h": 1}, 2000, 1125, W, H) == (150, 150, 450, 300)
    assert crop_box({"x": 12, "y": 7, "w": 5, "h": 5}, 2000, 1125, W, H) == (1800, 1050, 2000, 1125)
    assert crop_box({"x": 20, "y": 1, "w": 1, "h": 1}, 2000, 1125, W, H) is None


def test_decide_needs_bar_and_margin_and_is_deterministic():
    assert decide({"a": 0.40, "b": 0.30}, 0.28, 0.02) == ("a", 0.40, 0.40 - 0.30)
    assert decide({"a": 0.25, "b": 0.10}, 0.28, 0.02) is None   # below the bar
    assert decide({"a": 0.40, "b": 0.39}, 0.28, 0.02) is None   # no margin
    assert decide({"b": 0.4, "a": 0.4}, 0.28, -1)[0] == "a"      # tie -> id order
    assert decide({}) is None


def test_visual_phrases_are_visual_not_copies():
    assert subject("Keypad Controller: TCA8418 via I2C") == "keypad controller"
    assert subject("Qualcomm Snapdragon 845") == "qualcomm snapdragon 845"
    assert subject("Android 10 (API Level 29) and more words here") == "android 10 and more"
    ph = visual_phrases("Qualcomm Snapdragon 845")
    assert "the photo of qualcomm snapdragon 845" in ph
    assert "qualcomm snapdragon 845" not in ph
    assert visual_phrases("  ") == []


def test_link_pictures_with_stub_scorer(tmp_path):
    Image.new("RGB", (2000, 1125), "white").save(tmp_path / "slide_001.png")
    photo = el("photo", "image", 8.5, 1.2, 2.6, 1.5)
    other = el("p2", "image", 1, 4, 2, 2)
    idx = {"slide_size": {"w": W, "h": H, "units": "inches"},
           "slides": [slide(photo, other,
                            el("t1", "body", 1, 1, 3, 0.4, "Qualcomm Snapdragon 845"),
                            el("t2", "body", 1, 2, 3, 0.4, "Memory"))]}
    calls = []

    def scorer(img, texts):
        calls.append(img.size)
        return [0.40, 0.30] if len(calls) == 1 else [0.30, 0.29]

    recs = link_pictures(idx, tmp_path, scorer=scorer)
    assert calls[0] == (390, 225)          # crop of the photo bbox at 150 px/in
    assert photo["linked_to"] == "t1" and photo["needs_vision"] is False
    assert photo["trigger_phrases"][0] == "the photo of qualcomm snapdragon 845"
    assert "linked_to" not in other and other["needs_vision"] is True   # abstained
    assert [r["linked_to"] for r in recs] == ["t1", None]


def test_link_pictures_skips_without_slides(tmp_path):
    idx = {"slide_size": {"w": W, "h": H}, "slides": [slide(el("p", "image", 1, 1, 2, 2),
                                                           el("t", "body", 4, 1, 2, 1, "x"))]}
    assert link_pictures(idx, tmp_path, scorer=lambda *a: 1 / 0) == []
