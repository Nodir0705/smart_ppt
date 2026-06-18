"""End-to-end test: spoken phrase → matched element → image pixel bbox.

This is the same path the Tk viewer takes when it places its highlight, minus
the Tk widgets. It catches integration breakage (e.g. parser emitting bboxes
in a different unit, matcher returning a stale id, geom math mis-scaled) that
the per-layer tests would miss.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.geom import inches_to_image_px

ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "out" / "index.json"
SLIDES_DIR = ROOT / "out" / "slides"


@pytest.fixture(scope="module")
def loaded(real_encoder):
    if not INDEX_PATH.exists():
        pytest.skip("run `python3 -m src.index_deck test_deck.pptx` first")
    if not SLIDES_DIR.exists() or not (SLIDES_DIR / "slide_001.png").exists():
        pytest.skip("run `python3 -m src.render test_deck.pptx` first")
    from PIL import Image
    from src.matcher import Index, Matcher

    enc = real_encoder
    idx = Index(INDEX_PATH, enc)
    matcher = Matcher(idx, enc)
    raw = json.loads(INDEX_PATH.read_text())
    return matcher, raw, Image


@pytest.mark.slow
class TestEndToEnd:
    @pytest.mark.parametrize("slide,phrase,expected_id,expected_bbox_px", [
        # Matcher diagram box on slide 6: bbox (6.80, 2.60, 2.40, 1.40) inches
        # at 2001x1125 → x≈1020, y=390, w≈360, h=210
        (6, "let me explain the matcher", "s6_e2", (1020, 390, 360, 210)),
        # Microphone box on slide 6: bbox (0.60, 2.60, 2.40, 1.40)
        (6, "audio comes in through the microphone", "s6_e0", (90, 390, 360, 210)),
        # Sub-millisecond cell on slide 3: bbox row 3, col 3 of the table region
        (3, "the cost column shows sub millisecond time", "s3_e12", (990, 498, 450, 96)),
        # Stat callout on slide 2: container bbox (9.00, 1.60, 3.60, 2.00)
        (2, "the headline number is sub one millisecond", "s2_e0", (1350, 240, 540, 300)),
    ])
    def test_phrase_lands_at_expected_pixels(self, loaded, slide, phrase, expected_id, expected_bbox_px):
        matcher, raw, Image = loaded
        res = matcher.match(phrase, slide)
        assert res.element_id == expected_id, (
            f"phrase {phrase!r} matched {res.element_id} ({res.text!r}) score={res.score:.3f}, "
            f"expected {expected_id}")

        # find the bbox in inches
        el_bbox_in = None
        for s in raw["slides"]:
            if s["slide_number"] != slide:
                continue
            for el in s["elements"]:
                if el["id"] == expected_id:
                    bb = el["bbox"]
                    el_bbox_in = (bb["x"], bb["y"], bb["w"], bb["h"])
                    break
        assert el_bbox_in is not None

        # load slide PNG and transform
        img = Image.open(SLIDES_DIR / f"slide_{slide:03d}.png")
        slide_w_in = raw["slide_size"]["w"]
        slide_h_in = raw["slide_size"]["h"]
        out_px = inches_to_image_px(el_bbox_in, slide_w_in, slide_h_in, img.width, img.height)

        # tolerate small pixel-level drift (rounding during render, etc.)
        for got, want, label in zip(out_px, expected_bbox_px, ["x", "y", "w", "h"]):
            assert abs(got - want) <= 5, (
                f"{label} mismatch for {phrase!r}: got {got:.1f}, want {want}")
