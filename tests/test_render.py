"""Tests for the slide renderer.

Stage 4 needs PNG slides; verify the rendering pipeline produces them at the
right count and aspect ratio. The cache check is also exercised (second call
must skip the soffice/pdftoppm work).
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

from src.render import render_deck

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "test_deck.pptx"


@pytest.mark.slow
class TestRender:
    def test_renders_seven_pngs(self, tmp_path):
        pngs = render_deck(DECK, tmp_path)
        assert len(pngs) == 7
        for i, p in enumerate(pngs, start=1):
            assert p.name == f"slide_{i:03d}.png"
            assert p.exists() and p.stat().st_size > 1000  # non-empty PNG

    def test_aspect_ratio_matches_deck(self, tmp_path):
        from PIL import Image
        pngs = render_deck(DECK, tmp_path)
        w, h = Image.open(pngs[0]).size
        # deck is 13.33 x 7.50 inches → 1.777 aspect
        assert abs((w / h) - (13.33 / 7.50)) < 0.005

    def test_cache_skips_rerender(self, tmp_path):
        # first call: produces files
        pngs1 = render_deck(DECK, tmp_path)
        first_mtimes = [p.stat().st_mtime for p in pngs1]
        # sleep a tick so mtimes would differ if regenerated
        time.sleep(0.05)
        pngs2 = render_deck(DECK, tmp_path)
        second_mtimes = [p.stat().st_mtime for p in pngs2]
        assert first_mtimes == second_mtimes, "cache miss — files were re-rendered"

    def test_cache_invalidates_when_deck_newer(self, tmp_path):
        render_deck(DECK, tmp_path)  # warm the cache; result unused
        # touch the deck forward in time
        future = time.time() + 10
        # we can't really mutate test_deck.pptx mtime safely; instead copy it and use the copy
        deck_copy = tmp_path / "deck.pptx"
        shutil.copy(DECK, deck_copy)
        out_dir = tmp_path / "pngs2"
        out_dir.mkdir()
        # render once via the copy
        render_deck(deck_copy, out_dir)
        # touch copy forward to invalidate
        os.utime(deck_copy, (future, future))
        # re-render — should NOT skip
        before = sorted((out_dir / f"slide_{i:03d}.png").stat().st_mtime for i in range(1, 8))
        time.sleep(0.05)
        render_deck(deck_copy, out_dir)
        after = sorted((out_dir / f"slide_{i:03d}.png").stat().st_mtime for i in range(1, 8))
        assert before != after, "expected re-render after deck mtime advance"
