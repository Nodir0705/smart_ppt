"""Tests for the slide renderer.

Stage 4 needs PNG slides; verify the rendering pipeline produces them at the
right count and aspect ratio. The cache check is also exercised (second call
must skip the soffice/pdftoppm work).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from src.render import deck_pdf, render_deck

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "test_deck.pptx"


def test_deck_pdf_gives_soffice_a_private_profile(tmp_path, monkeypatch):
    # soffice serialises on its user profile, so two concurrent renders (another
    # test run, another agent) collide unless each gets a fresh profile dir.
    calls = []

    def fake_run(args, **kw):
        calls.append(args)
        outdir = Path(args[args.index("--outdir") + 1])
        (outdir / (Path(args[-1]).stem + ".pdf")).write_bytes(b"%PDF-1.4\n")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    deck_pdf(DECK, tmp_path)
    (args,) = calls
    assert any(a.startswith("-env:UserInstallation=file://") for a in args), args


def test_render_input_is_the_deck_itself_without_equation_boxes(tmp_path):
    # No mc:AlternateContent with a picture Fallback -> soffice gets the original
    # file; no copy is written.
    from src.render import _render_input
    assert _render_input(DECK, tmp_path) == DECK
    assert list(tmp_path.iterdir()) == []


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("soffice") is None, reason="soffice required")
def test_equation_box_renders_as_its_fallback_picture(tmp_path):
    # soffice renders an equation box's mc:Choice WITHOUT the equations (gaps
    # mid-sentence); the mc:Fallback is PowerPoint's own picture of the whole
    # box. The box's region in the PNG must show that picture (solid dark blue
    # in the corpus deck), not the near-white Choice rendering.
    from PIL import Image
    from tools.make_corpus import build_corpus
    deck = {d.name: d for d in build_corpus(tmp_path / "corpus")}["math_textbox"]
    png = render_deck(deck.path, tmp_path / "slides")[0]
    img = Image.open(png).convert("L")
    ppi = img.width / 10.0  # default template is 10 in wide
    box = img.crop((round(1 * ppi), round(2 * ppi), round(9 * ppi), round(3 * ppi)))
    mean = sum(box.getdata()) / (box.width * box.height)
    assert mean < 128, f"equation box region is not the fallback picture (mean L={mean:.0f})"


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
