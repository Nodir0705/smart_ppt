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


def _stub_soffice(monkeypatch) -> list[list[str]]:
    """Replace subprocess.run with a stub that writes the PDF soffice would;
    returns the list of argument lists it was called with."""
    calls: list[list[str]] = []

    def fake_run(args, **kw):
        calls.append(args)
        outdir = Path(args[args.index("--outdir") + 1])
        (outdir / (Path(args[-1]).stem + ".pdf")).write_bytes(b"%PDF-1.4\n")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_deck_pdf_gives_soffice_a_private_profile(tmp_path, monkeypatch):
    # soffice serialises on its user profile, so two concurrent renders (another
    # test run, another agent) collide unless each gets a fresh profile dir.
    calls = _stub_soffice(monkeypatch)
    deck_pdf(DECK, tmp_path)
    (args,) = calls
    assert any(a.startswith("-env:UserInstallation=file://") for a in args), args


def test_plain_deck_shares_one_pdf_for_pictures_and_text(tmp_path, monkeypatch):
    # No equation boxes -> the visual PDF already carries the text layer: the
    # text_layer call returns the same file, soffice ran once, on the deck itself.
    calls = _stub_soffice(monkeypatch)
    visual = deck_pdf(DECK, tmp_path)
    assert deck_pdf(DECK, tmp_path, text_layer=True) == visual
    assert len(calls) == 1 and calls[0][-1] == str(DECK)
    assert [p.name for p in tmp_path.iterdir()] == ["test_deck.pdf"]


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("soffice") is None, reason="soffice required")
def test_text_layer_pdf_keeps_the_equation_box_text(tmp_path):
    # The visual PDF draws an equation box from its Fallback picture — no glyphs.
    # refine needs glyph positions, so it gets a second PDF rendered from the
    # original pptx: the Choice text, laid out on the right lines (math blank).
    import pypdfium2 as pdfium
    from tools.make_corpus import build_corpus

    def page_text(pdf: Path) -> str:
        doc = pdfium.PdfDocument(str(pdf))
        try:
            return doc[0].get_textpage().get_text_range()
        finally:
            doc.close()

    deck = {d.name: d for d in build_corpus(tmp_path / "corpus")}["math_textbox"]
    visual = deck_pdf(deck.path, tmp_path)
    text = deck_pdf(deck.path, tmp_path, text_layer=True)
    assert text.name == "math_textbox.text.pdf" and visual.name == "math_textbox.pdf"
    assert "Let" in page_text(text) and "be languages" in page_text(text)
    assert "be languages" not in page_text(visual)
    assert "Regular languages" in page_text(visual)  # ordinary boxes keep their text


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("soffice") is None, reason="soffice required")
def test_table_math_fallback_renders_as_one_picture(tmp_path):
    # A table's math Fallback fills each cell with ONE picture of the whole
    # table; soffice stretches a cell fill into every cell, tiling the table
    # with miniatures of itself. The renderer must draw the picture once, at the
    # frame: the picture's red quadrant is then a single block half the frame
    # wide, not four quarter-wide blocks.
    from PIL import Image
    from tools.make_corpus import build_corpus
    deck = {d.name: d for d in build_corpus(tmp_path / "corpus")}["table_math_fallback"]
    png = render_deck(deck.path, tmp_path / "slides")[0]
    img = Image.open(png).convert("RGB")
    ppi = img.width / 10.0  # default template is 10 in wide; frame = (1, 2) 4x2 in
    frame = img.crop((round(1 * ppi), round(2 * ppi), round(5 * ppi), round(4 * ppi)))
    w, h = frame.size
    red = [(i % w, i // w) for i, (r, g, b) in enumerate(frame.getdata())
           if r > 180 and g < 90 and b < 90]
    assert red, "no red pixels: fallback picture not drawn at all"
    xs, ys = [p[0] for p in red], [p[1] for p in red]
    bw, bh = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
    fill = len(red) / (bw * bh)
    assert abs(bw / w - 0.5) < 0.1 and fill > 0.85, (
        f"red quadrant spans {bw / w:.2f} of the frame width with fill {fill:.2f} "
        f"(one block at the frame = 0.50 / ~1.0; tiled per cell = 0.75 / ~0.44)")


def test_build_index_refines_from_the_text_layer_pdf(tmp_path, monkeypatch):
    # The refine stage must read the text-layer PDF (the visual one has no
    # glyphs where equation boxes are). Wiring only: soffice stubbed, refine
    # captured.
    import src.build_index as B
    from tools.make_corpus import build_corpus
    deck = {d.name: d for d in build_corpus(tmp_path / "corpus")}["math_textbox"]
    _stub_soffice(monkeypatch)
    seen = []
    monkeypatch.setattr(B, "refine_index", lambda index, pdf: seen.append(Path(pdf)) or index)
    B.build_index(deck.path, tmp_path / "build", scope=False, render_slides=False, ocr=False)
    assert seen == [tmp_path / "build" / "math_textbox.text.pdf"]


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("soffice") is None, reason="soffice required")
def test_build_index_gives_equation_box_glyph_lines(tmp_path):
    # End to end: with the text-layer PDF an equation box should get glyph-
    # derived line boxes instead of the parser's guessed lines.
    from src.build_index import build_index
    from tools.make_corpus import build_corpus
    deck = {d.name: d for d in build_corpus(tmp_path / "corpus")}["math_textbox"]
    index = build_index(deck.path, tmp_path / "build", scope=False, render_slides=False,
                        ocr=False, vision_link=False)
    el = next(e for s in index["slides"] for e in s["elements"]
              if e["text"] == "Let A, B be languages")
    assert el["bbox_source"] == "pdf_text" and el["line_bboxes"], el


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
