"""Tests for PDF caching (render.deck_pdf) and text-bbox refinement."""
from __future__ import annotations

import shutil

import pytest

SOFFICE_MISSING = shutil.which("soffice") is None


@pytest.mark.skipif(SOFFICE_MISSING, reason="soffice not installed")
def test_deck_pdf_caches(tmp_path):
    from src.render import deck_pdf

    pdf = deck_pdf("test_deck.pptx", tmp_path)
    assert pdf.exists()
    assert pdf.suffix == ".pdf"
    mtime = pdf.stat().st_mtime

    # Second call must be a cache hit: same path, not regenerated.
    pdf2 = deck_pdf("test_deck.pptx", tmp_path)
    assert pdf2 == pdf
    assert pdf2.stat().st_mtime == mtime
