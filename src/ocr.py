"""ocr.py — deterministic, offline, CPU-only OCR fallback for image elements.

A picture with no author alt text but text baked into the raster (a screenshot,
an exported chart, a labelled diagram) is otherwise unreachable — the presenter
can say "the net revenue box" but nothing in the index matches. Tesseract (Apache
2.0, fully local) recovers that text so the image becomes pointable WITHOUT a
cloud VLM.

This output feeds trigger_phrases -> embeddings -> matches -> highlights, so it
is a PINNED BUILD-TIME INPUT in the same class as the encoder and the soffice PDF
layer: reproducibility is scoped per (tesseract version, eng.traineddata). Pin
those like any other model input. (Tesseract + traineddata belong in the project
dep-manifest — see the optimization audit's "no dep manifest" gap.)

Precision over recall: OCR readily hallucinates tokens on text-LESS but visually
structured images (shapes/lines), which would create a false pointable that
lights up — the exact false-positive failure that got hotword biasing and
lexical fusion rejected. Two deterministic filters defend against it:
  - per-word confidence >= MIN_CONF, and
  - >= MIN_ALPHA_CHARS alphabetic characters in the word
(verified: suppresses tesseract's "OS" hallucination on a shapes-only image
while keeping a clean "NET REVENUE" label).
"""

from __future__ import annotations

import csv
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

_log = logging.getLogger(__name__)

# psm 3 = automatic page segmentation; on a shapes-only image it emitted nothing
# (psm 6/11 hallucinated), and it still reads a centred label. Scattered diagram
# labels may want psm 11 — under-determined by a single test image, so left as a
# tunable rather than over-fit.
DEFAULT_PSM = 3
MIN_CONF = 60.0        # tesseract per-word confidence (0–100)
MIN_ALPHA_CHARS = 3    # reject 1–2 letter scraps ("OS") that survive on noise
_TESS = "tesseract"


def tesseract_available() -> bool:
    """True if the tesseract binary is on PATH."""
    return shutil.which(_TESS) is not None


def _keep(token: str) -> bool:
    return len(re.sub(r"[^A-Za-z]", "", token)) >= MIN_ALPHA_CHARS


def ocr_image_text(blob: bytes, *, psm: int = DEFAULT_PSM,
                   min_conf: float = MIN_CONF) -> str:
    """Return high-confidence text read from image `blob`, or "" if none.

    Deterministic for a fixed (tesseract version, traineddata). Never raises:
    any failure (missing binary, unreadable image, tesseract error) yields "".
    """
    if not blob or not tesseract_available():
        return ""
    try:
        with tempfile.TemporaryDirectory(prefix="smartppt_ocr_") as td:
            img = Path(td) / "img"
            img.write_bytes(blob)
            r = subprocess.run(
                [_TESS, str(img), "stdout", "--psm", str(psm), "-l", "eng", "tsv"],
                capture_output=True, text=True, timeout=60, check=False,
            )
            if r.returncode != 0:
                _log.warning("tesseract exited %d; skipping OCR", r.returncode)
                return ""
            words: list[str] = []
            for row in csv.DictReader(r.stdout.splitlines(), delimiter="\t"):
                try:
                    conf = float(row.get("conf", "-1"))
                except (TypeError, ValueError):
                    continue
                token = (row.get("text") or "").strip()
                if conf >= min_conf and token and _keep(token):
                    words.append(token)
            return re.sub(r"\s+", " ", " ".join(words)).strip()
    except Exception:  # pragma: no cover - defensive; OCR is best-effort
        _log.warning("OCR failed on an image; leaving it needs_vision", exc_info=True)
        return ""
