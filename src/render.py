"""Render a .pptx deck to per-slide PNGs (offline, idempotent).

Pipeline:
    pptx --(libreoffice headless)--> pdf --(pdftoppm)--> slide_NNN.png

The renderer is cache-aware: if the output PNGs are newer than the input pptx
and the count matches the deck, it skips re-rendering.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from lxml import etree
from pptx import Presentation

_MC_NS = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"
_A_BLIP = "{http://schemas.openxmlformats.org/drawingml/2006/main}blip"
_SLIDE_RE = re.compile(r"ppt/slides/slide\d+\.xml")


def _inline_fallbacks(slide_xml: bytes) -> bytes | None:
    """Replace each `mc:AlternateContent` whose Fallback holds a picture by that
    Fallback's children, in place. None when the slide has none."""
    root = etree.fromstring(slide_xml)
    acs = [ac for ac in root.iter(_MC_NS + "AlternateContent")
           if ac.find(_MC_NS + "Fallback/.//" + _A_BLIP) is not None]
    if not acs:
        return None
    for ac in acs:
        parent = ac.getparent()
        i = parent.index(ac)
        for child in list(ac.find(_MC_NS + "Fallback")):
            parent.insert(i, child)
            i += 1
        parent.remove(ac)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def _fallback_slides(pptx_path: Path) -> dict[str, bytes]:
    """Slide parts (zip name -> xml) rewritten with each equation box's Fallback
    picture in place of its `mc:AlternateContent`; empty when the deck has none.
    (PowerPoint wraps a text box holding an equation that way; soffice renders
    the Choice but drops the equations, leaving gaps mid-sentence.)"""
    patched: dict[str, bytes] = {}
    with zipfile.ZipFile(pptx_path) as zin:
        for name in zin.namelist():
            if _SLIDE_RE.fullmatch(name):
                xml = _inline_fallbacks(zin.read(name))
                if xml is not None:
                    patched[name] = xml
    return patched


def _picture_copy(pptx_path: Path, patched: dict[str, bytes], tmp_dir: Path) -> Path:
    """A copy of the pptx in `tmp_dir` with the `patched` slide parts swapped in.
    Deterministic: original entry order and compression, fixed timestamps."""
    copy = tmp_dir / pptx_path.name
    with zipfile.ZipFile(pptx_path) as zin, zipfile.ZipFile(copy, "w") as zout:
        for info in zin.infolist():
            zi = zipfile.ZipInfo(info.filename, date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = info.compress_type
            zout.writestr(zi, patched[info.filename] if info.filename in patched
                          else zin.read(info))
    return copy


def _count_slides(pptx_path: Path) -> int:
    return len(Presentation(str(pptx_path)).slides)


def _existing_pngs(out_dir: Path) -> list[Path]:
    return sorted(out_dir.glob("slide_*.png"))


def _pngs_are_fresh(pptx_path: Path, out_dir: Path, expected_count: int) -> bool:
    pngs = _existing_pngs(out_dir)
    if len(pngs) != expected_count:
        return False
    pptx_mtime = pptx_path.stat().st_mtime
    return all(p.stat().st_mtime >= pptx_mtime for p in pngs)


def _pdf_is_fresh(pptx_path: Path, pdf_path: Path) -> bool:
    return pdf_path.exists() and pdf_path.stat().st_mtime >= pptx_path.stat().st_mtime


def deck_pdf(pptx_path: str | Path, pdf_dir: str | Path = "out", *,
             text_layer: bool = False) -> Path:
    """Produce and cache a PDF of the pptx in `pdf_dir` (mtime-keyed).

    `<stem>.pdf` (default) is the visual PDF the slide PNGs come from: an
    equation box (`mc:AlternateContent` with a picture Fallback) is rendered
    from that picture, so the equations show — as pixels, with no text layer.
    `text_layer=True` gives `<stem>.text.pdf`, rendered from the original pptx:
    soffice lays out the Choice text on the right lines (blank where the math
    is), which is what refine_text needs for glyph boxes. A deck without such
    boxes has one PDF serving both uses, and `text_layer=True` returns it.
    """
    pptx_path = Path(pptx_path).resolve()
    pdf_dir = Path(pdf_dir).resolve()
    pdf_dir.mkdir(parents=True, exist_ok=True)
    patched = _fallback_slides(pptx_path)
    text_layer = text_layer and bool(patched)
    pdf_path = pdf_dir / (pptx_path.stem + (".text.pdf" if text_layer else ".pdf"))
    if _pdf_is_fresh(pptx_path, pdf_path):
        return pdf_path

    with tempfile.TemporaryDirectory(prefix="smartppt_pdf_") as td:
        td_path = Path(td)
        # A private user profile per run: soffice serialises on its profile, so
        # two concurrent renders (another test run, another agent) would collide.
        profile = td_path / "profile"
        src = (_picture_copy(pptx_path, patched, td_path)
               if patched and not text_layer else pptx_path)
        r = subprocess.run(
            [
                "soffice", "--headless", "--nologo", "--nofirststartwizard",
                f"-env:UserInstallation={profile.as_uri()}",
                "--convert-to", "pdf", "--outdir", str(td_path), str(src),
            ],
            check=False, capture_output=True, text=True, timeout=120,
        )
        produced = td_path / (pptx_path.stem + ".pdf")
        if not produced.exists():
            raise RuntimeError(
                f"soffice failed to produce PDF.\nstdout: {r.stdout}\nstderr: {r.stderr}"
            )
        shutil.move(str(produced), str(pdf_path))
    return pdf_path


def render_deck(pptx_path: str | Path, out_dir: str | Path, dpi: int = 150) -> list[Path]:
    """Render `pptx_path` into `out_dir/slide_NNN.png`. Returns ordered PNG paths.

    The intermediate PDF is cached at `out_dir.parent/<stem>.pdf` (via `deck_pdf`)
    so downstream tools (e.g. text-bbox refinement) can reuse it.
    """
    pptx_path = Path(pptx_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = _count_slides(pptx_path)

    if _pngs_are_fresh(pptx_path, out_dir, expected):
        return _existing_pngs(out_dir)

    # Wipe stale PNGs so cache stays consistent
    for p in _existing_pngs(out_dir):
        p.unlink()

    # 1) pptx → pdf (cached at out_dir.parent/<stem>.pdf so refine_text reuses it)
    pdf_path = deck_pdf(pptx_path, out_dir.parent)

    with tempfile.TemporaryDirectory(prefix="smartppt_render_") as td:
        td_path = Path(td)
        # 2) pdf → png (pdftoppm writes <prefix>-1.png, -2.png, ...)
        prefix = td_path / "slide"
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), str(pdf_path), str(prefix)],
            check=True, capture_output=True, text=True, timeout=120,
        )
        # 3) Move into out_dir with zero-padded names (slide_001.png, ...)
        produced = sorted(td_path.glob("slide-*.png"))
        if len(produced) != expected:
            raise RuntimeError(
                f"pdftoppm produced {len(produced)} pages, expected {expected}"
            )
        for i, src in enumerate(produced, start=1):
            dst = out_dir / f"slide_{i:03d}.png"
            shutil.move(str(src), str(dst))

    return _existing_pngs(out_dir)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Render a deck to PNGs.")
    ap.add_argument("deck", help="path to .pptx")
    ap.add_argument("-o", "--out", default="out/slides")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--force", action="store_true", help="ignore cache")
    args = ap.parse_args(argv)
    if args.force:
        out = Path(args.out)
        if out.exists():
            for p in out.glob("slide_*.png"):
                p.unlink()
    pngs = render_deck(args.deck, args.out, dpi=args.dpi)
    print(f"rendered {len(pngs)} slides → {args.out}")
    for p in pngs:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
