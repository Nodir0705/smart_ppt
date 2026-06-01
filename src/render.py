"""Render a .pptx deck to per-slide PNGs (offline, idempotent).

Pipeline:
    pptx --(libreoffice headless)--> pdf --(pdftoppm)--> slide_NNN.png

The renderer is cache-aware: if the output PNGs are newer than the input pptx
and the count matches the deck, it skips re-rendering.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from pptx import Presentation


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


def render_deck(pptx_path: str | Path, out_dir: str | Path, dpi: int = 150) -> list[Path]:
    """Render `pptx_path` into `out_dir/slide_NNN.png`. Returns ordered PNG paths."""
    pptx_path = Path(pptx_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = _count_slides(pptx_path)

    if _pngs_are_fresh(pptx_path, out_dir, expected):
        return _existing_pngs(out_dir)

    # Wipe stale PNGs so cache stays consistent
    for p in _existing_pngs(out_dir):
        p.unlink()

    with tempfile.TemporaryDirectory(prefix="smartppt_render_") as td:
        td_path = Path(td)
        # 1) pptx → pdf
        r = subprocess.run(
            [
                "soffice", "--headless", "--nologo", "--nofirststartwizard",
                "--convert-to", "pdf", "--outdir", str(td_path), str(pptx_path),
            ],
            check=False, capture_output=True, text=True, timeout=120,
        )
        pdf_path = td_path / (pptx_path.stem + ".pdf")
        if not pdf_path.exists():
            raise RuntimeError(
                f"soffice failed to produce PDF.\nstdout: {r.stdout}\nstderr: {r.stderr}"
            )
        # 2) pdf → png (pdftoppm writes <prefix>-1.png, -2.png, ...)
        prefix = td_path / "slide"
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), str(pdf_path), str(prefix)],
            check=True, capture_output=True, text=True, timeout=120,
        )
        # 3) Move into out_dir with zero-padded names (slide_001.png, slide_002.png, ...)
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
