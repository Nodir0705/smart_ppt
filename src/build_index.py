"""One-command index build: pptx → index.json → .refined → .scoped, in-process.

The three pipeline stages (parse → refine → scope) are pure functions; the
single-stage CLIs (`index_deck`, `refine_text`, `group_scope`) still exist for
debugging one stage in isolation. This module just chains them so onboarding a
new deck is one command instead of three remembered-in-order invocations.

The live app loads the richest available index (scoped > refined > base) via
`matcher.best_index`; the intermediates are written for inspection only.

    python3 -m src.build_index my_deck.pptx       # full build → out/index.scoped.json
    python3 -m src.build_index my_deck.pptx --no-scope   # stop at refined
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .group_scope import EXPANSIONS_PATH, expansions_for, scope_index
from .parser import parse_deck
from .refine_text import refine_index
from .render import deck_pdf, render_deck


def _write(path: Path, index: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2, ensure_ascii=False))


def build_index(
    deck: str | Path,
    out_dir: str | Path = "out",
    *,
    refine: bool = True,
    scope: bool = True,
    render_slides: bool = True,
    write_intermediates: bool = True,
    expansions_path: str = EXPANSIONS_PATH,
    ocr: bool = True,
    vision_link: bool = True,
    slide_map_dir: str | Path | None = None,
) -> dict:
    """Build the index for `deck` and return the richest produced stage.

    Each stage mutates-and-returns the same dict, so every intermediate is
    serialized to disk *before* the next stage runs — the on-disk files are
    correct snapshots of each stage. `refine`/`scope` require LibreOffice
    (soffice) on PATH; pass `refine=False` to skip the PDF/render dependency.
    `ocr=False` skips the tesseract fallback (bench "no-ocr" variant).
    `vision_link=False` skips CLIP picture->text linking (bench "no-clip" variant).
    `slide_map_dir` applies per-slide understanding maps (bench "map" variant).
    """
    deck = Path(deck)
    out = Path(out_dir)

    # OCR alt-text-less pictures when tesseract is present (parse_deck self-gates
    # and no-ops otherwise) so baked-in image text becomes pointable.
    index = parse_deck(str(deck), ocr=ocr)
    if write_intermediates:
        _write(out / "index.json", index)
    final = index

    if refine:
        slides_dir = out / "slides"
        if render_slides:
            # measured chart bars need the rendered PNGs; also caches the PDF
            render_deck(deck, slides_dir)
        pdf_path = deck_pdf(deck, out)
        final = refine_index(final, pdf_path)
        if slides_dir.exists():
            from .chart_detect import refine_chart_bboxes
            refine_chart_bboxes(final, str(slides_dir))
            if vision_link:
                from .vision_link import format_links, link_pictures
                links = link_pictures(final, slides_dir)
                if links:
                    print(format_links(links), flush=True)
        if write_intermediates:
            _write(out / "index.refined.json", final)

    if slide_map_dir is not None:
        from .slide_map import apply_slide_maps
        final = apply_slide_maps(final, Path(slide_map_dir))

    if scope:
        final = scope_index(final, expansions=expansions_for(final, expansions_path))
        if write_intermediates:
            _write(out / "index.scoped.json", final)

    return final


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build the smartppt index end-to-end (parse → refine → scope)."
    )
    ap.add_argument("deck", help="path to .pptx")
    ap.add_argument("-o", "--out", default="out", help="output directory (default: out)")
    ap.add_argument("--no-refine", action="store_true",
                    help="stop after parsing (skips the soffice PDF dependency)")
    ap.add_argument("--no-scope", action="store_true",
                    help="stop after refinement (no group/expansion phrases)")
    ap.add_argument("--no-render", action="store_true",
                    help="skip slide rendering (chart bars fall back to the heuristic)")
    ap.add_argument("--expansions", default=EXPANSIONS_PATH,
                    help=f"expansions JSON (default: {EXPANSIONS_PATH})")
    args = ap.parse_args(argv)

    deck_path = Path(args.deck)
    if not deck_path.exists():
        print(f"deck not found: {deck_path}", file=sys.stderr)
        return 2

    refine = not args.no_refine
    scope = not args.no_scope and refine  # scoping consumes the refined stage
    final = build_index(
        deck_path,
        args.out,
        refine=refine,
        scope=scope,
        render_slides=not args.no_render,
        expansions_path=args.expansions,
    )

    stage = "scoped" if scope else ("refined" if refine else "base")
    n_slides = len(final["slides"])
    n_elements = sum(len(s["elements"]) for s in final["slides"])
    n_phrases = sum(len(e.get("trigger_phrases", []))
                    for s in final["slides"] for e in s["elements"])
    print(f"built {stage} index — {n_slides} slides, {n_elements} elements, "
          f"{n_phrases} trigger phrases → {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
