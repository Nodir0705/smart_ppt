"""CLI: build out/index.json from a .pptx file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .parser import parse_deck


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Smart Pointer — Stage 1 indexer")
    ap.add_argument("deck", help="path to .pptx")
    ap.add_argument("-o", "--out", default="out/index.json", help="output JSON path")
    args = ap.parse_args(argv)

    deck_path = Path(args.deck)
    if not deck_path.exists():
        print(f"deck not found: {deck_path}", file=sys.stderr)
        return 2
    index = parse_deck(str(deck_path))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    total_elements = sum(len(s["elements"]) for s in index["slides"])
    print(f"wrote {out_path} — {len(index['slides'])} slides, {total_elements} elements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
