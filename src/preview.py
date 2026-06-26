"""Headless preview — composite the highlight onto the slide PNG with PIL.

Useful when you can't open the Tk viewer (CI, remote shell) but want to see
where the matcher is pointing for a given spoken phrase. Produces a PNG.

Example:
    python3 -m src.preview --slide 6 --phrase "let me explain the matcher" \
        --out /tmp/preview.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from .encoder import load_default_encoder
from .geom import inches_to_image_px
from .matcher import Index, Matcher


HIGHLIGHT_HALO = (40, 40, 40)           # dark outline box around the active element
HIGHLIGHT_WIDTH = 8                     # outline thickness in image px


def render_preview(
    index_path: Path,
    slides_dir: Path,
    slide_no: int,
    phrase: str,
    out_path: Path,
) -> dict:
    raw_index = json.loads(index_path.read_text())
    slide_w_in = raw_index["slide_size"]["w"]
    slide_h_in = raw_index["slide_size"]["h"]

    enc = load_default_encoder()
    idx = Index(index_path, enc)
    matcher = Matcher(idx, enc)
    res = matcher.match(phrase, slide_no)

    png_path = slides_dir / f"slide_{slide_no:03d}.png"
    if not png_path.exists():
        raise FileNotFoundError(png_path)
    img = Image.open(png_path).convert("RGB")

    info = {
        "slide": slide_no,
        "phrase": phrase,
        "matched_id": res.element_id,
        "matched_text": res.text,
        "score": res.score,
        "runner_up_score": res.runner_up_score,
    }

    if res.element_id is None:
        img.save(out_path)
        return info

    # find the element bbox in inches
    el_bbox_in = None
    for slide in raw_index["slides"]:
        if slide["slide_number"] != slide_no:
            continue
        for el in slide["elements"]:
            if el["id"] == res.element_id:
                bb = el["bbox"]
                el_bbox_in = (bb["x"], bb["y"], bb["w"], bb["h"])
                break
    assert el_bbox_in is not None

    x, y, w, h = inches_to_image_px(el_bbox_in, slide_w_in, slide_h_in, img.width, img.height)
    draw = ImageDraw.Draw(img)
    # single dark outline box around the active element
    draw.rectangle(
        [x, y, x + w, y + h],
        outline=HIGHLIGHT_HALO, width=5,
    )
    img.save(out_path)
    info["bbox_image_px"] = (round(x, 1), round(y, 1), round(w, 1), round(h, 1))
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Headless overlay preview.")
    ap.add_argument("--index", default="out/index.json")
    ap.add_argument("--slides", default="out/slides")
    ap.add_argument("--slide", type=int, required=True)
    ap.add_argument("--phrase", required=True)
    ap.add_argument("--out", default="/tmp/preview.png")
    args = ap.parse_args(argv)
    info = render_preview(
        index_path=Path(args.index),
        slides_dir=Path(args.slides),
        slide_no=args.slide,
        phrase=args.phrase,
        out_path=Path(args.out),
    )
    print(f"wrote {args.out}")
    for k, v in info.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
