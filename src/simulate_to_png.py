"""Headless simulation that produces visual artifacts.

Runs the same TTS → PlaybackSource(muted) → STT → Matcher → Policy pipeline
as the live simulator, but instead of drawing in a Tk window it composites
the ordered list of visited elements onto each slide PNG, with numbered
amber boxes (1, 2, 3, …) so you can see at a glance which elements the
narrator's voice triggered and in what order.

Output: one PNG per slide in `out/simulated/`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .audio import PlaybackSource
from .encoder import load_default_encoder
from .geom import inches_to_image_px
from .live import LiveTranscriber
from .live_controller import LiveController
from .matcher import Index, Matcher
from .stt import STTEngine, load_default_stt_engine, hotwords_from_index


HALO = (31, 31, 31)
AMBER = (255, 176, 0)  # two-tone box, same as the viewer: dark edge + amber line
LABEL_BG = (255, 212, 0)
LABEL_FG = (20, 20, 20)


def _pick_font(size: int) -> ImageFont.FreeTypeFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _draw_visit(img: Image.Image, bbox_px, order: int, font) -> None:
    x, y, w, h = bbox_px
    draw = ImageDraw.Draw(img)
    # single dark outline box around the active element
    # drawn OUTSIDE the element (PIL widths grow inward) so it never covers the text
    p = 6
    draw.rectangle([x - p, y - p, x + w + p, y + h + p], outline=HALO, width=6)
    draw.rectangle([x - p, y - p, x + w + p, y + h + p], outline=AMBER, width=3)
    # numbered badge in top-left of the box
    badge_d = 56
    bx, by = x - 8, y - 8
    draw.ellipse([bx, by, bx + badge_d, by + badge_d], fill=LABEL_BG, outline=HALO, width=3)
    label = str(order)
    # measure
    l, t, r, b = draw.textbbox((0, 0), label, font=font)
    tw, th = r - l, b - t
    draw.text((bx + (badge_d - tw) / 2 - l, by + (badge_d - th) / 2 - t),
              label, fill=LABEL_FG, font=font)


def simulate_slide(slide_no: int, wav_path: Path, slides_dir: Path,
                   raw_index: dict, engine: STTEngine, matcher: Matcher,
                   hotwords: list[str], out_path: Path,
                   tail_words: int = 8) -> dict:
    controller = LiveController(matcher, raw_index, tail_words=tail_words, slide_no=slide_no)
    buf = controller.buffer
    src = PlaybackSource(str(wav_path), mute=True)
    transcriber = LiveTranscriber(src, engine, buf, hotwords=hotwords)

    # Build element lookups for this slide
    el_meta: dict[str, dict] = {}
    for s in raw_index["slides"]:
        if s["slide_number"] != slide_no:
            continue
        for el in s["elements"]:
            bb = el["bbox"]
            el_meta[el["id"]] = {
                "type": el["type"],
                "text": el.get("text"),
                "bbox_in": (bb["x"], bb["y"], bb["w"], bb["h"]),
            }

    visits: list[tuple[float, str, dict | None]] = []
    last_key: tuple | None = None
    t0 = time.monotonic()

    def on_update(text, partial):
        nonlocal last_key
        ev = controller.feed()
        if ev is None:
            return
        key = (ev.active_id, ev.line_index)
        if ev.active_id and key != last_key:
            visits.append((time.monotonic() - t0, ev.active_id, ev.bbox))
            last_key = key

    transcriber.run(on_update=on_update)

    # Composite the slide image
    png_path = slides_dir / f"slide_{slide_no:03d}.png"
    img = Image.open(png_path).convert("RGB")
    slide_w_in = raw_index["slide_size"]["w"]
    slide_h_in = raw_index["slide_size"]["h"]
    font = _pick_font(34)
    for i, (_t, eid, line_bbox) in enumerate(visits, start=1):
        meta = el_meta.get(eid)
        if not meta:
            continue
        # prefer the focused wrapped-line sub-box when the controller picked one
        bbox_in = ((line_bbox["x"], line_bbox["y"], line_bbox["w"], line_bbox["h"])
                   if line_bbox else meta["bbox_in"])
        bbox_px = inches_to_image_px(bbox_in, slide_w_in, slide_h_in,
                                     img.width, img.height)
        _draw_visit(img, bbox_px, i, font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return {
        "slide": slide_no,
        "wav": str(wav_path),
        "out": str(out_path),
        "visits": [
            {"order": i + 1, "t": round(t, 1), "id": eid,
             "type": el_meta.get(eid, {}).get("type"),
             "text": el_meta.get(eid, {}).get("text")}
            for i, (t, eid, _lb) in enumerate(visits)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render highlight-trail PNGs from the TTS simulation.")
    ap.add_argument("--index", default="out/index.json")
    ap.add_argument("--slides", default="out/slides")
    ap.add_argument("--narration", default="out/narration")
    ap.add_argument("--out", default="out/simulated")
    args = ap.parse_args(argv)

    index_path = Path(args.index)
    slides_dir = Path(args.slides)
    narration_dir = Path(args.narration)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_index = json.loads(index_path.read_text())
    manifest_path = narration_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    print("loading models (BGE + STT)…")
    enc = load_default_encoder()
    idx = Index(index_path, enc)
    matcher = Matcher(idx, enc)
    engine = load_default_stt_engine(sample_rate=16000)
    hotwords = hotwords_from_index(raw_index)

    summary: list[dict] = []
    for item in manifest:
        sno = item["slide"]
        wav = Path(item["wav"])
        out_png = out_dir / f"slide_{sno:03d}_simulated.png"
        print(f"\nslide {sno}: {wav.name}")
        info = simulate_slide(sno, wav, slides_dir, raw_index,
                              engine, matcher, hotwords, out_png)
        for v in info["visits"]:
            print(f"  {v['order']:2d}. t={v['t']:5.1f}s  {v['id']:8s}  {(v['text'] or '')[:40]!r}")
        print(f"  → wrote {out_png}")
        summary.append(info)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {len(summary)} composite PNGs → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
