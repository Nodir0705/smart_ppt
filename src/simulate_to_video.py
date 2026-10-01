"""Render the live simulation as MP4 videos — slide image + narration audio
+ frame-by-frame animated highlight, exactly as the Tk viewer would show.

Outputs:
    out/video/slide_NNN.mp4   one per slide
    out/video/demo.mp4        all slides concatenated

How it works:
  1. Run the same pipeline as `src.simulate_to_png` (TTS → muted PlaybackSource
     → STT → matcher → policy). Record every (timestamp, active_element_id)
     transition, paced by audio time.
  2. Convert each transition into a target bbox in image-pixel space.
  3. Render frames at `fps`: for each frame compute the *interpolated*
     highlight bbox using the same lerp+ease_in_out_cubic logic the Tk
     viewer uses, draw it onto a copy of the slide PNG.
  4. Pipe the frames through ffmpeg with the narration WAV to produce MP4s.

This is the headless equivalent of running `src.simulate` on a machine with
a display — same animation, same timings, just baked to video so you can
watch it back.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
import wave
from pathlib import Path

from PIL import Image, ImageDraw

from .audio import PlaybackSource
from .encoder import load_default_encoder
from .geom import Bbox, ease_in_out_cubic, inches_to_image_px, lerp_bbox
from .live import LiveTranscriber
from .live_controller import LiveController
from .matcher import Index, Matcher
from .stt import STTEngine, load_default_stt_engine, hotwords_from_index


# match the Tk viewer's highlight look (src/viewer.py)
HALO = (40, 40, 40)
HIGHLIGHT_HALO_WIDTH = 3

# animation parameters — same as viewer
ANIM_FRAMES = 8   # highlight glide frames (~0.27s @30fps; snappier, was 12)
ANIM_FPS = 30  # frames-per-second of the output video


def _wav_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def _record_events(slide_no: int, wav_path: Path, raw_index: dict,
                   engine: STTEngine, matcher: Matcher,
                   hotwords: list[str], tail_words: int = 8,
                   normalizer=None) -> list[tuple[float, str | None, dict | None]]:
    """Run the full pipeline against a muted PlaybackSource; return a list of
    (timestamp_seconds, active_element_id) transitions paced by audio time.
    """
    controller = LiveController(matcher, raw_index, normalizer=normalizer,
                                tail_words=tail_words, slide_no=slide_no)
    buf = controller.buffer
    src = PlaybackSource(str(wav_path), mute=True)
    transcriber = LiveTranscriber(src, engine, buf, hotwords=hotwords)

    events: list[tuple[float, str | None, dict | None]] = []
    last_key: tuple | None = None
    t0 = time.monotonic()

    def on_update(text, partial):
        nonlocal last_key
        ev = controller.feed()
        if ev is None:
            return
        key = (ev.active_id, ev.line_index)
        if key != last_key:
            events.append((time.monotonic() - t0, ev.active_id, ev.line_bbox))
            last_key = key

    transcriber.run(on_update=on_update)
    return events


def _draw_highlight(img: Image.Image, bbox_px: Bbox) -> None:
    x, y, w, h = bbox_px
    if w <= 0 or h <= 0:
        return
    draw = ImageDraw.Draw(img)
    draw.rectangle([x, y, x + w, y + h], outline=HALO, width=HIGHLIGHT_HALO_WIDTH)


def _element_bbox_image_px(raw_index: dict, slide_no: int, element_id: str,
                           img_w: int, img_h: int) -> Bbox | None:
    if element_id is None:
        return None
    slide_w_in = raw_index["slide_size"]["w"]
    slide_h_in = raw_index["slide_size"]["h"]
    for s in raw_index["slides"]:
        if s["slide_number"] != slide_no:
            continue
        for el in s["elements"]:
            if el["id"] == element_id:
                bb = el["bbox"]
                return inches_to_image_px(
                    (bb["x"], bb["y"], bb["w"], bb["h"]),
                    slide_w_in, slide_h_in, img_w, img_h,
                )
    return None


def _element_axis_label_px(raw_index: dict, slide_no: int, element_id: str,
                           img_w: int, img_h: int) -> Bbox | None:
    """The element's x-axis label box (chart points), in image px, or None."""
    if element_id is None:
        return None
    slide_w_in = raw_index["slide_size"]["w"]
    slide_h_in = raw_index["slide_size"]["h"]
    for s in raw_index["slides"]:
        if s["slide_number"] != slide_no:
            continue
        for el in s["elements"]:
            if el["id"] == element_id and el.get("axis_label_bbox"):
                lb = el["axis_label_bbox"]
                return inches_to_image_px(
                    (lb["x"], lb["y"], lb["w"], lb["h"]),
                    slide_w_in, slide_h_in, img_w, img_h,
                )
    return None


def _render_and_encode(slide_no: int, wav_path: Path, slides_dir: Path,
                       raw_index: dict, events: list[tuple[float, str | None, dict | None]],
                       out_path: Path, fps: int, max_width: int) -> int:
    """Render highlight frames and stream them straight into ffmpeg.

    Frames (RGB, `max_width` × h, aspect-preserved) are generated in Python and
    piped to ffmpeg as raw rgb24 over stdin — no per-frame PNG encode and no
    temp directory (the old path wrote ~3,300 PNGs per slide, ~86% identical).
    x264 is pinned to `-threads 1` because its multi-threaded output is not
    byte-deterministic. Returns the number of frames written.
    """
    png_path = slides_dir / f"slide_{slide_no:03d}.png"
    base = Image.open(png_path).convert("RGB")
    if base.width > max_width:
        new_h = round(base.height * max_width / base.width)
        base = base.resize((max_width, new_h), Image.LANCZOS)
    img_w, img_h = base.size

    # Pre-compute target bboxes from events (data box + its x-axis label box).
    # A focused wrapped-line sub-box, when present, overrides the union box.
    def _target_box(eid, line_bbox):
        if line_bbox is not None:
            return inches_to_image_px(
                (line_bbox["x"], line_bbox["y"], line_bbox["w"], line_bbox["h"]),
                raw_index["slide_size"]["w"], raw_index["slide_size"]["h"],
                img_w, img_h,
            )
        return _element_bbox_image_px(raw_index, slide_no, eid, img_w, img_h)

    targets: list[tuple[float, Bbox | None]] = [
        (t, _target_box(eid, line_bbox)) for t, eid, line_bbox in events
    ]
    label_targets: list[Bbox | None] = [
        _element_axis_label_px(raw_index, slide_no, eid, img_w, img_h)
        for _t, eid, _lb in events
    ]

    dur = _wav_duration_s(wav_path)
    n_frames = int(dur * fps) + 1
    frame_dt = 1.0 / fps

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{img_w}x{img_h}",
        "-framerate", str(fps), "-i", "-",          # frames from stdin
        "-i", str(wav_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
        "-threads", "1",                              # deterministic x264 output
        "-c:a", "aac", "-b:a", "128k",
        "-shortest", "-movflags", "+faststart",
        str(out_path),
    ]

    current: Bbox | None = None
    current_label: Bbox | None = None     # static companion box on the x-axis label
    anim_start: Bbox | None = None
    anim_target: Bbox | None = None
    anim_step = 0
    event_idx = 0

    # Frame-bytes memo: only the ~14% of frames where the highlight changes (the
    # 12-frame transitions) need re-rendering; the static runs between them are
    # byte-identical, so render each unique frame once and re-pipe its bytes — skips
    # a full base.copy() + draw + tobytes() (~2.7MB) on every static frame.
    _UNSET = object()
    last_state = _UNSET
    last_bytes: bytes | None = None
    rendered = 0

    # stderr -> temp file so a chatty ffmpeg can't deadlock the stdin pipe.
    with tempfile.TemporaryFile() as errf:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errf)
        try:
            for i in range(n_frames):
                t = i * frame_dt

                # consume any events that should have fired by time t
                while event_idx < len(targets) and targets[event_idx][0] <= t:
                    new_t = targets[event_idx][1]
                    current_label = label_targets[event_idx]  # snaps with the element
                    if current is None and new_t is not None:
                        # appearing from nowhere — pop in, no animation
                        current = new_t
                        anim_target = new_t
                        anim_step = ANIM_FRAMES  # done
                    elif new_t is None:
                        current = None
                        anim_target = None
                        anim_step = ANIM_FRAMES
                    else:
                        # animate from current to new_t
                        anim_start = current
                        anim_target = new_t
                        anim_step = 0
                    event_idx += 1

                # advance animation
                if anim_target is not None and anim_step < ANIM_FRAMES and anim_start is not None:
                    anim_step += 1
                    eased = ease_in_out_cubic(anim_step / ANIM_FRAMES)
                    current = lerp_bbox(anim_start, anim_target, eased)
                    if anim_step >= ANIM_FRAMES:
                        current = anim_target

                state = (current, current_label)
                if state != last_state:
                    frame = base.copy()
                    if current is not None:
                        _draw_highlight(frame, current)
                    if current_label is not None:
                        _draw_highlight(frame, current_label)
                    last_bytes = frame.tobytes()
                    last_state = state
                    rendered += 1
                proc.stdin.write(last_bytes)
            proc.stdin.close()
            print(f"  rendered {rendered}/{n_frames} unique frames "
                  f"({n_frames - rendered} static frames reused)")
        except BrokenPipeError:
            pass  # ffmpeg died early — surfaced via returncode/stderr below
        ret = proc.wait()
        if ret != 0:
            errf.seek(0)
            raise RuntimeError(f"ffmpeg failed:\n{errf.read().decode(errors='replace')[-2000:]}")

    return n_frames


def _concat_videos(parts: list[Path], out_path: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in parts:
            f.write(f"file '{p.resolve()}'\n")
        list_path = f.name
    try:
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
            "-c", "copy", "-movflags", "+faststart", str(out_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed:\n{r.stderr[-2000:]}")
    finally:
        Path(list_path).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render the simulation as MP4s.")
    ap.add_argument("--index", default="out/index.json")
    ap.add_argument("--slides", default="out/slides")
    ap.add_argument("--narration", default="out/narration")
    ap.add_argument("--out", default="out/video")
    ap.add_argument("--fps", type=int, default=ANIM_FPS)
    ap.add_argument("--max-width", type=int, default=1280, help="video width in pixels")
    ap.add_argument("--only", type=int, default=None, help="render only this slide number")
    args = ap.parse_args(argv)

    index_path = Path(args.index)
    slides_dir = Path(args.slides)
    narration_dir = Path(args.narration)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_index = json.loads(index_path.read_text())
    manifest = json.loads((narration_dir / "manifest.json").read_text())
    if args.only is not None:
        manifest = [m for m in manifest if m["slide"] == args.only]

    print("loading models (BGE + STT)…")
    enc = load_default_encoder()
    idx = Index(index_path, enc)
    matcher = Matcher(idx, enc)
    engine = load_default_stt_engine(sample_rate=16000)
    hotwords = hotwords_from_index(raw_index)

    slide_videos: list[Path] = []
    for item in manifest:
        sno = item["slide"]
        wav = Path(item["wav"])
        out_mp4 = out_dir / f"slide_{sno:03d}.mp4"
        print(f"\nslide {sno}: {wav.name}")
        print("  running simulation…")
        events = _record_events(sno, wav, raw_index, engine, matcher, hotwords)
        for t, eid, lb in events:
            tag = " (line)" if lb else ""
            print(f"    t={t:5.1f}s  active={eid}{tag}")
        print(f"  rendering + encoding @ {args.fps} fps → {out_mp4.name}")
        n = _render_and_encode(
            sno, wav, slides_dir, raw_index, events,
            out_mp4, args.fps, args.max_width,
        )
        print(f"  streamed {n} frames")
        slide_videos.append(out_mp4)
        print(f"  ✓ {out_mp4} ({out_mp4.stat().st_size//1024} KB)")

    if len(slide_videos) > 1:
        master = out_dir / "demo.mp4"
        print(f"\nconcatenating {len(slide_videos)} clips → {master}")
        _concat_videos(slide_videos, master)
        print(f"  ✓ {master} ({master.stat().st_size//1024} KB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
