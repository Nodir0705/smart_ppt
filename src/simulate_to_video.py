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
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path

from PIL import Image, ImageDraw

from .audio import PlaybackSource
from .encoder import load_default_encoder
from .geom import Bbox, FitTransform, ease_in_out_cubic, inches_to_image_px, lerp_bbox
from .live import LiveTranscriber
from .matcher import Index, Matcher, MatchPolicy, PolicyConfig
from .stt import RollingBuffer, VoskEngine, hotwords_from_index


# match the Tk viewer's highlight look (src/viewer.py)
HALO = (40, 40, 40)
HIGHLIGHT_HALO_WIDTH = 3

# animation parameters — same as viewer
ANIM_FRAMES = 12
ANIM_FPS = 30  # frames-per-second of the output video


def _wav_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def _record_events(slide_no: int, wav_path: Path, raw_index: dict,
                   engine: VoskEngine, matcher: Matcher,
                   hotwords: list[str], tail_words: int = 8,
                   normalizer=None) -> list[tuple[float, str | None]]:
    """Run the full pipeline against a muted PlaybackSource; return a list of
    (timestamp_seconds, active_element_id) transitions paced by audio time.
    """
    policy = MatchPolicy(matcher, PolicyConfig(), normalizer=normalizer)
    buf = RollingBuffer(window_seconds=4.0)
    src = PlaybackSource(str(wav_path), mute=True)
    transcriber = LiveTranscriber(src, engine, buf, hotwords=hotwords)

    events: list[tuple[float, str | None]] = []
    last_active: str | None = None
    t0 = time.monotonic()

    def on_update(text, partial):
        nonlocal last_active
        tail = buf.tail(tail_words)
        if not tail:
            return
        _res, active, _why = policy.step(tail, slide_no)
        if active != last_active:
            events.append((time.monotonic() - t0, active))
            last_active = active

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


def _render_video_frames(slide_no: int, wav_path: Path, slides_dir: Path,
                         raw_index: dict, events: list[tuple[float, str | None]],
                         frames_dir: Path, fps: int, max_width: int) -> int:
    """Render all frames for one slide into `frames_dir/00001.png … `.

    Returns the number of frames written. Frames are at `max_width` × h
    (preserving aspect) so the encoded MP4 isn't enormous.
    """
    png_path = slides_dir / f"slide_{slide_no:03d}.png"
    base = Image.open(png_path).convert("RGB")
    if base.width > max_width:
        new_h = round(base.height * max_width / base.width)
        base = base.resize((max_width, new_h), Image.LANCZOS)
    img_w, img_h = base.size

    # Pre-compute target bboxes from events
    targets: list[tuple[float, Bbox | None]] = [
        (t, _element_bbox_image_px(raw_index, slide_no, eid, img_w, img_h))
        for t, eid in events
    ]

    dur = _wav_duration_s(wav_path)
    n_frames = int(dur * fps) + 1
    frame_dt = 1.0 / fps

    current: Bbox | None = None
    anim_start: Bbox | None = None
    anim_target: Bbox | None = None
    anim_step = 0
    event_idx = 0

    for i in range(n_frames):
        t = i * frame_dt

        # consume any events that should have fired by time t
        while event_idx < len(targets) and targets[event_idx][0] <= t:
            new_t = targets[event_idx][1]
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

        frame = base.copy()
        if current is not None:
            _draw_highlight(frame, current)
        frame.save(frames_dir / f"{i:05d}.png")

    return n_frames


def _encode_video(frames_dir: Path, wav_path: Path, out_path: Path, fps: int) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%05d.png"),
        "-i", str(wav_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{r.stderr[-2000:]}")


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

    print("loading models (BGE + Vosk)…")
    enc = load_default_encoder()
    idx = Index(index_path, enc)
    matcher = Matcher(idx, enc)
    engine = VoskEngine(sample_rate=16000)
    hotwords = hotwords_from_index(raw_index)
    # same deterministic ASR alias normalization the live app uses
    from .asr_normalize import build_alias_map, normalize as _norm
    _alias = build_alias_map(raw_index)
    normalizer = (lambda t: _norm(t, _alias)) if _alias else None

    slide_videos: list[Path] = []
    for item in manifest:
        sno = item["slide"]
        wav = Path(item["wav"])
        out_mp4 = out_dir / f"slide_{sno:03d}.mp4"
        print(f"\nslide {sno}: {wav.name}")
        print("  running simulation…")
        events = _record_events(sno, wav, raw_index, engine, matcher, hotwords,
                                normalizer=normalizer)
        for t, eid in events:
            print(f"    t={t:5.1f}s  active={eid}")
        with tempfile.TemporaryDirectory(prefix="smartppt_frames_") as td:
            frames_dir = Path(td)
            print(f"  rendering frames @ {args.fps} fps…")
            n = _render_video_frames(
                sno, wav, slides_dir, raw_index, events,
                frames_dir, args.fps, args.max_width,
            )
            print(f"  encoding {n} frames + audio → {out_mp4.name}")
            _encode_video(frames_dir, wav, out_mp4, args.fps)
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
