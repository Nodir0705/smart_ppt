"""Render step: cue sheet + word timings + slide PNG -> ``out/video/<deck>/slide_NNN.mp4``,
then ``deck.mp4``. Deterministic: every frame is ``draw(base, state_at(i / fps))``.

- Frames: 30 fps, ``--size`` (default 1920x1080), the slide PNG letterboxed on
  near-black. Consecutive frames with an equal ``state_at`` reuse the previous
  frame's bytes. They are piped as raw rgb24 into ffmpeg as in
  ``src.simulate_to_video._render_and_encode`` (``-threads 1`` x264, but
  ``-crf 20``), with the slide wav as aac 128k, ``-shortest``.
- Slides = the scripted ones (``script/<deck>/slide_NNN.md``). One without a
  cue sheet, timings, voice wav or index entry is refused with the reason; an
  empty script is skipped (voice skips it too) with a printed line. The wav is
  always ``voice/slide_NNN.wav`` (the timings json's ``wav`` is informational).
- ``deck.mp4``: one ffmpeg concat-filter pass over the slide MP4s; every slide
  but the last gets ``tpad`` (last frame held) + ``apad`` (silence) of 0.6 s.
  Built only when no slide was refused, so it is never missing a slide (the
  old deck is left alone); skipped under ``--only``.
- Hash cache: ``slide_NNN.hash`` = sha1(cue sheet, script words, timings json,
  index slide, VERSION, size, fps, PNG bytes); ``deck.hash`` = sha1(slide hashes, gap).

Decisions where the plan left details open:
- Invalid cues (spec §4): failing cues are dropped and the rest re-validated
  until clean (dropping a focus can orphan a later glide). Each round drops the
  cues with a cue-local error first; overlap errors (``concurrency``, ``zoom
  overlaps``) are dropped only when nothing else fails, since a cue dropped for
  another reason may be what pushed them over the limit. The drops are printed
  with their original cue index and saved to ``slide_NNN.dropped.txt``, which a
  cached rerun prints again. A sheet-level error (no cue index) refuses the
  slide. Timings whose words differ from the script words are stale and
  refuse the slide ("rerun voice").
- Effect boxes: a target's (unit union or element) bbox padded 6 px; underline
  uses the unpadded line box; a glide starts from the box of the focus/glide
  before it and keeps that focus's spotlight. Zoom: the target box (or its
  ``region``) is the source, the lens avoids every pointable element's box
  (non-pointable ones are decoration, footers, backgrounds).
- ``validate`` (CLI) prints ``slide N: VALID`` or ``slide N: K errors`` plus the
  errors (never the word INVALID, so a grep for VALID stays honest); it uses
  the timings only when they match the script, and otherwise adds a
  ``note: timings missing/stale`` line (render would refuse the slide).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from src.simulate_to_video import _wav_duration_s
from tools.video import ROOT, VERSION, numbered, paths
from tools.video.cues import effective_until, resolve, validate
from tools.video.effects import Effect, draw, lens_box, state_at
from tools.video.geometry import Letterbox
from tools.video.script import sentences, words

FPS = 30
GAP_S = 0.6
OVERLAP = ("concurrency:", "zoom overlaps")
SHIM = Path.home() / ".venvs" / "ffmpeg-static" / "shim"
ENCODE = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20",
          "-threads", "1",                    # multi-threaded x264 is not byte-deterministic
          "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart"]


def use_static_ffmpeg() -> None:
    """Put the static ffmpeg first on PATH (the system one is broken here)."""
    if SHIM.is_dir() and str(SHIM) not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] = f"{SHIM}{os.pathsep}{os.environ['PATH']}"


def _sha(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


# ---------- one slide ----------

def playable(sheet, slide, sents, rows) -> tuple[list, list[str]]:
    """The sheet's valid cues and a log of the dropped ones (original indices)."""
    if [[w["word"] for w in r] for r in rows] != sents:
        raise ValueError("timings are stale (script words changed): rerun voice")
    cues = sheet.get("cues") if isinstance(sheet, dict) else None
    keep, log = list(range(len(cues) if isinstance(cues, list) else 0)), []
    while True:
        errs = validate({"cues": [cues[i] for i in keep] if cues is not None else None}, slide, sents, rows)
        if not errs:
            return [cues[i] for i in keep], log
        bad: dict[int, list[str]] = {}
        for e in errs:
            m = re.match(r"cue (\d+): (.*)", e)
            if not m:
                raise ValueError(e)
            bad.setdefault(keep[int(m[1])], []).append(m[2])
        drop = {i for i, ms in bad.items() if not all(m.startswith(OVERLAP) for m in ms)} or set(bad)
        log += [f"cue {i}: {m}" for i in sorted(drop) for m in bad[i]]
        keep = [i for i in keep if i not in drop]


def build_plan(cues, slide, sents, rows, lb: Letterbox) -> list[Effect]:
    """Validated cues -> ``Effect`` timeline in frame px and slide-audio seconds."""
    iv = effective_until({"cues": cues}, rows)
    occupied = [lb.box_px(e["bbox"]) for e in slide["elements"] if e.get("pointable") and e.get("bbox")]
    plan, held = [], None                     # held: (box, spotlight) of the current focus/glide
    for i in sorted(range(len(cues)), key=lambda i: (iv[i][0], i)):
        c, (t0, t1), kind = cues[i], iv[i], cues[i]["effect"]
        r = resolve(c, slide, sents)
        box = lb.box_px(r["bbox"])
        if kind == "focus":
            held = (box, bool(c.get("spotlight")))
            plan.append(Effect(kind, t0, t1, box, {"spotlight": held[1]}))
        elif kind == "glide":
            plan.append(Effect(kind, t0, t1, box, {"from": held[0], "spotlight": held[1]}))
            held = (box, held[1])
        elif kind == "underline":
            plan.append(Effect(kind, t0, t1, lb.box_px(r["line_bbox"], pad_px=0)))
        elif kind == "ring":
            plan.append(Effect(kind, t0, t1, box))
        else:                                 # zoom
            if "region" in c:
                (x, y, w, h), (rx, ry, rw, rh) = r["bbox"], c["region"]
                box = lb.box_px((x + rx * w, y + ry * h, rw * w, rh * h))
            plan.append(Effect(kind, t0, t1, box, {"lens": lens_box(box, lb.size, occupied + [box])}))
    return plan


def render_slide(png, slide, slide_size, sheet, sents, rows, wav, out_mp4,
                 size=(1920, 1080), fps=FPS) -> list[str]:
    """Render one slide MP4; returns the dropped-cue log. Raises ValueError if refused."""
    cues, dropped = playable(sheet, slide, sents, rows)
    img = Image.open(png)
    lb = Letterbox(slide_size["w"], slide_size["h"], img.width, img.height, *size)
    base = lb.place(img)
    plan = build_plan(cues, slide, sents, rows, lb)
    n = int(_wav_duration_s(Path(wav)) * fps) + 1
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-framerate", str(fps), "-i", "-",
           "-i", str(wav), *ENCODE, "-shortest", str(out_mp4)]
    last, buf, unique = None, b"", 0
    with tempfile.TemporaryFile() as errf:   # stderr to a file: a chatty ffmpeg can't block the pipe
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errf)
        try:
            for i in range(n):
                state = state_at(i / fps, plan)
                if state != last:
                    buf, last, unique = draw(base, state).tobytes(), state, unique + 1
                proc.stdin.write(buf)
            proc.stdin.close()
        except BrokenPipeError:
            pass                              # surfaced by the return code below
        if proc.wait():
            errf.seek(0)
            raise RuntimeError(f"ffmpeg failed:\n{errf.read().decode(errors='replace')[-2000:]}")
    print(f"  {n} frames ({unique} drawn), {len(plan)} effects -> {out_mp4}")
    return dropped


def concat(parts: list[Path], out: Path, gap: float = GAP_S) -> None:
    """``parts`` joined with ``gap`` s of held last frame + silence between them."""
    chains, pads = [], ""
    for k in range(len(parts)):
        if k < len(parts) - 1:
            chains.append(f"[{k}:v]tpad=stop_mode=clone:stop_duration={gap}[v{k}];"
                          f"[{k}:a]apad=pad_dur={gap}[a{k}]")
            pads += f"[v{k}][a{k}]"
        else:
            pads += f"[{k}:v][{k}:a]"
    graph = ";".join(chains + [f"{pads}concat=n={len(parts)}:v=1:a=1[v][a]"])
    inputs = [a for p in parts for a in ("-i", str(p))]
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", graph,
                        "-map", "[v]", "-map", "[a]", *ENCODE, str(out)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"ffmpeg concat failed:\n{r.stderr[-2000:]}")


# ---------- deck ----------

def _index(P):
    index = json.loads(P.index.read_text())
    return {s["slide_number"]: s for s in index["slides"]}, index["slide_size"]


def _pick(files, only):
    return [(n, f) for n, f in files if only is None or n == only]


def no_sheets(P, only) -> str:
    return f"no cue sheets in {P.cues}" + (f" for slide {only}" if only is not None else "")


def _load(P, n):
    sents = [words(s) for s in sentences(P.script / f"slide_{n:03d}.md")]
    tf = P.timings / f"slide_{n:03d}.json"
    return sents, json.loads(tf.read_text()) if tf.exists() else None


def check(deck: str, only: int | None = None, root: Path = ROOT) -> int:
    """The ``validate`` step: 0 if every cue sheet is valid, else 1."""
    P = paths(deck, root)
    slides, _ = _index(P)
    files, bad = _pick(numbered(P.cues, "json"), only), 0
    if not files:
        print(no_sheets(P, only))
        return 1
    for n, f in files:
        if n not in slides or not (P.script / f"slide_{n:03d}.md").exists():
            print(f"slide {n}: 1 errors\n  sheet: slide {n} has no " + ("index entry" if n not in slides else "script"))
            bad += 1
            continue
        sents, tim = _load(P, n)
        rows = tim["sentences"] if tim and [[w["word"] for w in r] for r in tim["sentences"]] == sents else None
        errs = validate(json.loads(f.read_text()), slides[n], sents, rows)
        print(f"slide {n}: " + (f"{len(errs)} errors" if errs else "VALID"))
        for e in errs:
            print(f"  {e}")
        if rows is None:
            print(f"  note: timings {'missing' if tim is None else 'stale'}, run voice before render")
        bad += bool(errs)
    return 1 if bad else 0


def render(deck: str, only: int | None = None, size=(1920, 1080), root: Path = ROOT) -> int:
    """The ``render`` step: every scripted slide, then deck.mp4 if none was refused. 1 if any refused."""
    P = paths(deck, root)
    slides, slide_size = _index(P)
    scripted, sheets = _pick(numbered(P.script, "md"), only), dict(_pick(numbered(P.cues, "json"), only))
    if not scripted:
        print(f"no scripts in {P.script}" + (f" for slide {only}" if only is not None else ""))
        return 1
    if not sheets:
        print(no_sheets(P, only))
        return 1
    P.out.mkdir(parents=True, exist_ok=True)
    parts, keys, refused = [], [], 0
    for n, _ in scripted:
        sents, tim = _load(P, n)
        if not sents:
            print(f"slide {n}: empty script, not in the video")
            continue
        name = f"slide_{n:03d}"
        wav = P.voice / f"{name}.wav"
        why = ("no cue sheet, run direct" if n not in sheets else "not in the index" if n not in slides
               else "no timings, run voice first" if tim is None or not wav.exists() else None)
        if why:
            print(f"slide {n}: {why}")
            refused += 1
            continue
        sheet = json.loads(sheets[n].read_text())
        png = P.slides / f"{name}.png"
        mp4, hf, lf = P.out / f"{name}.mp4", P.out / f"{name}.hash", P.out / f"{name}.dropped.txt"
        key = _sha(json.dumps([sheet, sents, tim, slides[n], VERSION, list(size), FPS], sort_keys=True).encode()
                   + png.read_bytes())
        if mp4.exists() and hf.exists() and hf.read_text() == key:
            print(f"slide {n}: cached {mp4}")
            dropped = lf.read_text().splitlines() if lf.exists() else []
        else:
            print(f"slide {n}: rendering")
            try:
                dropped = render_slide(png, slides[n], slide_size, sheet, sents, tim["sentences"], wav, mp4, size)
            except ValueError as e:
                print(f"  refused: {e}")
                refused += 1
                continue
            lf.write_text("".join(f"{e}\n" for e in dropped))
            hf.write_text(key)                # last: a hash only ever sits next to a finished mp4 + log
        for e in dropped:
            print(f"  dropped {e}")
        parts.append(mp4)
        keys.append(key)
    if only is None and refused:
        print(f"deck: skipped ({refused} slides refused)")
    elif only is None and parts:
        deck_mp4, dh = P.out / "deck.mp4", P.out / "deck.hash"
        key = _sha(json.dumps([keys, GAP_S]).encode())
        if deck_mp4.exists() and dh.exists() and dh.read_text() == key:
            print(f"deck: cached {deck_mp4}")
        else:
            concat(parts, deck_mp4)
            dh.write_text(key)
            print(f"deck: {len(parts)} slides -> {deck_mp4}")
    return 1 if refused else 0
