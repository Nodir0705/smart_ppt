"""Contact sheets: ``python3 -m tools.video sheets <deck> [--only N] [--size WxH]`` ->
``out/video/<deck>/sheets/slide_NNN_cK.png``, one per played cue (K = its index
in the cue sheet), for judging effects without watching the video.

Each sheet is a 2x2 grid of half-size frames: the cue's start + 0.1 s, + 0.5 s,
+ 1.0 s and its last frame, with the cue summary and its anchor sentence
printed underneath. Frames are the renderer's own (``draw(base, state_at(i /
FPS, plan))`` from ``render.build_plan``), on its frame grid, so every effect
live at that moment is in them, not only cue K. Offsets past the cue's end
clamp to its last frame; a cue that holds to the end of the slide ends on the
slide's last frame (wav length, as in ``render``). Dropped cues get no sheet
(printed with the reason); a slide's old sheets are deleted before its new
ones are written. Uses the voice wav and timings, never ffmpeg.
"""
from __future__ import annotations

import json
import math
import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.simulate_to_video import _wav_duration_s
from tools.video import ROOT, numbered, paths
from tools.video.cues import effective_until
from tools.video.effects import draw, state_at
from tools.video.geometry import Letterbox
from tools.video.render import FPS, _index, _load, _pick, build_plan, no_sheets, playable

OFFSETS = (0.1, 0.5, 1.0)
INK, PAPER = (230, 236, 242), (16, 20, 26)


def frames_for(t0: float, t1: float, n: int, fps: int = FPS) -> list[int]:
    """Frame indices at t0 + OFFSETS and the cue's last frame (``n`` = frames in the slide)."""
    end = n - 1 if math.isinf(t1) else min(n - 1, math.ceil(t1 * fps - 1e-6) - 1)
    first = math.ceil(t0 * fps - 1e-6)
    return [min(end, max(first, round((t0 + d) * fps))) for d in OFFSETS] + [end]


def _summary(k, c, sents, t0, t1) -> str:
    at = c["at"]
    extra = "".join(f" {key}={json.dumps(c[key])}" for key in ("until", "line", "spotlight", "region") if key in c)
    end = "end of slide" if math.isinf(t1) else f"{t1:.2f} s"
    return (f"c{k} {c['effect']} {c['target']} at s{at['s']}w{at['w']} "
            f"'{sents[at['s']][at['w']]}' {t0:.2f} s -> {end}{extra}")


def _sheet(frames, labels, lines, size) -> Image.Image:
    W, H = size
    cw, ch = W // 2, H // 2
    fs = max(12, H // 45)
    font = ImageFont.load_default(size=fs)
    wrapped = [w for ln in lines for w in textwrap.wrap(ln, max(20, int(W / (fs * 0.55))))]
    img = Image.new("RGB", (cw * 2, ch * 2 + (len(wrapped) + 1) * (fs + 6)), PAPER)
    d = ImageDraw.Draw(img)
    for i, (f, label) in enumerate(zip(frames, labels)):
        x, y = (i % 2) * cw, (i // 2) * ch
        img.paste(f.resize((cw, ch), Image.LANCZOS), (x, y))
        d.rectangle([x, y, x + len(label) * fs * 0.6 + 8, y + fs + 8], fill=PAPER)
        d.text((x + 4, y + 4), label, fill=INK, font=font)
    for j, ln in enumerate(wrapped):
        d.text((8, ch * 2 + 6 + j * (fs + 6)), ln, fill=INK, font=font)
    return img


def sheets(deck: str, only: int | None = None, size=(1920, 1080), root: Path = ROOT) -> int:
    """Write the contact sheets; 1 if any slide had to be skipped."""
    P = paths(deck, root)
    slides, slide_size = _index(P)
    files, bad = _pick(numbered(P.cues, "json"), only), 0
    if not files:
        print(no_sheets(P, only))
        return 1
    out = P.out / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    for n, f in files:
        name = f"slide_{n:03d}"
        sents, tim = _load(P, n) if (P.script / f"{name}.md").exists() else ([], None)
        wav, png = P.voice / f"{name}.wav", P.slides / f"{name}.png"
        why = ("no script" if not sents else "not in the index" if n not in slides
               else "no timings, run voice first" if tim is None or not wav.exists() else None)
        if why:
            print(f"slide {n}: {why}")
            bad += 1
            continue
        sheet, rows = json.loads(f.read_text()), tim["sentences"]
        try:
            cues, dropped = playable(sheet, slides[n], sents, rows, slide_size["w"])
        except ValueError as e:
            print(f"slide {n}: {e}")
            bad += 1
            continue
        for e in dropped:
            print(f"slide {n}: dropped {e} (no sheet)")
        gone = {int(re.match(r"cue (\d+)", e)[1]) for e in dropped}
        kept = [k for k in range(len(sheet["cues"])) if k not in gone]
        img = Image.open(png)
        lb = Letterbox(slide_size["w"], slide_size["h"], img.width, img.height, *size)
        base = lb.place(img)
        plan = build_plan(cues, slides[n], sents, rows, lb)
        iv = effective_until({"cues": cues}, rows)
        total = int(_wav_duration_s(wav) * FPS) + 1
        for old in out.glob(f"{name}_c*.png"):
            old.unlink()
        for k, c, (t0, t1) in zip(kept, cues, iv):
            idx = frames_for(t0, t1, total)
            frames = [draw(base, state_at(i / FPS, plan)) for i in idx]
            labels = [f"{tag}  t={i / FPS:.2f} s" for tag, i in zip(("+0.1 s", "+0.5 s", "+1.0 s", "end"), idx)]
            text = [f"slide {n} · {_summary(k, c, sents, t0, t1)}", f"s{c['at']['s']}: {' '.join(sents[c['at']['s']])}"]
            dest = out / f"{name}_c{k}.png"
            _sheet(frames, labels, text, size).save(dest)
        print(f"slide {n}: {len(kept)} sheets -> {out}/{name}_c*.png")
    return 1 if bad else 0
