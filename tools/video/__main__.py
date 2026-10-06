"""CLI: ``python3 -m tools.video {packets-draft,voice,packets-direct,validate,render,sheets,all} <deck>
[--only N] [--size WxH] [--voice NAME]``.

``packets-draft`` / ``packets-direct`` write the agent packets and end with a
``todo:`` line (see ``packets``); the draft and direct steps themselves are the
Claude Code workflows ``.claude/workflows/video-draft.js`` / ``video-direct.js``,
which the CLI cannot run, so ``all`` = voice + render (render validates each
sheet first). ``sheets`` writes per-cue contact sheets (see ``sheets``).
ffmpeg comes from PATH, with ``~/.venvs/ffmpeg-static/shim`` prepended when
it exists. ``--root`` (hidden, default the repo) lets tests run
every step on a temporary tree.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tools.video import ROOT, packets, render, sheets, voice


def _size(s: str) -> tuple[int, int]:
    w, h = (int(v) for v in s.lower().split("x"))
    return w, h


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m tools.video")
    ap.add_argument("step", choices=["packets-draft", "voice", "packets-direct", "validate", "render", "sheets", "all"])
    ap.add_argument("deck")
    ap.add_argument("--only", type=int, help="this slide number only")
    ap.add_argument("--voice", default=voice.DEFAULT_VOICE,
                    help=f"voice step: one of {sorted(voice.VOICES)} or piper:<model.onnx> | kokoro:<voice id>"
                         f" (default {voice.DEFAULT_VOICE})")
    ap.add_argument("--size", type=_size, default=(1920, 1080), help="WxH, default 1920x1080")
    ap.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    render.use_static_ffmpeg()
    if a.step == "packets-draft":
        packets.draft(a.deck, a.only, a.root)
    if a.step == "packets-direct":
        packets.direct(a.deck, a.only, a.root)
    if a.step in ("voice", "all"):
        voice.voice(a.deck, a.only, a.root, name=a.voice)
    if a.step == "validate":
        return render.check(a.deck, a.only, a.root)
    if a.step == "sheets":
        return sheets.sheets(a.deck, a.only, a.size, a.root)
    if a.step in ("render", "all"):
        return render.render(a.deck, a.only, a.size, a.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
