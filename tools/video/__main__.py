"""CLI: ``python3 -m tools.video {voice,validate,render,all} <deck> [--only N] [--size WxH]``.

``all`` = voice + render (render validates each sheet first); the draft and
direct steps are Claude Code workflows that run between them. ffmpeg comes from
PATH, with ``~/.venvs/ffmpeg-static/shim`` prepended when it exists.
"""
from __future__ import annotations

import argparse

from tools.video import render, voice


def _size(s: str) -> tuple[int, int]:
    w, h = (int(v) for v in s.lower().split("x"))
    return w, h


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m tools.video")
    ap.add_argument("step", choices=["voice", "validate", "render", "all"])
    ap.add_argument("deck")
    ap.add_argument("--only", type=int, help="this slide number only")
    ap.add_argument("--size", type=_size, default=(1920, 1080), help="WxH, default 1920x1080")
    a = ap.parse_args(argv)
    render.use_static_ffmpeg()
    if a.step in ("voice", "all"):
        voice.voice(a.deck, a.only)
    if a.step == "validate":
        return render.check(a.deck, a.only)
    if a.step in ("render", "all"):
        return render.render(a.deck, a.only, a.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
