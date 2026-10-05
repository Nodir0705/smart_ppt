"""Jarvis video director: script -> voice + word timings -> cue sheet -> HUD render.
See docs/superpowers/specs/2026-10-06-jarvis-video-director-design.md."""
import re
from pathlib import Path
from types import SimpleNamespace

VERSION = 1
ROOT = Path(__file__).resolve().parents[2]


def paths(deck: str, root: Path = ROOT) -> SimpleNamespace:
    """Every per-deck input and output location (``root`` = the repo; tests pass a tmp dir)."""
    root = Path(root)
    bench, out = root / "out" / "bench" / deck, root / "out" / "video" / deck
    return SimpleNamespace(script=root / "script" / deck, index=bench / "map" / "index.scoped.json",
                           slides=bench / "default" / "slides", out=out, voice=out / "voice",
                           timings=out / "timings", cues=out / "cues", packets=out / "packets")


def numbered(d: Path, ext: str) -> list[tuple[int, Path]]:
    """``slide_NNN.<ext>`` files in ``d`` as sorted ``(N, path)``; other names
    (a leftover ``slide_004_old.md``, an unpadded ``slide_4.md``) are ignored,
    since every step rebuilds the name as ``slide_{n:03d}``."""
    return sorted((int(m[1]), f) for f in Path(d).glob(f"slide_*.{ext}")
                  if (m := re.fullmatch(rf"slide_(\d{{3}})\.{ext}", f.name)))
