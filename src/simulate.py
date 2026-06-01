"""End-to-end simulation: TTS narrates each slide; the viewer plays the audio
out the speakers AND feeds it into STT; the highlight tracks the spoken word.

Usage:
    python3 -m src.simulate                              # use defaults
    python3 -m src.simulate --script scripts/narration.json
    python3 -m src.simulate --pause 1.0                  # pause between slides
    python3 -m src.simulate --mute                       # STT only, no speakers

Prereqs (created on demand):
    out/index.json        — `python3 -m src.index_deck test_deck.pptx`
    out/slides/*.png      — `python3 -m src.render     test_deck.pptx`
    out/narration/*.wav   — `python3 -m src.tts`        (auto-runs if missing)
    out/voices/en_US-amy-low.onnx — Piper voice (manual download — see README)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .matcher import PolicyConfig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Smart Pointer — end-to-end TTS simulation")
    ap.add_argument("--index", default="out/index.json")
    ap.add_argument("--slides", default="out/slides")
    ap.add_argument("--script", default="scripts/narration.json")
    ap.add_argument("--narration-dir", default="out/narration")
    ap.add_argument("--pause", type=float, default=0.6, help="seconds between slides")
    ap.add_argument("--mute", action="store_true", help="don't play to speakers (STT only)")
    ap.add_argument("--threshold", type=float, default=PolicyConfig.threshold)
    ap.add_argument("--gap", type=float, default=PolicyConfig.gap)
    ap.add_argument("--stick", type=int, default=PolicyConfig.stickiness)
    args = ap.parse_args(argv)

    index_path = Path(args.index)
    slides_dir = Path(args.slides)
    script_path = Path(args.script)
    narration_dir = Path(args.narration_dir)

    if not index_path.exists():
        print(f"index not found: {index_path}  (run `python3 -m src.index_deck test_deck.pptx`)")
        return 2
    if not slides_dir.exists():
        print(f"slides not found: {slides_dir}  (run `python3 -m src.render test_deck.pptx`)")
        return 2

    # build narration WAVs if missing
    manifest_path = narration_dir / "manifest.json"
    if not manifest_path.exists():
        print("narration not found; synthesizing now…")
        from .tts import synthesize_script
        manifest = synthesize_script(script_path, narration_dir)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    manifest = json.loads(manifest_path.read_text())

    # if the user asked to mute the speakers, swap the WAV paths through
    # PlaybackSource(mute=True) by transforming the manifest into objects
    # that the viewer can identify — but ViewerApp.start_playlist just takes
    # dicts, so we pass an extra flag via env. Simpler: build items here.
    items = [{"slide": e["slide"], "wav": e["wav"]} for e in manifest]
    if args.mute:
        # patch PlaybackSource at construction time so the simulator can opt out
        # of audible playback without spreading flags through ViewerApp.
        from . import audio as _audio
        _orig = _audio.PlaybackSource
        class _MutedPlaybackSource(_orig):  # type: ignore[misc]
            def __init__(self, *a, **kw):
                kw.setdefault("mute", True)
                super().__init__(*a, **kw)
        _audio.PlaybackSource = _MutedPlaybackSource

    from .viewer import ViewerApp
    app = ViewerApp(
        index_path=index_path,
        slides_dir=slides_dir,
        threshold=args.threshold, gap=args.gap, stickiness=args.stick,
        start_slide=items[0]["slide"] if items else 1,
    )
    app.start_playlist(items, inter_slide_pause_s=args.pause)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
