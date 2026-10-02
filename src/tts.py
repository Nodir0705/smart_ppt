"""Offline TTS via Piper. Synthesizes the narration script into per-slide WAVs.

Used by `src/simulate.py` to drive the end-to-end demo. The simulator
"presents" by playing each slide's WAV out the speakers while feeding the
same audio bytes into the STT pipeline.

Output is 16 kHz mono PCM — exactly what VoskEngine wants — so the simulator
doesn't need any resampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import wave
from pathlib import Path

from .pcm import resample, to_mono


TARGET_RATE = 16000  # matches Vosk small-en-us
TARGET_CHANNELS = 1


def _resample(pcm: bytes, src_rate: int, src_channels: int) -> bytes:
    if src_channels != TARGET_CHANNELS:
        pcm = to_mono(pcm, 2) if src_channels == 2 else pcm
    if src_rate != TARGET_RATE:
        pcm = resample(pcm, src_rate, TARGET_RATE, channels=TARGET_CHANNELS)
    return pcm


def _write_wav(wav_path: Path, pcm: bytes) -> None:
    # write-then-rename: an interrupted write never leaves a truncated WAV that
    # synthesize_slides' exists() cache check would trust
    tmp = Path(f"{wav_path}.tmp")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(TARGET_CHANNELS)
        w.setsampwidth(2)
        w.setframerate(TARGET_RATE)
        w.writeframes(pcm)
    os.replace(tmp, wav_path)


class PiperTTS:
    def __init__(self, model_path: str | Path) -> None:
        from piper import PiperVoice
        self.voice = PiperVoice.load(str(model_path))

    def _pcm16k(self, text: str) -> bytes:
        """Synthesize `text` to 16 kHz mono 16-bit PCM bytes."""
        # piper yields its native sample rate; collect, then resample to 16 kHz
        chunks: list[bytes] = []
        native_rate = None
        native_channels = None
        for chunk in self.voice.synthesize(text):
            # piper >=1.4 yields AudioChunk objects with .audio_int16_bytes
            if hasattr(chunk, "audio_int16_bytes"):
                pcm = chunk.audio_int16_bytes
                native_rate = chunk.sample_rate
                native_channels = chunk.sample_channels
            else:
                pcm = bytes(chunk)
                native_rate = native_rate or self.voice.config.sample_rate
                native_channels = native_channels or 1
            chunks.append(pcm)
        return _resample(b"".join(chunks), native_rate or TARGET_RATE, native_channels or 1)

    def synthesize(self, text: str, wav_path: Path) -> None:
        """Synthesize `text` and write a 16 kHz mono 16-bit WAV to `wav_path`."""
        _write_wav(wav_path, self._pcm16k(text))

    def synthesize_segments(self, segments: list[str], wav_path: Path,
                            gap_ms: int) -> list[float]:
        """One WAV of `segments`, each followed by `gap_ms` of silence; returns
        each segment's start time in seconds.

        Piper adds no reliable pause between sentences; an explicit gap keeps
        consecutive narrated elements out of the same STT tail window."""
        gap = b"\x00\x00" * int(TARGET_RATE * gap_ms / 1000)
        parts = [self._pcm16k(s) + gap for s in segments]
        starts, at = [], 0
        for part in parts:
            starts.append(round(at / 2 / TARGET_RATE, 4))
            at += len(part)
        _write_wav(wav_path, b"".join(parts))
        return starts


def synthesize_slides(slides: list[dict], voice_path: str | Path, out_dir: Path,
                      gap_ms: int, tts: PiperTTS | None = None,
                      need_starts: bool = False) -> list[dict]:
    """Per-slide WAVs from `[{"slide", "segments"}]`; returns a manifest.

    WAV names are keyed on a hash of (segments, gap, voice), so a changed
    narration rule can never silently replay stale audio. Segment start times
    (s) go in a `<wav>.json` sidecar and the entry's `starts`; a cached WAV from
    before sidecars has no `starts`, and is re-synthesized only if `need_starts`
    (so other evals' audio caches stay byte-identical)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for entry in slides:
        key = hashlib.sha1(json.dumps(
            [entry["segments"], gap_ms, str(voice_path)]).encode()).hexdigest()[:10]
        wav = out_dir / f"slide_{entry['slide']:03d}_{key}.wav"
        side = Path(f"{wav}.json")
        if not wav.exists() or (need_starts and not side.exists()):
            tts = tts or PiperTTS(voice_path)
            starts = tts.synthesize_segments(entry["segments"], wav, gap_ms)
            side.write_text(json.dumps({"starts": starts}))
        item = {"slide": entry["slide"], "text": " ".join(entry["segments"]), "wav": str(wav)}
        if side.exists():
            item["starts"] = json.loads(side.read_text())["starts"]
        manifest.append(item)
    return manifest


def synthesize_script(script_path: Path, out_dir: Path) -> list[dict]:
    script = json.loads(script_path.read_text())
    voice_path = Path(script["voice"])
    if not voice_path.is_absolute():
        voice_path = script_path.parent.parent / voice_path
    tts = PiperTTS(voice_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for entry in script["slides"]:
        sno = entry["slide"]
        wav_path = out_dir / f"slide_{sno:03d}.wav"
        if wav_path.exists():
            print(f"  cache hit: {wav_path.name}")
        else:
            print(f"  synthesizing slide {sno}…")
            tts.synthesize(entry["text"], wav_path)
        manifest.append({
            "slide": sno,
            "text": entry["text"],
            "wav": str(wav_path),
        })
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Synthesize narration to per-slide WAVs.")
    ap.add_argument("--script", default="scripts/narration.json")
    ap.add_argument("--out", default="out/narration")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    script_path = Path(args.script)
    out_dir = Path(args.out)
    if args.force:
        for p in out_dir.glob("slide_*.wav"):
            p.unlink()
    manifest = synthesize_script(script_path, out_dir)
    # write a manifest JSON so the simulator doesn't need to re-parse the script
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest)} narration WAVs → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
