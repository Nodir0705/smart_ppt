"""Offline TTS via Piper. Synthesizes the narration script into per-slide WAVs.

Used by `src/simulate.py` to drive the end-to-end demo. The simulator
"presents" by playing each slide's WAV out the speakers while feeding the
same audio bytes into the STT pipeline.

Output is 16 kHz mono PCM — exactly what VoskEngine wants — so the simulator
doesn't need any resampling.
"""

from __future__ import annotations

import argparse
import audioop
import json
import wave
from pathlib import Path


TARGET_RATE = 16000  # matches Vosk small-en-us
TARGET_CHANNELS = 1


def _resample(pcm: bytes, src_rate: int, src_channels: int) -> bytes:
    if src_channels != TARGET_CHANNELS:
        pcm = audioop.tomono(pcm, 2, 0.5, 0.5) if src_channels == 2 else pcm
    if src_rate != TARGET_RATE:
        pcm, _ = audioop.ratecv(pcm, 2, TARGET_CHANNELS, src_rate, TARGET_RATE, None)
    return pcm


class PiperTTS:
    def __init__(self, model_path: str | Path) -> None:
        from piper import PiperVoice
        self.voice = PiperVoice.load(str(model_path))

    def synthesize(self, text: str, wav_path: Path) -> None:
        """Synthesize `text` and write a 16 kHz mono 16-bit WAV to `wav_path`."""
        # piper writes its native sample rate; we re-open + resample to 16 kHz
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(TARGET_CHANNELS)
            w.setsampwidth(2)
            w.setframerate(TARGET_RATE)
            # collect all bytes first, resample, then write
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
            full = b"".join(chunks)
            resampled = _resample(full, native_rate or TARGET_RATE, native_channels or 1)
            w.writeframes(resampled)


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
