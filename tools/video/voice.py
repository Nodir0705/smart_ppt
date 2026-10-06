"""Voice step: ``script/<deck>/slide_NNN.md`` -> ``out/video/<deck>/voice/slide_NNN.wav``
+ ``timings/slide_NNN.json`` (``{"sentences": [[{word, start, end}]], "wav", "duration",
"coverage"}``, seconds into the slide wav).

Per slide: the voice (``--voice``: a ``VOICES`` name or a raw ``piper:<model.onnx>`` |
``kokoro:<voice id>`` spec) speaks each sentence, joined with a 350 ms gap into a wav at
the voice's own rate (Kokoro 24 kHz, Piper 16-22.05 kHz; render resamples to 48 kHz).
Each sentence's own audio, resampled to 16 kHz (``src.pcm.resample_poly``), goes to the
Whistle server with ``"words": true``; ``timings.align`` maps Whistle's words onto the
script words and the sentence start is added.

Decisions where the plan left details open:
- Whistle is started by constructing ``src.stt.WhistleEngine`` (its venv/weights
  checks and subprocess launch); the word request is written to ``engine.proc``
  and read back with ``engine._read`` here, so ``src/stt.py`` stays unchanged.
- ``coverage`` (stored per sentence) = script words ``timings.match`` matched /
  non-punctuation words; sentences under 85 % are printed with ``FLAG``.
- Cache key = sha1(sentences, voice spec, VERSION), stored in
  ``timings/slide_NNN.hash``; the slide is skipped when the key, the json and the
  wav all exist. Both hash and json are deleted before re-voicing, so a failed run
  never leaves them describing a different wav. The voice and Whistle start
  lazily, so a fully cached run loads neither. A script with no sentences is skipped.
- Every backend is deterministic: the same sentence list gives the same samples
  (``deterministic`` checks it), so a re-voice reproduces the cached wav.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import wave
from pathlib import Path

from src.pcm import resample_poly
from src.stt import WhistleEngine
from src.tts import TARGET_RATE
from tools.video import ROOT, VERSION, numbered, paths
from tools.video.script import norm, sentences, words
from tools.video.timings import align, match

VOICES = {"amy-low": "piper:out/voices/en_US-amy-low.onnx",   # Piper, 16 kHz, the v1 voice
          "kokoro-heart": "kokoro:af_heart"}                    # Kokoro-82M, 24 kHz
DEFAULT_VOICE = "kokoro-heart"
KOKORO_PYTHON = Path.home() / ".venvs/smartppt-kokoro/bin/python"
KOKORO_MODEL = ROOT / "out/voices/kokoro-v1.0.onnx"
KOKORO_VOICES = ROOT / "out/voices/voices-v1.0.bin"
GAP_MS = 350
MIN_COVERAGE = 0.85


class Piper:
    """A Piper voice (``rate`` = its config's). Piper's VITS graph draws its noise from
    RandomNormalLike, whose generator is seeded when the session is built and advances on
    every run, so the same text came out different on every call (amy-low: 3 runs, 3
    lengths). ``synth`` builds a fresh session after ``onnxruntime.set_seed(0)``, so the
    same sentence list gives the same samples.
    ponytail: one ~0.5 s session per slide, so a sentence's audio depends on the sentences
    before it on its slide; a session per sentence if that ever matters."""

    def __init__(self, model: Path) -> None:
        from piper import PiperVoice
        self.model = str(model)
        self.v = PiperVoice.load(self.model)
        self.rate = self.v.config.sample_rate

    def synth(self, sents: list[str]) -> list[bytes]:
        import onnxruntime as ort
        ort.set_seed(0)
        self.v.session = ort.InferenceSession(self.model, providers=["CPUExecutionProvider"])
        return [b"".join(c.audio_int16_bytes for c in self.v.synthesize(s)) for s in sents]


class Kokoro:
    """Kokoro-82M (Apache-2.0) voice ``vid``, run by ``kokoro_tts.py`` in its own venv,
    one process per slide (~1.5 s to start). Setup: ``python3 -m venv ~/.venvs/smartppt-kokoro
    && ~/.venvs/smartppt-kokoro/bin/pip install kokoro-onnx``; ``kokoro-v1.0.onnx`` and
    ``voices-v1.0.bin`` from github.com/thewh1teagle/kokoro-onnx/releases (model-files-v1.0)
    into ``out/voices/``."""
    rate = 24000

    def __init__(self, vid: str, python: Path = KOKORO_PYTHON, model: Path = KOKORO_MODEL,
                 voices: Path = KOKORO_VOICES) -> None:
        for what, path in (("venv python", python), ("model", model), ("voices", voices)):
            if not Path(path).exists():
                raise RuntimeError(f"Kokoro: {what} not found at {path} (see the Kokoro docstring)")
        self.cmd = [str(python), str(Path(__file__).with_name("kokoro_tts.py")), str(model), str(voices), vid]

    def synth(self, sents: list[str]) -> list[bytes]:
        out = subprocess.run(self.cmd, input=json.dumps(sents).encode(), capture_output=True)
        if out.returncode:
            raise RuntimeError(f"kokoro failed (exit {out.returncode}): {out.stderr.decode()[-800:]}")
        head, _, body = out.stdout.partition(b"\n")
        parts, at = [], 0
        for n in json.loads(head):
            parts.append(body[at:at + n])
            at += n
        return parts


def resolve(name: str) -> str:
    """A ``VOICES`` name or a raw spec -> the spec (``piper:<model.onnx>`` | ``kokoro:<voice id>``)."""
    spec = VOICES.get(name, name)
    kind, _, arg = spec.partition(":")
    if kind not in ("piper", "kokoro") or not arg:
        raise ValueError(f"unknown voice {name!r}: use one of {sorted(VOICES)} "
                         "or piper:<model.onnx> | kokoro:<voice id>")
    return spec


def load_tts(name: str):
    kind, _, arg = resolve(name).partition(":")
    return Piper(ROOT / arg) if kind == "piper" else Kokoro(arg)


def deterministic(tts, sents: list[str]) -> bool:
    """Same sentences twice -> identical samples?"""
    return tts.synth(sents) == tts.synth(sents)


def whistle_words(engine: WhistleEngine, pcm: bytes) -> list[dict]:
    """Whistle word timestamps (seconds into ``pcm``, 16 kHz mono PCM16)."""
    engine.proc.stdin.write(json.dumps({"n": len(pcm), "words": True}).encode() + b"\n" + pcm)
    engine.proc.stdin.flush()
    reply = engine._read()
    if not reply.get("ok"):
        raise RuntimeError(f"whistle server error: {reply.get('error')}")
    return reply["words"]


def voice_slide(sents: list[str], wav: Path, tts, asr) -> dict:
    parts, rate = tts.synth(sents), tts.rate
    gap = b"\x00\x00" * int(rate * GAP_MS / 1000)
    starts, at = [], 0
    for p in parts:
        starts.append(at // 2)
        at += len(p) + len(gap)
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(p + gap for p in parts))
    rows, coverage = [], []
    for sent, p, st in zip(sents, parts, starts):
        heard = asr(resample_poly(p, rate, TARGET_RATE))
        ws, t0 = words(sent), st / rate
        rows.append([{"word": a["word"], "start": round(a["start"] + t0, 3), "end": round(a["end"] + t0, 3)}
                     for a in align(ws, heard)])
        got, said = match(ws, heard), [k for k, w in enumerate(ws) if norm(w)]
        coverage.append(round(sum(got[k] is not None for k in said) / max(len(said), 1), 3))
    return {"sentences": rows, "wav": str(wav), "duration": at / 2 / rate, "coverage": coverage}


def _report(n: int, coverage: list[float], cached: bool) -> None:
    low = [f"s{i} {c:.0%}" for i, c in enumerate(coverage) if c < MIN_COVERAGE]
    print(f"slide {n}: {'cached, ' if cached else ''}words matched "
          + " ".join(f"{c:.0%}" for c in coverage)
          + (f"  FLAG under {MIN_COVERAGE:.0%}: {', '.join(low)}" if low else ""))


def voice(deck: str, only: int | None = None, root: Path = ROOT, tts=None, asr=None,
          name: str = DEFAULT_VOICE) -> list[int]:
    """Voice every scripted slide (or ``only``) with voice ``name``; returns the slides (re)voiced."""
    spec = resolve(name)
    P = paths(deck, root)
    P.voice.mkdir(parents=True, exist_ok=True)
    P.timings.mkdir(parents=True, exist_ok=True)
    engine, done = None, []

    def whistle(pcm):
        nonlocal engine
        engine = engine or WhistleEngine()
        return whistle_words(engine, pcm)

    try:
        for n, md in numbered(P.script, "md"):
            sents = sentences(md)
            if (only is not None and n != only) or not sents:
                continue
            name = f"slide_{n:03d}"
            tj, hf, wav = P.timings / f"{name}.json", P.timings / f"{name}.hash", P.voice / f"{name}.wav"
            key = hashlib.sha1(json.dumps([sents, spec, VERSION]).encode()).hexdigest()
            cached = hf.exists() and hf.read_text() == key and tj.exists() and wav.exists()
            if cached:
                data = json.loads(tj.read_text())
            else:
                tts = tts or load_tts(spec)
                for f in (hf, tj):             # the wav is overwritten first: if voicing fails, render
                    f.unlink(missing_ok=True)  # refuses (no timings) and the next run re-voices
                data = voice_slide(sents, wav, tts, asr or whistle)
                tj.write_text(json.dumps(data, indent=1))
                hf.write_text(key)
                done.append(n)
            _report(n, data["coverage"], cached)
    finally:
        if engine is not None:
            engine.close()
    return done
