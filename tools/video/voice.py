"""Voice step: ``script/<deck>/slide_NNN.md`` -> ``out/video/<deck>/voice/slide_NNN.wav``
+ ``timings/slide_NNN.json`` (``{"sentences": [[{word, start, end}]], "wav", "duration",
"coverage"}``, seconds into the slide wav).

Per slide: Piper speaks each sentence (``PiperTTS.synthesize_segments``, 350 ms
gap); each sentence's own span (its start to the next start minus the gap) goes
to the Whistle server with ``"words": true``; ``timings.align`` maps Whistle's
words onto the script words and the sentence start is added.

Decisions where the plan left details open:
- Whistle is started by constructing ``src.stt.WhistleEngine`` (its venv/weights
  checks and subprocess launch); the word request is written to ``engine.proc``
  and read back with ``engine._read`` here, so ``src/stt.py`` stays unchanged.
- Offsets use the cut's sample position (where the audio Whistle heard starts),
  not the 4-decimal rounded start; the cut itself comes from that start, so it
  can sit 1 sample (62.5 us) off the true sentence start.
- ``coverage`` (stored per sentence) = script words ``timings.match`` matched /
  non-punctuation words; sentences under 85 % are printed with ``FLAG``.
- Cache key = sha1(sentences, voice model path relative to the repo, VERSION),
  stored in ``timings/slide_NNN.hash``; the slide is skipped when the key, the
  json and the wav all exist. Both hash and json are deleted before re-voicing,
  so a failed run never leaves them describing a different wav. Piper and
  Whistle start lazily, so a fully cached run loads neither. A script with no
  sentences is skipped.
"""
from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

from src.stt import WhistleEngine
from src.tts import TARGET_RATE, PiperTTS
from tools.video import ROOT, VERSION, numbered, paths
from tools.video.script import norm, sentences, words
from tools.video.timings import align, match

VOICE_REL = "out/voices/en_US-amy-low.onnx"
GAP_MS = 350
MIN_COVERAGE = 0.85


def whistle_words(engine: WhistleEngine, pcm: bytes) -> list[dict]:
    """Whistle word timestamps (seconds into ``pcm``, 16 kHz mono PCM16)."""
    engine.proc.stdin.write(json.dumps({"n": len(pcm), "words": True}).encode() + b"\n" + pcm)
    engine.proc.stdin.flush()
    reply = engine._read()
    if not reply.get("ok"):
        raise RuntimeError(f"whistle server error: {reply.get('error')}")
    return reply["words"]


def voice_slide(sents: list[str], wav: Path, tts, asr) -> dict:
    starts = tts.synthesize_segments(sents, wav, gap_ms=GAP_MS)
    with wave.open(str(wav), "rb") as w:
        pcm = w.readframes(w.getnframes())
    total, gap = len(pcm) // 2, int(TARGET_RATE * GAP_MS / 1000)
    cuts = [round(s * TARGET_RATE) for s in starts] + [total]
    rows, coverage = [], []
    for i, sent in enumerate(sents):
        heard = asr(pcm[2 * cuts[i]:2 * max(cuts[i], cuts[i + 1] - gap)])
        ws, t0 = words(sent), cuts[i] / TARGET_RATE
        rows.append([{"word": a["word"], "start": round(a["start"] + t0, 3), "end": round(a["end"] + t0, 3)}
                     for a in align(ws, heard)])
        got, said = match(ws, heard), [k for k, w in enumerate(ws) if norm(w)]
        coverage.append(round(sum(got[k] is not None for k in said) / max(len(said), 1), 3))
    return {"sentences": rows, "wav": str(wav), "duration": total / TARGET_RATE, "coverage": coverage}


def _report(n: int, coverage: list[float], cached: bool) -> None:
    low = [f"s{i} {c:.0%}" for i, c in enumerate(coverage) if c < MIN_COVERAGE]
    print(f"slide {n}: {'cached, ' if cached else ''}words matched "
          + " ".join(f"{c:.0%}" for c in coverage)
          + (f"  FLAG under {MIN_COVERAGE:.0%}: {', '.join(low)}" if low else ""))


def voice(deck: str, only: int | None = None, root: Path = ROOT, tts=None, asr=None) -> list[int]:
    """Voice every scripted slide (or ``only``); returns the slides (re)voiced."""
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
            key = hashlib.sha1(json.dumps([sents, VOICE_REL, VERSION]).encode()).hexdigest()
            cached = hf.exists() and hf.read_text() == key and tj.exists() and wav.exists()
            if cached:
                data = json.loads(tj.read_text())
            else:
                tts = tts or PiperTTS(ROOT / VOICE_REL)
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
