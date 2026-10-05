"""Task 4: voice step (fakes for Piper/Whistle) + deterministic HUD render + deck concat."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import wave

import pytest
from PIL import Image, ImageDraw

from src.tts import TARGET_RATE, PiperTTS
from tools.video.render import SHIM, concat, render_slide
from tools.video.script import words
from tools.video.voice import voice

FPS = 30


# ---- voice ------------------------------------------------------------------

class _FakeTTS(PiperTTS):
    """No voice model: every sentence is 0.5 s of a constant tone."""
    def __init__(self):
        pass

    def _pcm16k(self, text):
        return b"\x10\x00" * (TARGET_RATE // 2)


class _Boom:
    def __getattr__(self, name):
        raise AssertionError(f"cached slide must not touch the TTS ({name})")


def _boom_asr(pcm):
    raise AssertionError("cached slide must not call Whistle")


def test_voice_offsets_flags_and_caches(tmp_path, capsys):
    d = tmp_path / "script" / "D"
    d.mkdir(parents=True)
    md = d / "slide_002.md"
    md.write_text("# intro\nFirst it reads the page.\nThen it grades every answer.\n")
    heard = iter([["First", "it", "reads", "the", "page."], ["then", "it", "grades", "answer"]])
    sizes = []

    def asr(pcm):
        sizes.append(len(pcm))
        return [{"word": w, "start": 0.1 * k, "end": 0.1 * k + 0.1} for k, w in enumerate(next(heard))]

    assert voice("D", root=tmp_path, tts=_FakeTTS(), asr=asr) == [2]
    assert sizes == [TARGET_RATE, TARGET_RATE]            # 0.5 s each: the 350 ms gap is cut off
    tim = json.loads((tmp_path / "out/video/D/timings/slide_002.json").read_text())
    assert tim["duration"] == pytest.approx(1.7)          # 2 x (0.5 s + 0.35 s gap)
    assert [w["word"] for w in tim["sentences"][1]] == words("Then it grades every answer.")
    assert tim["sentences"][1][0] == {"word": "Then", "start": 0.85, "end": 0.95}   # + sentence start
    assert tim["coverage"] == [1.0, 0.8]                   # "every" was not heard
    assert "FLAG" in capsys.readouterr().out

    assert voice("D", root=tmp_path, tts=_Boom(), asr=_boom_asr) == []   # unchanged -> skipped
    md.write_text("First it reads the page.\n")
    heard = iter([["first", "it", "reads", "the", "page"]])
    assert voice("D", root=tmp_path, tts=_FakeTTS(), asr=asr) == [2]   # edited -> redone


# ---- render -----------------------------------------------------------------

def _box(x, y, w, h):
    return {"x": x, "y": y, "w": w, "h": h}


SLIDE = {
    "slide_number": 1,
    "elements": [
        {"id": "e1", "text": "Reading pages", "bbox": _box(1, 1, 5, 1), "line_bboxes": [_box(1, 1, 5, 1)],
         "line_texts": ["Reading pages"], "pointable": True, "map_unit": "u1"},
        {"id": "e2", "text": "Grading answers fast", "bbox": _box(1, 3.5, 5, 1.2),
         "line_bboxes": [_box(1, 3.5, 4, 0.5), _box(1, 4.1, 2, 0.5)],
         "line_texts": ["Grading answers", "fast"], "pointable": True, "map_unit": "u2"},
        {"id": "e3", "text": "8 GB memory", "bbox": _box(8, 5, 3, 1), "line_bboxes": [_box(8, 5, 3, 1)],
         "line_texts": ["8 GB memory"], "pointable": True, "map_unit": "u3"},
    ],
    "map": {"units": {"u1": ["e1"], "u2": ["e2"], "u3": ["e3"]},
            "relations": [{"from": "u1", "to": "u2", "type": "flow"}], "info": {}},
}
SIZE_IN = {"w": 13.333, "h": 7.5}
SENTS = [words("We start by reading pages."), words("Then grading answers needs eight GB memory.")]
SHEET = {"slide": 1, "version": 1, "cues": [
    {"effect": "focus", "target": "u1", "at": {"s": 0, "w": 3}, "spotlight": True},
    {"effect": "glide", "target": "u2", "at": {"s": 1, "w": 0}},
    {"effect": "underline", "target": "e2", "at": {"s": 1, "w": 2}},
    {"effect": "ring", "target": "e3", "at": {"s": 1, "w": 4}},
    {"effect": "zoom", "target": "e3", "at": {"s": 1, "w": 6}},
]}
ROWS = [[{"word": w, "start": round(0.1 + 0.15 * k, 3), "end": round(0.25 + 0.15 * k, 3)} for k, w in enumerate(SENTS[0])],
        [{"word": w, "start": round(1.0 + 0.12 * k, 3), "end": round(1.12 + 0.12 * k, 3)} for k, w in enumerate(SENTS[1])]]


@pytest.fixture
def ffmpeg(monkeypatch):
    if SHIM.is_dir():
        monkeypatch.setenv("PATH", f"{SHIM}{os.pathsep}{os.environ['PATH']}")
    exe = shutil.which("ffmpeg")
    if not exe or subprocess.run([exe, "-version"], capture_output=True).returncode:
        pytest.skip("no working ffmpeg")


def _decode(mp4, *fmt):
    return subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), *fmt, "-"],
                          capture_output=True, check=True).stdout


def _video(mp4, w=320, h=180):
    raw = _decode(mp4, "-an", "-f", "rawvideo", "-pix_fmt", "rgb24")
    n = w * h * 3
    return [raw[i:i + n] for i in range(0, len(raw), n)]


def _audio_s(mp4):
    return len(_decode(mp4, "-vn", "-f", "s16le", "-ac", "1", "-ar", str(TARGET_RATE))) / 2 / TARGET_RATE


@pytest.fixture
def inputs(tmp_path):
    png = tmp_path / "slide_001.png"
    img = Image.new("RGB", (320, 180), (235, 235, 240))
    d = ImageDraw.Draw(img)
    for e in SLIDE["elements"]:
        b = e["bbox"]
        d.rectangle([b["x"] * 24, b["y"] * 24, (b["x"] + b["w"]) * 24, (b["y"] + b["h"]) * 24], fill=(60, 90, 140))
    img.save(png)
    wav = tmp_path / "slide_001.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_RATE)
        w.writeframes(b"\x00\x00" * 2 * TARGET_RATE)        # 2.0 s of silence
    return png, wav


def test_render_is_deterministic_and_matches_wav(ffmpeg, inputs, tmp_path):
    png, wav = inputs
    hashes = []
    for k in range(2):
        mp4 = tmp_path / f"run{k}.mp4"
        dropped = render_slide(png, SLIDE, SIZE_IN, SHEET, SENTS, ROWS, wav, mp4, size=(320, 180), fps=FPS)
        assert dropped == []                               # every cue rendered (the plan is not empty)
        frames = _video(mp4)
        assert len(set(frames)) > 1                        # effects actually animate
        assert len(frames) / FPS == pytest.approx(2.0, abs=0.1)
        assert _audio_s(mp4) == pytest.approx(2.0, abs=0.1)
        hashes.append(hashlib.sha256(b"".join(frames)).hexdigest())
    assert hashes[0] == hashes[1]


def test_render_drops_failing_cues_and_refuses_stale_timings(ffmpeg, inputs, tmp_path):
    png, wav = inputs
    sheet = {"cues": [{"effect": "focus", "target": "u3", "at": {"s": 0, "w": 3}},   # fails the why rule
                      {"effect": "ring", "target": "nope", "at": {"s": 0, "w": 1}},
                      {"effect": "glide", "target": "u2", "at": {"s": 1, "w": 0}}]}  # valid until its focus goes
    dropped = render_slide(png, SLIDE, SIZE_IN, sheet, SENTS, ROWS, wav, tmp_path / "d.mp4", size=(320, 180))
    assert [e.split(":")[0] for e in dropped] == ["cue 0", "cue 1", "cue 2"]
    assert "why" in dropped[0] and "glide needs a current focus" in dropped[2]
    stale = [SENTS[0], words("Then grading answers needs eight GB of memory.")]
    with pytest.raises(ValueError, match="stale"):
        render_slide(png, SLIDE, SIZE_IN, SHEET, stale, ROWS, wav, tmp_path / "s.mp4", size=(320, 180))


def test_concat_adds_gap_between_slides(ffmpeg, inputs, tmp_path):
    png, wav = inputs
    mp4 = tmp_path / "one.mp4"
    render_slide(png, SLIDE, SIZE_IN, SHEET, SENTS, ROWS, wav, mp4, size=(320, 180))
    deck = tmp_path / "deck.mp4"
    concat([mp4, mp4], deck)
    assert len(_video(deck)) / FPS == pytest.approx(4.6, abs=0.15)   # 2 s + 0.6 s gap + 2 s
    assert _audio_s(deck) == pytest.approx(4.6, abs=0.15)
