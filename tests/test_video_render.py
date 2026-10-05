"""Task 4: voice step (fakes for Piper/Whistle) + deterministic HUD render + deck concat."""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import wave
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from src.tts import TARGET_RATE, PiperTTS
import tools.video.render as R
from tools.video import paths
from tools.video.render import SHIM, check, concat, no_sheets, playable, render, render_slide
from tools.video.script import words
from tools.video.voice import voice, whistle_words

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


def test_voice_failure_leaves_no_stale_cache(tmp_path):
    """A failed re-voice has already overwritten the wav: no hash or timings may survive it."""
    md = tmp_path / "script/D/slide_002.md"
    md.parent.mkdir(parents=True)
    md.write_text("First it reads the page.\n")
    ok = lambda pcm: [{"word": "first", "start": 0.0, "end": 0.1}]
    assert voice("D", root=tmp_path, tts=_FakeTTS(), asr=ok) == [2]
    md.write_text("First it reads the page.\nThen it grades.\n")
    calls = []

    def flaky(pcm):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("whistle died")
        return ok(pcm)

    with pytest.raises(RuntimeError, match="whistle died"):
        voice("D", root=tmp_path, tts=_FakeTTS(), asr=flaky)
    T = tmp_path / "out/video/D/timings"
    assert not (T / "slide_002.hash").exists() and not (T / "slide_002.json").exists()  # render refuses
    md.write_text("First it reads the page.\n")                        # revert
    assert voice("D", root=tmp_path, tts=_FakeTTS(), asr=ok) == [2]     # re-voiced, not "cached"


def test_whistle_words_asks_for_word_timestamps():
    said = [{"word": "hi", "start": 0.0, "end": 0.2}]
    eng = SimpleNamespace(proc=SimpleNamespace(stdin=io.BytesIO()), _read=lambda: {"ok": True, "words": said})
    assert whistle_words(eng, b"\x01\x00" * 4) == said
    head, pcm = eng.proc.stdin.getvalue().split(b"\n", 1)
    assert json.loads(head) == {"n": 8, "words": True} and pcm == b"\x01\x00" * 4
    eng = SimpleNamespace(proc=SimpleNamespace(stdin=io.BytesIO()), _read=lambda: {"ok": False, "error": "x"})
    with pytest.raises(RuntimeError, match="whistle server error"):
        whistle_words(eng, b"\x00\x00")


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


def test_playable_drops_only_the_cue_at_fault():
    """A mark failing the why rule must not push a later valid mark over the limit."""
    sheet = {"cues": [{"effect": "underline", "target": "e2", "at": {"s": 1, "w": 2}},
                      {"effect": "ring", "target": "e3", "at": {"s": 1, "w": 4}},
                      {"effect": "ring", "target": "e1", "at": {"s": 1, "w": 0}}]}   # why fails, starts first
    cues, log = playable(sheet, SLIDE, SENTS, ROWS)
    assert [e.split(":")[0] for e in log] == ["cue 2"] and "why" in log[0]
    assert cues == sheet["cues"][:2]


def test_playable_drops_overlap_only_when_nothing_else_fails():
    """Three marks, each valid alone: only the one over the limit goes."""
    sheet = {"cues": [{"effect": "underline", "target": "e2", "at": {"s": 1, "w": 2}},
                      {"effect": "ring", "target": "e3", "at": {"s": 1, "w": 4}},
                      {"effect": "ring", "target": "e2", "at": {"s": 1, "w": 1}}]}
    cues, log = playable(sheet, SLIDE, SENTS, ROWS)
    assert len(cues) == 2 and len(log) == 1 and log[0].split(": ", 1)[1].startswith("concurrency:")


def test_render_deck_is_complete_or_not_built(ffmpeg, tmp_path, capsys, monkeypatch):
    P = paths("D", tmp_path)
    for d in (P.script, P.index.parent, P.slides, P.cues, P.timings, P.voice):
        d.mkdir(parents=True)
    P.index.write_text(json.dumps({"slide_size": SIZE_IN, "slides": [SLIDE, dict(SLIDE, slide_number=2)]}))
    script = "We start by reading pages.\nThen grading answers needs eight GB memory.\n"
    bad_cue = {"effect": "ring", "target": "nope", "at": {"s": 0, "w": 1}}
    for n, sheet in ((1, dict(SHEET, cues=SHEET["cues"] + [bad_cue])), (2, {"slide": 2, "version": 1, "cues": []})):
        Image.new("RGB", (320, 180), (200, 200, 200)).save(P.slides / f"slide_{n:03d}.png")
        with wave.open(str(P.voice / f"slide_{n:03d}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(TARGET_RATE)
            w.writeframes(b"\x00\x00" * 2 * TARGET_RATE)
        (P.script / f"slide_{n:03d}.md").write_text(script)
        (P.cues / f"slide_{n:03d}.json").write_text(json.dumps(sheet))
        (P.timings / f"slide_{n:03d}.json").write_text(json.dumps(   # wav moved: render uses P.voice
            {"sentences": ROWS, "wav": "/gone/slide.wav", "duration": 2.0, "coverage": [1.0, 1.0]}))
    (P.script / "slide_002_old.md").write_text("leftover, not a slide\n")
    (P.script / "slide_4.md").write_text(script)                         # not canonical: ignored
    deck, dh = P.out / "deck.mp4", P.out / "deck.hash"

    def run(**kw):
        rc = render("D", size=(320, 180), root=tmp_path, **kw)
        return rc, capsys.readouterr().out

    rc, out = run()
    assert rc == 0 and "dropped cue 5" in out and deck.exists()
    assert len(set(_video(P.out / "slide_002.mp4"))) == 1               # "cues": [] -> a plain slide
    good = (deck.read_bytes(), dh.read_text())

    rc, out = run()
    assert rc == 0 and out.count("cached") == 3 and "dropped cue 5" in out   # the drop log survives the cache

    def boom(*a, **kw):
        raise RuntimeError("ffmpeg died")

    s2 = P.cues / "slide_002.json"
    for site, gone in (("render_slide", "slide 2: cached"), ("concat", "deck: cached")):
        sheet2 = s2.read_text()                # a failed rebuild must not leave its old hash behind
        s2.write_text(json.dumps({"slide": 2, "version": 1, "cues": [], "edited": True}))
        with monkeypatch.context() as m:
            m.setattr(R, site, boom)
            with pytest.raises(RuntimeError, match="ffmpeg died"):
                run()
        capsys.readouterr()
        s2.write_text(sheet2)                                            # revert
        rc, out = run()
        assert rc == 0 and gone not in out
    good = (deck.read_bytes(), dh.read_text())

    tf = P.timings / "slide_001.json"
    tim1 = tf.read_text()
    tf.unlink()
    P.index.write_text(json.dumps({"slide_size": SIZE_IN, "slides": [SLIDE]}))
    rc, out = run()
    assert rc == 1 and "slide 1: no timings, run voice first" in out and "slide 2: not in the index" in out
    tf.write_text(tim1)
    P.index.write_text(json.dumps({"slide_size": SIZE_IN, "slides": [SLIDE, dict(SLIDE, slide_number=2)]}))

    (P.script / "slide_002.md").write_text(script.replace("GB memory", "GB of memory"))
    rc, out = run()
    assert rc == 1 and "rerun voice" in out and "deck: skipped (1 slides refused)" in out
    assert (deck.read_bytes(), dh.read_text()) == good                   # the old deck is left alone
    assert check("D", only=2, root=tmp_path) == 0                       # valid cues, stale timings: a note
    assert capsys.readouterr().out.startswith("slide 2: VALID\n  note: timings stale")

    (P.script / "slide_002.md").write_text(script)
    (P.script / "slide_003.md").write_text(script)
    rc, out = run()
    assert rc == 1 and "slide 3: no cue sheet, run direct" in out and "deck: skipped" in out
    rc, out = run(only=3)
    assert rc == 1 and out == no_sheets(P, 3) + "\n"
    for f in P.cues.iterdir():
        f.unlink()
    rc, out = run()
    assert rc == 1 and out == no_sheets(P, None) + "\n"
    assert (deck.read_bytes(), dh.read_text()) == good
