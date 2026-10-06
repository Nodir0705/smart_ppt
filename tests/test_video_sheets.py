"""Contact sheets (tools/video/sheets.py), the lens in a real plan, and validate's warnings."""
from __future__ import annotations

import json
import math
import wave

from PIL import Image

from tools.video import paths
from tools.video.geometry import Letterbox, overlaps
from tools.video.render import build_plan, check
from tools.video.script import words
from tools.video.sheets import frames_for, sheets


def _box(x, y, w, h):
    return {"x": x, "y": y, "w": w, "h": h}


SLIDE = {
    "slide_number": 1,
    "elements": [
        {"id": "e0", "type": "title", "text": "Grading", "bbox": _box(0.5, 0.3, 4, 0.6), "pointable": False},
        {"id": "e1", "text": "Reading pages", "bbox": _box(1, 1.5, 4, 1), "line_bboxes": [_box(1, 1.5, 4, 1)],
         "line_texts": ["Reading pages"], "pointable": True, "map_unit": "u1"},
        {"id": "e2", "text": "Grading answers fast", "bbox": _box(1, 3.5, 4.5, 1.2),
         "line_bboxes": [_box(1, 3.5, 4, 0.5), _box(1, 4.1, 2, 0.5)],
         "line_texts": ["Grading answers", "fast"], "pointable": True, "map_unit": "u2"},
        {"id": "e3", "text": "8 GB memory", "bbox": _box(8, 5, 3, 1), "line_bboxes": [_box(8, 5, 3, 1)],
         "line_texts": ["8 GB memory"], "pointable": True, "map_unit": "u3"},
    ],
    "map": {"units": {"u1": ["e1"], "u2": ["e2"], "u3": ["e3"]},
            "relations": [{"from": "u1", "to": "u2", "type": "flow"}], "info": {}},
}
SIZE_IN = {"w": 13.333, "h": 7.5}
SCRIPT = "We start by reading pages.\nThen grading answers needs eight GB memory.\n"
SENTS = [words(s) for s in SCRIPT.splitlines()]
SHEET = {"slide": 1, "version": 1, "cues": [
    {"effect": "focus", "target": "u1", "at": {"s": 0, "w": 3}, "spotlight": True},
    {"effect": "glide", "target": "u2", "at": {"s": 1, "w": 0}},
    {"effect": "ring", "target": "nope", "at": {"s": 1, "w": 1}},               # dropped: no sheet
    {"effect": "underline", "target": "e2", "at": {"s": 1, "w": 2}},
    {"effect": "zoom", "target": "e3", "at": {"s": 1, "w": 6}},
]}
ROWS = [[{"word": w, "start": round(0.1 + 0.15 * k, 3), "end": round(0.25 + 0.15 * k, 3)} for k, w in enumerate(SENTS[0])],
        [{"word": w, "start": round(1.0 + 0.12 * k, 3), "end": round(1.12 + 0.12 * k, 3)} for k, w in enumerate(SENTS[1])]]


def _deck(tmp_path, sheet=SHEET):
    P = paths("D", tmp_path)
    for d in (P.script, P.index.parent, P.slides, P.cues, P.timings, P.voice):
        d.mkdir(parents=True)
    P.index.write_text(json.dumps({"slide_size": SIZE_IN, "slides": [SLIDE]}))
    Image.new("RGB", (320, 180), (220, 220, 225)).save(P.slides / "slide_001.png")
    with wave.open(str(P.voice / "slide_001.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 32000)                                      # 2.0 s
    (P.script / "slide_001.md").write_text(SCRIPT)
    (P.cues / "slide_001.json").write_text(json.dumps(sheet))
    (P.timings / "slide_001.json").write_text(json.dumps({"sentences": ROWS, "duration": 2.0}))
    return P


def test_frames_for_sits_on_the_render_grid():
    assert frames_for(1.0, 4.0, 200) == [33, 45, 60, 119]          # +0.1, +0.5, +1.0 s and the last frame
    assert frames_for(1.0, 1.3, 200) == [33, 38, 38, 38]           # short cue: clamped to its last frame
    assert frames_for(1.0, math.inf, 50) == [33, 45, 49, 49]       # held to the slide's end


def test_sheets_one_png_per_kept_cue(tmp_path, capsys):
    P = _deck(tmp_path)
    out = P.out / "sheets"
    out.mkdir(parents=True)
    (out / "slide_001_c9.png").write_bytes(b"stale")
    assert sheets("D", size=(320, 180), root=tmp_path) == 0
    log = capsys.readouterr().out
    assert "cue 2" in log and "dropped" in log
    names = sorted(f.name for f in out.iterdir())
    assert names == [f"slide_001_c{k}.png" for k in (0, 1, 3, 4)]
    img = Image.open(out / "slide_001_c4.png")
    assert img.width == 320 and img.height > 180                    # 2x2 half-size grid + the text band
    a = (out / "slide_001_c4.png").read_bytes()
    sheets("D", only=1, size=(320, 180), root=tmp_path)
    assert (out / "slide_001_c4.png").read_bytes() == a              # deterministic


def test_lens_in_a_plan_avoids_the_live_focus_and_the_title():
    lb = Letterbox(SIZE_IN["w"], SIZE_IN["h"], 320, 180, 640, 360)
    cues = [c for c in SHEET["cues"] if c["target"] != "nope"]
    plan = build_plan(cues, SLIDE, SENTS, ROWS, lb)
    zoom = next(e for e in plan if e.kind == "zoom")
    lens = zoom.extra["lens"]
    focus = next(e for e in plan if e.kind == "glide").box                     # u2, live during the zoom
    title = lb.box_px(SLIDE["elements"][0]["bbox"])
    assert not overlaps(lens, focus) and not overlaps(lens, zoom.box) and not overlaps(lens, title)


def test_validate_prints_dead_air_warnings(tmp_path, capsys):
    late = {"slide": 1, "version": 1, "cues": [{"effect": "underline", "target": "e2", "at": {"s": 1, "w": 2}}]}
    _deck(tmp_path, late)            # sentence 0 ("reading pages") refers to u1 at 0.1 s; first cue at 1.24 s
    assert check("D", root=tmp_path) == 0
    assert capsys.readouterr().out == "slide 1: VALID\n"
    rows = [[dict(w, start=w["start"] - 3.0, end=w["end"] - 3.0) for w in ROWS[0]], ROWS[1]]
    (tmp_path / "out/video/D/timings/slide_001.json").write_text(json.dumps({"sentences": rows}))
    assert check("D", root=tmp_path) == 0                          # a warning is not an error
    out = capsys.readouterr().out
    assert out.startswith("slide 1: VALID\n  warning: dead air")
