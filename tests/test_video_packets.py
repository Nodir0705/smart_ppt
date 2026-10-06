"""Task 5: agent packets (draft, direct), packet hashing, and the CLI's packet/validate steps."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys

from tools.video import ROOT, paths
from tools.video.__main__ import main
from tools.video.packets import GUIDANCE, VOCAB, direct, draft


def box(x, y, w, h):
    return {"x": x, "y": y, "w": w, "h": h, "units": "inches"}


def _slide(n, message, elements, units, info, relations=(), order=None):
    return {"slide_number": n, "elements": elements,
            "map": {"message": message, "units": units, "info": info,
                    "relations": list(relations), "order": order or list(units)}}


SLIDES = [
    _slide(1, "What the checker is.",
           [{"id": "s1_e0", "map_role": "title", "type": "title", "text": "AI Handout Checker",
             "bbox": box(0.7, 0.8, 6.0, 0.4), "line_texts": ["AI Handout Checker"],
             "line_bboxes": [box(0.7, 0.8, 6.0, 0.4)], "value": None, "pointable": True, "map_unit": "u1"}],
           {"u1": ["s1_e0"]}, {"u1": {"summary": "Title.", "talk": ["the checker"], "terms": ["a i"]}}),
    _slide(2, "Reading, then grading.",
           [{"id": "s2_e0", "map_role": "card_heading", "type": "body", "text": "Stage A read",
             "bbox": box(1.0, 2.0, 2.0, 0.3), "line_texts": ["Stage A read"],
             "line_bboxes": [box(1.0, 2.0, 2.0, 0.3)], "value": None, "pointable": True, "map_unit": "u1"},
            {"id": "s2_e1", "map_role": "bullet", "type": "bullet", "text": "Turns pixels into text catches circles",
             "bbox": box(1.0, 2.4, 2.4, 0.5), "line_texts": ["Turns pixels into text", "catches circles"],
             "line_bboxes": [box(1.0, 2.4, 2.4, 0.2), box(1.0, 2.7, 1.5, 0.2)],
             "value": None, "pointable": True, "map_unit": "u1"},
            {"id": "s2_e2", "map_role": "stat", "type": "body", "text": "8 GB",
             "bbox": box(5.0, 2.0, 1.0, 0.3), "line_texts": ["8 GB"],
             "line_bboxes": [box(5.0, 2.0, 1.0, 0.3)], "value": "8 GB", "pointable": True, "map_unit": "u2"},
            {"id": "s2_e3", "map_role": "footer", "type": "body", "text": "Page 2",
             "bbox": box(12.0, 7.0, 0.5, 0.1), "line_texts": ["Page 2"],
             "line_bboxes": [box(12.0, 7.0, 0.5, 0.1)], "value": None, "pointable": False}],
           {"u1": ["s2_e0", "s2_e1"], "u2": ["s2_e2"]},
           {"u1": {"summary": "Stage A reads.", "talk": ["first it reads the page"], "terms": ["stage a"]},
            "u2": {"summary": "Memory.", "talk": ["a modest gpu"], "terms": ["eight gigabytes"]}},
           relations=[{"from": "u1", "to": "u2", "type": "flow"}], order=["u1", "u2"]),
    _slide(3, "Thanks.", [], {}, {}),
]
SCRIPT = "# slide two\nFirst it reads the page.\nThen it fits in eight gigabytes.\n"


def _deck(root):
    P = paths("D", root)
    P.index.parent.mkdir(parents=True)
    P.index.write_text(json.dumps({"deck": "D", "slide_size": {"w": 13.333, "h": 7.5}, "slides": SLIDES}))
    P.script.mkdir(parents=True)
    return P


def _json(p):
    return json.loads(p.read_text())


# ---- draft ------------------------------------------------------------------

def test_draft_packets(tmp_path, capsys):
    P = _deck(tmp_path)
    (P.script / "slide_002.md").write_text(SCRIPT)
    assert draft("D", root=tmp_path) == [1, 3]                 # slide 2 already has a script
    out = capsys.readouterr().out
    assert "slide 2:" in out and out.rstrip().endswith("todo: 1 3")

    d = P.packets / "draft"
    p2 = _json(d / "slide_002.json")
    assert p2["slide"] == 2 and p2["png"] == str(P.slides / "slide_002.png")
    assert p2["message"] == "Reading, then grading."
    assert p2["units"]["u1"] == {"members": ["s2_e0", "s2_e1"], "summary": "Stage A reads.",
                                 "talk": ["first it reads the page"], "terms": ["stage a"]}
    assert p2["relations"] == [{"from": "u1", "to": "u2", "type": "flow"}]
    assert [e["id"] for e in p2["elements"]] == ["s2_e0", "s2_e1", "s2_e2"]       # footer is not pointable
    assert p2["elements"][1] == {"id": "s2_e1", "role": "bullet", "text": "Turns pixels into text catches circles",
                                 "bbox": [1.0, 2.4, 2.4, 0.5]}
    assert (p2["prev_message"], p2["next_message"]) == ("What the checker is.", "Thanks.")
    assert _json(d / "slide_001.json")["prev_message"] is None
    assert _json(d / "slide_003.json")["next_message"] is None

    assert draft("D", only=3, root=tmp_path) == [3]
    assert capsys.readouterr().out.splitlines() == [f"slide 3: {d / 'slide_003.json'}", "todo: 3"]
    (P.script / "slide_003.md").write_text("Thanks for listening.\n")
    assert draft("D", only=3, root=tmp_path) == []
    assert capsys.readouterr().out.splitlines()[-1] == "todo: none"


# ---- direct -----------------------------------------------------------------

def test_direct_packet_contents(tmp_path):
    P = _deck(tmp_path)
    (P.script / "slide_002.md").write_text(SCRIPT)
    assert direct("D", root=tmp_path) == [2]                   # slides 1 and 3 have no script
    p = _json(P.packets / "direct" / "slide_002.json")
    assert p["slide"] == 2 and p["png"] == str(P.slides / "slide_002.png")
    assert p["map"]["order"] == ["u1", "u2"] and p["map"]["message"] == "Reading, then grading."
    assert p["map"]["units"]["u2"]["terms"] == ["eight gigabytes"]
    assert p["map"]["relations"] == [{"from": "u1", "to": "u2", "type": "flow"}]
    assert p["elements"][1]["lines"] == [[0, "Turns pixels into text"], [1, "catches circles"]]
    assert p["elements"][2]["value"] == "8 GB" and p["elements"][2]["role"] == "stat"
    assert "s2_e3" not in {e["id"] for e in p["elements"]}
    assert p["sentences"] == [[[0, "First"], [1, "it"], [2, "reads"], [3, "the"], [4, "page."]],
                              [[0, "Then"], [1, "it"], [2, "fits"], [3, "in"], [4, "eight"], [5, "gigabytes."]]]
    assert p["vocabulary"] == VOCAB and GUIDANCE in json.dumps(p["vocabulary"])
    assert len(p["hash"]) == 40


def test_direct_hash_tracks_content_not_location(tmp_path):
    P = _deck(tmp_path / "a")
    (P.script / "slide_002.md").write_text(SCRIPT)
    pk = P.packets / "direct" / "slide_002.json"
    direct("D", root=tmp_path / "a")
    h = _json(pk)["hash"]
    direct("D", root=tmp_path / "a")
    assert _json(pk)["hash"] == h                              # unchanged inputs -> same hash

    shutil.copytree(tmp_path / "a" / "script", tmp_path / "b" / "script")
    shutil.copytree(tmp_path / "a" / "out" / "bench", tmp_path / "b" / "out" / "bench")
    direct("D", root=tmp_path / "b")                           # another checkout: png path differs
    assert _json(paths("D", tmp_path / "b").packets / "direct" / "slide_002.json")["hash"] == h

    (P.script / "slide_002.md").write_text(SCRIPT + "# a comment changes nothing\n")
    direct("D", root=tmp_path / "a")
    assert _json(pk)["hash"] == h
    (P.script / "slide_002.md").write_text(SCRIPT.replace("eight gigabytes", "eight gigs"))
    direct("D", root=tmp_path / "a")
    assert _json(pk)["hash"] != h                              # a changed sentence -> new hash


def test_direct_todo_is_slides_whose_sheet_hash_differs(tmp_path, capsys):
    P = _deck(tmp_path)
    for n in (1, 2):
        (P.script / f"slide_{n:03d}.md").write_text(SCRIPT)
    (P.script / "slide_003.md").write_text("# nothing to say yet\n")
    assert direct("D", root=tmp_path) == [1, 2]                # no cue sheets yet
    out = capsys.readouterr().out
    assert "slide 3: empty script, no packet" in out and out.rstrip().endswith("todo: 1 2")

    P.cues.mkdir(parents=True)
    h1 = _json(P.packets / "direct" / "slide_001.json")["hash"]
    (P.cues / "slide_001.json").write_text(json.dumps({"slide": 1, "version": 1, "hash": h1, "cues": []}))
    (P.cues / "slide_002.json").write_text(json.dumps({"slide": 2, "version": 1, "cues": []}))   # no hash
    assert direct("D", root=tmp_path) == [2]
    out = capsys.readouterr().out
    assert "slide 1: fresh" in out and "slide 2: stale" in out
    (P.cues / "slide_002.json").write_text("{not json")
    assert direct("D", root=tmp_path) == [2]
    (P.script / "slide_001.md").write_text(SCRIPT.replace("reads", "scans"))   # user edit
    assert direct("D", root=tmp_path) == [1, 2]
    assert direct("D", only=2, root=tmp_path) == [2]
    capsys.readouterr()


# ---- CLI --------------------------------------------------------------------

def test_cli_packets_and_validate_exit_codes(tmp_path, capsys):
    P = _deck(tmp_path)
    (P.script / "slide_002.md").write_text(SCRIPT)
    root = ["--root", str(tmp_path)]
    assert main(["packets-draft", "D", *root]) == 0
    assert main(["packets-direct", "D", *root]) == 0
    assert (P.packets / "draft" / "slide_001.json").exists() and (P.packets / "direct" / "slide_002.json").exists()
    assert capsys.readouterr().out.rstrip().endswith("todo: 2")

    assert main(["validate", "D", *root]) == 1                 # no cue sheets yet
    assert capsys.readouterr().out.startswith("no cue sheets")
    P.cues.mkdir(parents=True)
    good = {"slide": 2, "version": 1, "cues": [{"effect": "focus", "target": "u1", "at": {"s": 0, "w": 2}}]}
    (P.cues / "slide_002.json").write_text(json.dumps(good))
    assert main(["validate", "D", "--only", "2", *root]) == 0
    assert capsys.readouterr().out.startswith("slide 2: VALID")
    bad = {"slide": 2, "version": 1, "cues": [{"effect": "ring", "target": "s2_e3", "at": {"s": 1, "w": 4}}]}
    (P.cues / "slide_002.json").write_text(json.dumps(bad))
    assert main(["validate", "D", "--only", "2", *root]) == 1
    assert "not pointable" in capsys.readouterr().out
    r = subprocess.run([sys.executable, "-m", "tools.video", "validate", "D", "--only", "2", *root],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 1 and "slide 2: 1 errors" in r.stdout   # the process exit code, not just main()


def test_direct_workflow_carries_the_guidance_verbatim():
    assert GUIDANCE in (ROOT / ".claude" / "workflows" / "video-direct.js").read_text()
