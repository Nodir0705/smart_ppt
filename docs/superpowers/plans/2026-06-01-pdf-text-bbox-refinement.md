# PDF-text-layer bbox refinement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Snap each text element's `bbox` to glyph-exact coordinates from the soffice PDF text layer and emit per-line sub-boxes, without changing the parser/matcher/viewer/enrichment contracts.

**Architecture:** Keep the `python-pptx` parser unchanged. Add a post-parse transform that opens the cached soffice PDF (already produced by the render step), reads per-word coordinates with `pdfplumber`, assigns words to elements by geometric containment, and writes refined `bbox` + `line_bboxes` into a new `out/index.refined.json`. PDF points ÷ 72 = inches — the same coordinate system the parser already emits.

**Tech Stack:** Python 3.10, `pdfplumber` (already installed), `python-pptx`, `soffice` (already used by the render step), `pytest`.

**Spec:** `docs/superpowers/specs/2026-06-01-pdf-text-bbox-refinement-design.md`

> **Git note:** this directory is not currently a git repo. Task 0 initializes one so the commit steps work. If you prefer no version control, skip Task 0 and the `git commit` steps — every other step stands alone.

> **Working directory:** all commands assume the project root `/home/netvision/smartppt` as cwd (where `pytest.ini` lives). Tests reference `test_deck.pptx` by relative path, matching existing tests.

---

## File Structure

- `src/render.py` — **modify.** Add cached `deck_pdf(pptx_path, pdf_dir)`; refactor `render_deck` to rasterize from it. PNG output stays byte-for-byte identical (guarded by existing `tests/test_render.py`).
- `src/refine_text.py` — **create.** Pure helpers (`words_to_inches`, `assign_words`, `cluster_lines`, `union_bbox`, refinement rule) + `refine_index` + thin CLI.
- `tests/test_refine_text.py` — **create.** Unit tests (no soffice) + one soffice-gated integration test.
- `out/test_deck.pdf` — new cached artifact (generated, not committed).
- `out/index.refined.json` — new output artifact (generated, not committed).

---

## Task 0: Initialize git (optional, enables commits)

**Files:** none (repo metadata only).

- [ ] **Step 1: Init repo and ignore generated artifacts**

```bash
cd /home/netvision/smartppt
git init
printf '%s\n' 'out/' '__pycache__/' '*.pyc' '.pytest_cache/' '.ruff_cache/' > .gitignore
git add .gitignore
git commit -m "chore: init repo with gitignore for generated artifacts"
```

- [ ] **Step 2: Snapshot current state as the baseline**

```bash
git add -A
git commit -m "chore: baseline before PDF text-bbox refinement"
```

---

## Task 1: Cache the PDF in `render.py`

**Files:**
- Modify: `src/render.py`
- Test: `tests/test_refine_text.py` (new file; first test lands here)

- [ ] **Step 1: Write the failing test for `deck_pdf` caching**

Create `tests/test_refine_text.py` with:

```python
"""Tests for PDF caching (render.deck_pdf) and text-bbox refinement."""
from __future__ import annotations

import shutil

import pytest

SOFFICE_MISSING = shutil.which("soffice") is None


@pytest.mark.skipif(SOFFICE_MISSING, reason="soffice not installed")
def test_deck_pdf_caches(tmp_path):
    from src.render import deck_pdf

    pdf = deck_pdf("test_deck.pptx", tmp_path)
    assert pdf.exists()
    assert pdf.suffix == ".pdf"
    mtime = pdf.stat().st_mtime

    # Second call must be a cache hit: same path, not regenerated.
    pdf2 = deck_pdf("test_deck.pptx", tmp_path)
    assert pdf2 == pdf
    assert pdf2.stat().st_mtime == mtime
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_refine_text.py::test_deck_pdf_caches -v`
Expected: FAIL with `ImportError: cannot import name 'deck_pdf' from 'src.render'`
(If soffice is absent the test is SKIPPED — that is acceptable; proceed and rely on Task 3's gated integration test instead.)

- [ ] **Step 3: Add `deck_pdf` and refactor `render_deck` to use it**

In `src/render.py`, add these two functions above `render_deck`:

```python
def _pdf_is_fresh(pptx_path: Path, pdf_path: Path) -> bool:
    return pdf_path.exists() and pdf_path.stat().st_mtime >= pptx_path.stat().st_mtime


def deck_pdf(pptx_path: str | Path, pdf_dir: str | Path = "out") -> Path:
    """Produce and cache `pdf_dir/<stem>.pdf` from the pptx (mtime-keyed)."""
    pptx_path = Path(pptx_path).resolve()
    pdf_dir = Path(pdf_dir).resolve()
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / (pptx_path.stem + ".pdf")
    if _pdf_is_fresh(pptx_path, pdf_path):
        return pdf_path

    with tempfile.TemporaryDirectory(prefix="smartppt_pdf_") as td:
        td_path = Path(td)
        r = subprocess.run(
            [
                "soffice", "--headless", "--nologo", "--nofirststartwizard",
                "--convert-to", "pdf", "--outdir", str(td_path), str(pptx_path),
            ],
            check=False, capture_output=True, text=True, timeout=120,
        )
        produced = td_path / (pptx_path.stem + ".pdf")
        if not produced.exists():
            raise RuntimeError(
                f"soffice failed to produce PDF.\nstdout: {r.stdout}\nstderr: {r.stderr}"
            )
        shutil.move(str(produced), str(pdf_path))
    return pdf_path
```

Then replace the body of `render_deck`'s `with tempfile.TemporaryDirectory(...)` block so the `# 1) pptx → pdf` step is gone and the PDF comes from `deck_pdf`. The new `render_deck` reads:

```python
def render_deck(pptx_path: str | Path, out_dir: str | Path, dpi: int = 150) -> list[Path]:
    """Render `pptx_path` into `out_dir/slide_NNN.png`. Returns ordered PNG paths."""
    pptx_path = Path(pptx_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = _count_slides(pptx_path)

    if _pngs_are_fresh(pptx_path, out_dir, expected):
        return _existing_pngs(out_dir)

    # Wipe stale PNGs so cache stays consistent
    for p in _existing_pngs(out_dir):
        p.unlink()

    # 1) pptx → pdf (cached at out_dir.parent/<stem>.pdf so refine_text reuses it)
    pdf_path = deck_pdf(pptx_path, out_dir.parent)

    with tempfile.TemporaryDirectory(prefix="smartppt_render_") as td:
        td_path = Path(td)
        # 2) pdf → png (pdftoppm writes <prefix>-1.png, -2.png, ...)
        prefix = td_path / "slide"
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), str(pdf_path), str(prefix)],
            check=True, capture_output=True, text=True, timeout=120,
        )
        # 3) Move into out_dir with zero-padded names (slide_001.png, ...)
        produced = sorted(td_path.glob("slide-*.png"))
        if len(produced) != expected:
            raise RuntimeError(
                f"pdftoppm produced {len(produced)} pages, expected {expected}"
            )
        for i, src in enumerate(produced, start=1):
            dst = out_dir / f"slide_{i:03d}.png"
            shutil.move(str(src), str(dst))

    return _existing_pngs(out_dir)
```

(The existing imports `shutil`, `subprocess`, `tempfile`, `Path` are already at the top of the file — no new imports needed.)

- [ ] **Step 4: Run the new test and the render regression tests**

Run: `python3 -m pytest tests/test_refine_text.py::test_deck_pdf_caches tests/test_render.py -v`
Expected: PASS (or `test_deck_pdf_caches` SKIPPED if no soffice); all 4 `test_render.py` tests PASS — proves PNG output is unchanged by the refactor.

- [ ] **Step 5: Commit**

```bash
git add src/render.py tests/test_refine_text.py
git commit -m "feat(render): cache soffice PDF via deck_pdf, reuse in render_deck"
```

---

## Task 2: Pure refinement core in `src/refine_text.py`

**Files:**
- Create: `src/refine_text.py`
- Test: `tests/test_refine_text.py`

- [ ] **Step 1: Write failing unit tests for the pure helpers**

Append to `tests/test_refine_text.py`:

```python
def test_words_to_inches_scale_one():
    from src.refine_text import words_to_inches

    words = [{"text": "Hi", "x0": 72.0, "x1": 144.0, "top": 36.0, "bottom": 72.0}]
    out = words_to_inches(words, 1.0, 1.0)
    assert out[0]["text"] == "Hi"
    assert out[0]["x0"] == pytest.approx(1.0)
    assert out[0]["x1"] == pytest.approx(2.0)
    assert out[0]["y0"] == pytest.approx(0.5)
    assert out[0]["y1"] == pytest.approx(1.0)


def test_assign_words_smallest_box_wins():
    from src.refine_text import assign_words

    big = {"x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0, "units": "inches"}
    small = {"x": 1.0, "y": 1.0, "w": 2.0, "h": 2.0, "units": "inches"}
    word = {"text": "x", "x0": 1.5, "y0": 1.5, "x1": 2.0, "y1": 2.0}  # center (1.75,1.75)
    assigned = assign_words([("big", big), ("small", small)], [word])
    assert assigned["small"] == [word]
    assert assigned["big"] == []


def test_assign_words_drops_outside():
    from src.refine_text import assign_words

    box = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "units": "inches"}
    word = {"text": "x", "x0": 5.0, "y0": 5.0, "x1": 5.5, "y1": 5.5}
    assigned = assign_words([("e", box)], [word])
    assert assigned["e"] == []


def test_cluster_lines_two_rows():
    from src.refine_text import cluster_lines

    w = lambda t, x0, y0: {"text": t, "x0": x0, "y0": y0, "x1": x0 + 0.5, "y1": y0 + 0.2}
    words = [w("b", 2.0, 0.0), w("a", 1.0, 0.0), w("c", 1.0, 1.0)]  # row0: a,b ; row1: c
    lines = cluster_lines(words, y_tol=0.03)
    assert [[x["text"] for x in line] for line in lines] == [["a", "b"], ["c"]]


def test_union_bbox_tight():
    from src.refine_text import union_bbox

    words = [
        {"x0": 1.0, "y0": 1.0, "x1": 2.0, "y1": 1.5},
        {"x0": 2.5, "y0": 1.1, "x1": 3.0, "y1": 1.4},
    ]
    box = union_bbox(words)
    assert box == {"x": 1.0, "y": 1.0, "w": 2.0, "h": 0.5, "units": "inches"}


def test_apply_refinement_no_words_keeps_shape():
    from src.refine_text import _apply_refinement

    e = {"id": "e1", "type": "body", "text": "hi",
         "bbox": {"x": 1.0, "y": 1.0, "w": 4.0, "h": 2.0, "units": "inches"}}
    _apply_refinement(e, [])
    assert e["bbox_source"] == "shape"
    assert e["line_bboxes"] == []
    assert e["shape_bbox"] == {"x": 1.0, "y": 1.0, "w": 4.0, "h": 2.0, "units": "inches"}
    assert e["bbox"] == e["shape_bbox"]


def test_apply_refinement_sets_pdf_text():
    from src.refine_text import _apply_refinement

    e = {"id": "e1", "type": "body", "text": "hi there",
         "bbox": {"x": 0.0, "y": 0.0, "w": 9.0, "h": 3.0, "units": "inches"}}
    words = [
        {"text": "hi", "x0": 1.0, "y0": 1.0, "x1": 1.5, "y1": 1.2},
        {"text": "there", "x0": 1.6, "y0": 1.0, "x1": 2.4, "y1": 1.2},
    ]
    _apply_refinement(e, words)
    assert e["bbox_source"] == "pdf_text"
    assert e["shape_bbox"]["w"] == 9.0
    assert e["bbox"] == {"x": 1.0, "y": 1.0, "w": 1.4, "h": 0.2, "units": "inches"}
    assert len(e["line_bboxes"]) == 1
    assert e["bbox"]["w"] < e["shape_bbox"]["w"]  # tighter than the shape box
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_refine_text.py -k "words_to_inches or assign_words or cluster_lines or union_bbox or apply_refinement" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.refine_text'`

- [ ] **Step 3: Write `src/refine_text.py` pure core**

Create `src/refine_text.py`:

```python
"""Refine text-element bboxes using the soffice PDF text layer (offline, exact).

The python-pptx parser gives each shape's container box, not where the glyphs
land inside it. This transform opens the cached PDF (same one the renderer
produces), reads per-word coordinates with pdfplumber, assigns words to elements
by geometric containment, and rewrites each text element's `bbox` to the tight
glyph union plus per-line `line_bboxes`. PDF points / 72 = inches — the units the
parser already emits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pdfplumber

from src.render import deck_pdf

PT_PER_INCH = 72.0

# Visual or geometry-tuned elements are left untouched (chart sub-regions are
# tuned by chart_geom; images/arrows carry no PDF text worth snapping to).
EXCLUDED_TYPES = {"chart", "chart_category", "image", "picture", "arrow"}


def words_to_inches(words: list[dict], sx: float, sy: float) -> list[dict]:
    """Convert pdfplumber word dicts (points) to inch-space word dicts.

    Input words have x0/x1/top/bottom in points. Output dicts have
    text/x0/y0/x1/y1 in inches (top-left origin).
    """
    out = []
    for w in words:
        out.append({
            "text": w["text"],
            "x0": w["x0"] / PT_PER_INCH * sx,
            "y0": w["top"] / PT_PER_INCH * sy,
            "x1": w["x1"] / PT_PER_INCH * sx,
            "y1": w["bottom"] / PT_PER_INCH * sy,
        })
    return out


def _center(w: dict) -> tuple[float, float]:
    return ((w["x0"] + w["x1"]) / 2.0, (w["y0"] + w["y1"]) / 2.0)


def _box_area(b: dict) -> float:
    return max(0.0, b["w"]) * max(0.0, b["h"])


def _contains(b: dict, cx: float, cy: float) -> bool:
    return (b["x"] <= cx <= b["x"] + b["w"]) and (b["y"] <= cy <= b["y"] + b["h"])


def assign_words(elements: list[tuple[str, dict]], words: list[dict]) -> dict:
    """Map element id -> list of inch-words whose center it owns.

    `elements` is a list of (element_id, shape_bbox). Each word is assigned to
    the smallest-area element whose box contains the word center; words inside no
    box are dropped.
    """
    assigned: dict[str, list[dict]] = {eid: [] for eid, _ in elements}
    for w in words:
        cx, cy = _center(w)
        best_id = None
        best_area = None
        for eid, bbox in elements:
            if _contains(bbox, cx, cy):
                area = _box_area(bbox)
                if best_area is None or area < best_area:
                    best_area = area
                    best_id = eid
        if best_id is not None:
            assigned[best_id].append(w)
    return assigned


def cluster_lines(words: list[dict], y_tol: float = 0.03) -> list[list[dict]]:
    """Group inch-words into lines (top->bottom), words within a line left->right.

    A word joins a line when its vertical span overlaps the line's span, or their
    vertical centers are within `y_tol` inches.
    """
    if not words:
        return []
    ws = sorted(words, key=lambda w: ((w["y0"] + w["y1"]) / 2.0, w["x0"]))
    lines: list[list[dict]] = []
    for w in ws:
        cy = (w["y0"] + w["y1"]) / 2.0
        placed = False
        for line in lines:
            ly0 = min(x["y0"] for x in line)
            ly1 = max(x["y1"] for x in line)
            lcy = (ly0 + ly1) / 2.0
            overlap = min(w["y1"], ly1) - max(w["y0"], ly0)
            if overlap > 0 or abs(cy - lcy) <= y_tol:
                line.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])
    for line in lines:
        line.sort(key=lambda w: w["x0"])
    lines.sort(key=lambda line: min(w["y0"] for w in line))
    return lines


def union_bbox(words: list[dict]) -> dict | None:
    """Tight inch bbox enclosing the given inch-words, or None if empty."""
    if not words:
        return None
    x0 = min(w["x0"] for w in words)
    y0 = min(w["y0"] for w in words)
    x1 = max(w["x1"] for w in words)
    y1 = max(w["y1"] for w in words)
    return {
        "x": round(x0, 4), "y": round(y0, 4),
        "w": round(x1 - x0, 4), "h": round(y1 - y0, 4),
        "units": "inches",
    }


def _is_refinable(e: dict) -> bool:
    return (
        e.get("type") not in EXCLUDED_TYPES
        and not e.get("needs_vision", False)
        and bool((e.get("text") or "").strip())
    )


def _apply_refinement(e: dict, words: list[dict]) -> None:
    """Mutate element `e` in place: preserve shape_bbox, set bbox/line_bboxes."""
    e["shape_bbox"] = dict(e["bbox"])
    if not words:
        e["line_bboxes"] = []
        e["bbox_source"] = "shape"
        return
    lines = cluster_lines(words)
    e["line_bboxes"] = [union_bbox(line) for line in lines]
    e["bbox"] = union_bbox(words)
    e["bbox_source"] = "pdf_text"
```

- [ ] **Step 4: Run the unit tests to verify they pass**

Run: `python3 -m pytest tests/test_refine_text.py -k "words_to_inches or assign_words or cluster_lines or union_bbox or apply_refinement" -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/refine_text.py tests/test_refine_text.py
git commit -m "feat(refine): pure PDF-word -> element bbox refinement core"
```

---

## Task 3: `refine_index` + CLI, with gated integration test

**Files:**
- Modify: `src/refine_text.py` (add `refine_index` + `main`)
- Test: `tests/test_refine_text.py`

- [ ] **Step 1: Write the failing integration + no-op tests**

Append to `tests/test_refine_text.py`:

```python
def _slide(index, n):
    return next(s for s in index["slides"] if s["slide_number"] == n)


def test_refine_index_pure_no_pdf_words(monkeypatch):
    """With an empty PDF (no words on any page), every text element falls back
    to shape source — no crash, schema additive."""
    import src.refine_text as rt

    index = {
        "deck": "x.pptx",
        "slide_size": {"w": 13.333, "h": 7.5, "units": "inches"},
        "slides": [{
            "slide_number": 1,
            "elements": [
                {"id": "s1_e0", "type": "body", "text": "hello",
                 "bbox": {"x": 1.0, "y": 1.0, "w": 4.0, "h": 1.0, "units": "inches"},
                 "needs_vision": False},
                {"id": "s1_e1", "type": "image", "text": "",
                 "bbox": {"x": 0.0, "y": 0.0, "w": 2.0, "h": 2.0, "units": "inches"},
                 "needs_vision": True},
            ],
        }],
    }

    class _Page:
        width, height = 960.0, 540.0
        def extract_words(self):
            return []

    class _PDF:
        pages = [_Page()]
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(rt.pdfplumber, "open", lambda _p: _PDF())
    out = rt.refine_index(index, "ignored.pdf")
    e0 = out["slides"][0]["elements"][0]
    assert e0["bbox_source"] == "shape"          # refinable but no words
    assert e0["line_bboxes"] == []
    assert "bbox_source" not in out["slides"][0]["elements"][1]  # image untouched


@pytest.mark.skipif(SOFFICE_MISSING, reason="soffice not installed")
def test_refine_index_real_deck_tightens_text(tmp_path):
    import json as _json

    from src.refine_text import refine_index
    from src.render import deck_pdf

    index = _json.load(open("out/index.json"))
    pdf = deck_pdf("test_deck.pptx", tmp_path)
    refined = refine_index(index, pdf)

    slide2 = _slide(refined, 2)
    pdf_text = [e for e in slide2["elements"] if e.get("bbox_source") == "pdf_text"]
    assert pdf_text, "expected at least one refined text element on slide 2"

    sw = refined["slide_size"]["w"]
    sh = refined["slide_size"]["h"]
    eps = 0.05

    # At least one element is strictly tighter horizontally than its shape box.
    assert any(e["bbox"]["w"] < e["shape_bbox"]["w"] - eps for e in pdf_text)

    for e in pdf_text:
        assert e["line_bboxes"], f"{e['id']} has no line boxes"
        for ln in e["line_bboxes"]:
            # each line sits within the original shape box (+ tolerance)
            assert ln["x"] >= e["shape_bbox"]["x"] - eps
            assert ln["y"] >= e["shape_bbox"]["y"] - eps
            assert ln["x"] + ln["w"] <= e["shape_bbox"]["x"] + e["shape_bbox"]["w"] + eps
            assert ln["y"] + ln["h"] <= e["shape_bbox"]["y"] + e["shape_bbox"]["h"] + eps
            # and within slide bounds
            assert -eps <= ln["x"] and ln["x"] + ln["w"] <= sw + eps
            assert -eps <= ln["y"] and ln["y"] + ln["h"] <= sh + eps
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_refine_text.py -k "refine_index" -v`
Expected: FAIL with `ImportError: cannot import name 'refine_index' from 'src.refine_text'`

- [ ] **Step 3: Add `refine_index` and `main` to `src/refine_text.py`**

Append to `src/refine_text.py`:

```python
def refine_index(index: dict, pdf_path) -> dict:
    """Refine every text element's bbox in `index` (mutated and returned)."""
    slide_size = index["slide_size"]
    with pdfplumber.open(str(pdf_path)) as pdf:
        pages = pdf.pages
        if len(pages) != len(index["slides"]):
            raise RuntimeError(
                f"PDF has {len(pages)} pages but index has "
                f"{len(index['slides'])} slides"
            )
        for slide, page in zip(index["slides"], pages):
            sx = slide_size["w"] / (page.width / PT_PER_INCH)
            sy = slide_size["h"] / (page.height / PT_PER_INCH)
            words = words_to_inches(page.extract_words(), sx, sy)
            refinable = [(e["id"], e["bbox"]) for e in slide["elements"]
                         if _is_refinable(e)]
            assigned = assign_words(refinable, words)
            for e in slide["elements"]:
                if _is_refinable(e):
                    _apply_refinement(e, assigned.get(e["id"], []))
    return index


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Refine text element bboxes via the PDF text layer."
    )
    ap.add_argument("--input", default="out/index.json")
    ap.add_argument("--out", default="out/index.refined.json")
    ap.add_argument("--deck", default=None,
                    help="path to .pptx (default: index['deck'])")
    ap.add_argument("--pdf-dir", default="out",
                    help="where to cache the soffice PDF")
    args = ap.parse_args(argv)

    index = json.loads(Path(args.input).read_text())
    deck = args.deck or index["deck"]
    pdf_path = deck_pdf(deck, args.pdf_dir)
    refined = refine_index(index, pdf_path)
    Path(args.out).write_text(json.dumps(refined, indent=2))

    n = sum(1 for s in refined["slides"] for e in s["elements"]
            if e.get("bbox_source") == "pdf_text")
    total = sum(1 for s in refined["slides"] for e in s["elements"]
                if _is_refinable(e))
    print(f"refined {n}/{total} text elements → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the integration tests to verify they pass**

Run: `python3 -m pytest tests/test_refine_text.py -k "refine_index" -v`
Expected: PASS (`test_refine_index_pure_no_pdf_words` always; `test_refine_index_real_deck_tightens_text` PASS with soffice, else SKIPPED)

- [ ] **Step 5: Run the CLI end-to-end and eyeball the summary**

Run: `python3 -m src.refine_text`
Expected: prints `refined N/M text elements → out/index.refined.json` with `N > 0`, and `out/index.refined.json` exists.
Sanity check one element:

```bash
python3 -c "import json; d=json.load(open('out/index.refined.json')); s=next(s for s in d['slides'] if s['slide_number']==2); e=next(e for e in s['elements'] if e.get('bbox_source')=='pdf_text'); print('refined w=%.2f shape w=%.2f lines=%d' % (e['bbox']['w'], e['shape_bbox']['w'], len(e['line_bboxes'])))"
```
Expected: `refined w` clearly smaller than `shape w`, `lines >= 1`.

- [ ] **Step 6: Commit**

```bash
git add src/refine_text.py tests/test_refine_text.py
git commit -m "feat(refine): refine_index transform + CLI writing index.refined.json"
```

---

## Task 4: Full regression + docs

**Files:**
- Modify: `tasks/todo.md`

- [ ] **Step 1: Run the entire test suite**

Run: `python3 -m pytest -q`
Expected: all previously-green tests still pass plus the new ones — `13X passed` (126 prior + the new refine tests), 0 failed. The 4 `test_render.py` tests passing confirms the `render.py` refactor kept PNG output identical.

- [ ] **Step 2: Visual spot-check with the existing headless preview (optional but recommended)**

Render a preview pointing the viewer geometry at a refined element to confirm the tighter box lands on the glyphs:

```bash
python3 -m src.render test_deck.pptx
python3 -m src.preview --slide 2 --phrase "parse the deck offline" --out /tmp/refine_check.png --index out/index.refined.json
```
Expected: a PNG at `/tmp/refine_check.png` with the highlight hugging the bullet text. (If `src.preview` does not accept `--index`, skip this step — it is a manual visual aid, not a gate.)

- [ ] **Step 3: Record the milestone in `tasks/todo.md`**

Append this section to `tasks/todo.md`:

```markdown
## PDF text-layer bbox refinement (2026-06-01)

**Status:** complete. Text element bboxes are now glyph-exact from the soffice
PDF text layer, with per-line sub-boxes.

- [x] `src/render.py` — cached `deck_pdf()`; `render_deck` rasterizes from it (PNG output unchanged, guarded by test_render.py)
- [x] `src/refine_text.py` — pure core (`words_to_inches`, `assign_words` w/ smallest-box tiebreak, `cluster_lines`, `union_bbox`) + `refine_index` + CLI
- [x] `tests/test_refine_text.py` — 7 unit + 2 integration (1 soffice-gated)
- [x] `out/index.refined.json` — index with refined `bbox`, additive `shape_bbox`, `line_bboxes`, `bbox_source`

**Design choices**
- PDF points / 72 = inches, scale derived per page (`slide_w / (page_w_pt/72)`)
  so it is robust to any soffice page scaling.
- Words assigned to the smallest-area containing element so merged
  diagram_label/stat boxes don't steal each other's text.
- `{chart, chart_category, image, picture, arrow}` excluded — chart sub-regions
  stay as tuned by chart_geom; visual elements carry no snap-worthy PDF text.
- No-match elements fall back to `bbox_source="shape"` with the original box, so
  text baked into images never regresses.

**Scope boundary:** per-line *data* is emitted now. Highlighting a single spoken
line (matcher picks which line; viewer draws that one box) is a noted follow-up.

**How to run**
\```
python3 -m src.refine_text                                  # out/index.json → out/index.refined.json
python3 -m src.refine_text --input out/index.enriched.json --out out/index.refined.json
\```
```

- [ ] **Step 4: Commit**

```bash
git add tasks/todo.md
git commit -m "docs: record PDF text-layer bbox refinement milestone"
```

---

## Self-Review Notes

- **Spec coverage:** `deck_pdf` caching (Task 1) ✓; pure core `words_to_inches`/`assign_words` smallest-box tiebreak/`cluster_lines`/`union_bbox` (Task 2) ✓; exclude set + no-match fallback + schema additions `shape_bbox`/`line_bboxes`/`bbox_source` (Task 2–3) ✓; standalone CLI matching enrich convention (Task 3) ✓; per-page defensive scale (Task 3) ✓; unit + gated integration + regression tests (Tasks 2–4) ✓; scope boundary recorded (Task 4) ✓.
- **Type consistency:** box dict is always `{x,y,w,h,units}`; inch-word dict is always `{text,x0,y0,x1,y1}`; pdfplumber word dict (`x0,x1,top,bottom`) is only read inside `words_to_inches`. Function names match across tasks (`deck_pdf`, `refine_index`, `_apply_refinement`, `_is_refinable`).
- **No placeholders:** every code/test step shows complete code and exact commands with expected output.
