# PDF-text-layer bbox refinement — design

**Date:** 2026-06-01
**Status:** approved (brainstorming → ready for plan)
**Author:** Smart Pointer team

## Problem

`python-pptx` gives the parser each shape's *container* box, not where the glyphs
actually land inside it. When text is centered, top/middle/bottom-anchored,
wrapped, or autofit, the rendered text occupies only part of the shape. The
current bullet code works around this by *estimating* line heights
(`font size × 1.2`) — those estimates are the source of imprecise highlights.

We want **glyph-exact text locations** for the highlight overlay, plus per-line
sub-boxes, without disturbing the parser, matcher, viewer, or enrichment.

## Key insight

The render pipeline already converts `pptx → PDF` (via `soffice`) before
rasterizing to PNG, then throws the PDF away. That PDF carries a real **text
layer** with exact per-word coordinates. `pdfplumber` (already installed) reads
it. No OCR, no new heavy dependencies, fully deterministic and offline.

Coordinates align directly with the parser's existing model:

- PDF page for this deck is `960 × 540 pt` = `13.333″ × 7.5″` at 72 pt/inch.
- PDF points ÷ 72 = inches, **same top-left origin and same units** the parser
  already emits in `bbox`.
- To stay robust against any soffice page scaling, derive the scale per page
  defensively: `scale = slide_width_in / (page_width_pt / 72)` (≈ 1.0 here)
  rather than hardcoding 72.

## Approach (chosen)

Keep the `python-pptx` parser exactly as-is (it is good at semantic structure,
element typing, trigger phrases, tables, charts). Add a **post-parse refinement
pass** that snaps each text element's `bbox` to the real glyph positions from the
PDF text layer, and emits per-line sub-boxes.

Rejected alternatives:

- **Full vision pipeline** (PaddleOCR → YOLO → chart detector → GPT/Gemini):
  solves problems we don't have yet, breaks the offline/deterministic design,
  cloud cost. Kept as a *separate optional future track* only if text-in-images
  or a real relationship graph is later needed.
- **OCR on the rendered PNG** (PaddleOCR/Tesseract): pixel-ish with recognition
  errors, heavy, non-deterministic. Only *needed* for text baked into images —
  out of scope for this change.

## Components

### 1. `src/render.py` — cache the PDF (small refactor)

Extract a cached `deck_pdf(pptx_path, out_dir) -> Path` that produces and caches
`out/<stem>.pdf`, keyed on pptx mtime (same freshness pattern already used for
the PNGs). `render_deck` calls it instead of using a throwaway tempdir; PNG
output is byte-for-byte unchanged. `refine_text` reuses `deck_pdf`.

### 2. `src/refine_text.py` — new transform module

**Pure, unit-testable core** (operates on plain word dicts; no soffice needed):

- `words_to_inches(words, scale)` → convert pdfplumber word boxes
  `(text, x0, top, x1, bottom)` in points to inch-space boxes.
- `assign_words(shape_bbox, words)` → words whose **center** falls inside the
  element's *original shape box*. When a word's center falls inside more than one
  element box, it goes to the **smallest-area** containing element (handles
  merged `diagram_label` / `stat` boxes that overlap their text).
- `cluster_lines(words, y_tol)` → group assigned words into lines by
  vertical-span overlap; sort lines top→bottom, words within a line left→right.
- `union_bbox(boxes)` → tight enclosing box of a set of boxes.

**Refinement rule**, applied per slide:

- **Refine** every element with non-empty `text` and `needs_vision == false`,
  **except** types in `{chart, chart_category, image, picture, arrow}` (visual,
  or carefully tuned by `chart_geom` — left untouched to protect the per-bar /
  per-slice / per-point work).
- For each refined element:
  - `bbox` ← union of its line boxes (tight glyph box).
  - `line_bboxes` ← ordered list of per-line boxes (units: inches).
  - `bbox_source` ← `"pdf_text"`.
  - `shape_bbox` ← the original python-pptx box (preserved for fallback/debug).
- **No-match fallback:** if no PDF words land inside the shape box (e.g. text
  baked into an image), leave `bbox` unchanged, set `line_bboxes = []` and
  `bbox_source = "shape"`. Nothing regresses.

### 3. CLI / pipeline placement

Standalone transform matching the existing `enrich_*` convention, so the Stage 1
parser stays pure (no soffice dependency at parse time):

```
python3 -m src.refine_text                 # out/index.json → out/index.refined.json
python3 -m src.refine_text --input out/index.enriched.json --out out/index.refined.json
```

Refinement (touches `bbox` / `line_bboxes`) and enrichment (touches
`trigger_phrases`) are orthogonal and compose in either order. A `--refine`
convenience flag on `index_deck` is optional; the standalone module is primary.

## Schema additions

Each refined element gains:

```json
{
  "bbox":        { "x": ..., "y": ..., "w": ..., "h": ..., "units": "inches" },
  "shape_bbox":  { "x": ..., "y": ..., "w": ..., "h": ..., "units": "inches" },
  "line_bboxes": [ { "x": ..., "y": ..., "w": ..., "h": ..., "units": "inches" }, ... ],
  "bbox_source": "pdf_text" | "shape"
}
```

`bbox` now holds the tight glyph box (was the shape box). Existing consumers that
read `bbox` keep working and immediately get a tighter highlight. New fields are
additive; no existing field is removed or renamed.

## Scope boundary

In scope: **producing** precise element `bbox` + `line_bboxes` in the index, and
the tighter element-level highlight the viewer gets for free.

Out of scope (noted follow-up): **highlighting an individual line** — having the
matcher decide *which* line is being spoken and the viewer draw that single
`line_bboxes` entry. That needs matcher + viewer changes and is a separate task.

## Testing

- **Unit (no soffice):** `words_to_inches`, `assign_words` containment, the
  smallest-box overlap tiebreak, `cluster_lines`, `union_bbox` — all on synthetic
  word dicts.
- **Integration (gated on soffice availability):** run on `test_deck.pptx` and
  assert:
  - slide-2 bullet element's refined `bbox` is **strictly tighter** than its
    `shape_bbox` (smaller area / shorter right edge);
  - `len(line_bboxes)` == the rendered bullet-line count;
  - every line box ⊆ the original `shape_bbox`;
  - all refined boxes sit within slide bounds.
- **Regression:** the existing 126 tests stay green. Only a new module and a new
  output file are added; the sole edit to existing code is the internal
  `render.py` PDF-caching refactor, which must keep identical PNG output.

## Files

- `src/render.py` — add cached `deck_pdf(...)`; refactor `render_deck` to use it.
- `src/refine_text.py` — new pure core + thin CLI.
- `tests/test_refine_text.py` — unit + gated integration tests.
- `out/test_deck.pdf` — new cached artifact.
- `out/index.refined.json` — new output artifact.
