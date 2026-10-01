# Multi-deck benchmark harness (`deck_bench`) — design

**Date:** 2026-10-01 · **Branch:** `feat/deck-bench`
**Program:** step 1 of 5 (1 harness → 2 OCR: PaddleOCR vs Tesseract → 3 layout:
LayoutLM vs pptx → 4 vision: Florence-2 / SmolVLM / CLIP / ViT baseline → 5 deploy
winners). Steps 2–5 get their own specs; this harness is what they are measured on.
**Hardware rule (2026-10-01):** heavy models may use the GPU at index-build time;
the live grounding loop stays CPU-only and deterministic. Offline only.

## Goal

Score live grounding (recall / precision / flicker) on any number of user decks,
not just `test_deck.pptx`, with zero manual labelling, fast enough to run many
decks × many variants.

## Decisions

- User decks go in `decks/*.pptx` (git-ignored — may be private).
- Narration is **auto-generated** from each deck's own text, synthesized with the
  existing Piper voice; the narrated order is the expected highlight order.
  Known limit: verbatim narration is easier than real speech, so absolute scores
  run high; comparisons between variants on the same narration stay fair.
- Eval runs **unpaced** on an audio-time clock (see §3) — verified identical to the
  real-time audit on test_deck.

## 1. Pipeline per deck

`python3 -m tools.deck_bench [--decks decks] [--variant default] [--only NAME]`

For each `decks/<name>.pptx`, into `out/bench/<name>/`:

1. **Truth parse** — `parse_deck(deck, ocr=False)`: the XML-only parse (plus alt
   text, which the parser already reads). Narration and expected order come ONLY
   from this, never from the index under test — otherwise an OCR/VLM hallucination
   would be narrated and scored as a hit (circularity).
2. **Narration** — `narrate(truth_index)` → `narration.json` + `expected.json`.
3. **Audio** — synthesize per slide into `audio/` (see §4).
4. **Index under test** — `build_index(deck, out/bench/<name>/<variant>/, **VARIANTS[variant])`.
5. **Eval** — grounding audit of the index under test against the narration audio
   and `expected.json`.

**Id invariant:** element ids are positional (`s{n}_e{i}`), and OCR / future
vision steps only fill text on *existing* image elements, so the truth parse and
the index under test share the same id set. A test asserts this for
`ocr=False` vs `ocr=True` on every corpus deck.

**Variants:** a module-level `VARIANTS = {"default": {}, "no-ocr": {"ocr": False}}`
mapping a name to `build_index` kwargs. `build_index` gains an `ocr: bool = True`
kwarg (it currently hardcodes `ocr=True`). Later steps add entries; no registry.

## 2. Narration generator — `tools/narrate.py`

Pure function `narrate(index) -> (script_slides, expected_slides)`.

- **Allowlist** of narrated types: `title, bullet, body, caption, stat,
  diagram_label, table_cell, chart_category`. Everything else (`chart` container,
  `arrow`, `image`, any `needs_vision` element) is not narrated in step 1.
  An `image` with non-empty alt text IS narrated (its text is the alt text).
- **Reading order:** band elements into rows by bbox top `y` (a new band starts
  when `y` exceeds the band's first `y` by > 0.3 in), sort bands by `y`, sort
  within a band by `x`. The title (if any) always goes first. `chart_category`
  elements keep their index order, emitted at their parent chart's position.
- **Phrasing:** text verbatim, whitespace-collapsed, truncated to the first 15
  words. `chart_category`: `"{category} is {value}"` (value formatted `%g`, omitted
  when `None`); multi-series (`series` set, text `"Cat — Series"`):
  `"{series} {category} is {value}"`.
- **Duplicates:** elements on the same slide whose normalized text is identical
  are narrated once; that expected step lists all their ids as alternates.
- **Empty slides:** a slide with no narratable element is omitted from both outputs
  and counted as `skipped_slides` in the report.
- Output formats match the existing ones: script `{"voice", "slides": [{"slide",
  "segments": [str, ...]}]}`; expected `{"slides": [{"slide", "expected": [[id,
  ...], ...]}]}` (same shape as `tests/fixtures/expected_highlights.json`).

## 3. Unpaced eval on an audio-time clock

`Partial.timestamp` defaults to `time.monotonic()` and `RollingBuffer` prunes its
4 s window by it, so the trajectory depends on wall-clock pacing; that is why the
audit currently plays each WAV in real time.

- `SherpaEngine(..., audio_clock=False)`: when `True`, `transcribe_stream` stamps
  each `Partial` with seconds of audio consumed (`len(chunk)/2/sample_rate`
  accumulated). Default `False` — the live viewer is untouched.
- `tools/timeline_audit.py`: `record_trajectory(..., fast=False)`; `fast=True`
  uses `FileSource(wav, chunk_ms=100)` (same chunking as `PlaybackSource`) and
  requires an audio-clock engine.
- **Verified (2026-10-01):** test_deck with audio clock + unpaced FileSource gives
  byte-identical per-slide trajectories to the real-time baseline (38/40, 1 false,
  flicker 2), twice; wall time 14.8 s vs ~2 min.
- `timeline_audit.run()` gains `manifest_path` / `fixture_path` parameters
  (defaults = today's `out/narration/manifest.json` and the test_deck fixture, so
  `tests/test_grounding_eval.py` is unchanged), and a manifest slide missing from
  the fixture is a counted skip, not a `KeyError`.

## 4. Audio synthesis

- Each narration segment is synthesized separately and joined with an explicit
  silence gap `GAP_MS = 900` (recorded in the report): Piper adds no reliable pause
  between sentences, and without a gap consecutive elements share one 4 s tail,
  which would measure window size instead of matching.
- New `PiperTTS.synthesize_segments(segments, wav_path, gap_ms)` in `src/tts.py`.
- Cache: WAV filename = `slide_{n:03d}_{sha1(segments + gap_ms + voice)[:10]}.wav`,
  so a changed narration rule can never replay stale audio. Voice path is written
  **absolute** into generated scripts.

## 5. Report

`out/bench/report.md` + `out/bench/report.json`, per variant:

- per deck: recall, precision, false highlights, flicker, slides evaluated /
  skipped; status `OK` / `FAILED: <reason>` (a failing deck never aborts the run);
- per deck **inventory** (from the truth parse): element counts by type,
  `needs_vision` count, images with/without alt text, image-only slides (a slide
  whose only elements are images) — this sizes steps 2–4;
- wall time per phase (build index / eval) per deck, so later comparisons carry cost;
- totals across decks (summed hits/steps, false, flicker).

## 6. Testing

- `tests/test_narrate.py`: allowlist, banded reading order on a two-column fixture,
  title first, chart phrasing single/multi-series, duplicate alternates, 15-word
  truncation, empty slide omitted, alt-text image narrated.
- Id-invariant test over the corpus (`ocr=False` vs `ocr=True` id sets equal).
- `timeline_audit`: missing-fixture slide is skipped, not a crash.
- Slow (opt-in `SMARTPPT_RUN_BENCH=1`) end-to-end: bench two corpus decks into a
  tmp dir, assert `report.json` has both decks with status `OK`.
- Regression: `SMARTPPT_RUN_EVAL=1 tests/test_grounding_eval.py` still passes
  (real-time path, default paths).

## Out of scope

OCR / layout / vision comparisons (steps 2–4); paraphrased or recorded narration;
parallel deck execution (unneeded at ~15 s/deck); changing the live viewer's clock.
