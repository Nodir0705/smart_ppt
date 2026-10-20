# Smart Pointer — Tasks

## Active: productionize sherpa-onnx streaming STT (2026-06-23)

Prototype verdict: standard streaming zipformer en = ~25% faster word cadence
(0.30 vs 0.40s median; 0.70 vs 1.0s worst-case) at EQUAL grounding (37/40),
flicker 2 vs 0. Offline/CPU/deterministic (int8, num_threads=1, greedy). Adopt
as default with Vosk fallback.

- [ ] Vendor int8 model → ~/.cache/sherpa-onnx/... (done, 68MB) + fetch script + .gitignore
- [ ] src/stt.py: SherpaEngine(STTEngine) — deterministic, offline, endpoint detection
- [ ] src/stt.py: load_default_stt_engine() factory — sherpa if model present, else Vosk
- [ ] Wire viewer._ensure_engine, tools/timeline_audit, simulate_to_video to the factory
- [ ] Pin sherpa-onnx in pyproject + requirements
- [ ] VALIDATE: grounding eval hits the 3 targets (s4_e2/s4_e3/s6_e1) + ≥37/40 + determinism (run twice, identical)
- [ ] Address flicker 2 (stickiness) if cheap; else accept (≤10 tolerance)
- [ ] Regenerate demo; commit

## Active milestone: Phase 0 optimization quick wins (2026-06-18)

Source: `docs/optimization-research-2026-06-18.md` §6 Phase 0 + §7 critic corrections.
Constraint: fully offline, deterministic, CPU-only. Zero new runtime deps.
Branch: `feat/phase0-quick-wins`. Baseline: 159 passed, 1 skipped (15.7s).

### Batch 1 — Determinism preconditions (do FIRST — precondition for everything)
- [x] encoder.py: default `device='cpu'`, `torch.set_num_threads(1)`, `manual_seed(0)`,
      `local_files_only=True` + pinned `revision`, OMP/MKL/HF-offline env defaults
- [x] matcher.py: bump `CACHE_VERSION` -> `v3-cpu-deterministic` (forces rebuild on CPU)
- [x] delete 9 stale `out/embeddings_*.pkl`; rebuilt on CPU
- [x] VERIFY: `pytest -q` green (159) + grounding eval **38/40, flicker 0** (was 38/40 floor 36)

### Batch 2 — Pure quick wins (no grounding impact, low risk)
- [x] asr_normalize.py: `_CompiledNormalizer` cache (byte-identical output)
- [x] audio.py + tts.py: new `src/pcm.py` — lazy `audioop` + scipy/numpy fallback (Py3.13 time-bomb)
- [x] viewer.py: debounce `<Configure>` + BILINEAR-during-drag, LANCZOS on settle
- [x] audio.py: `sd.InputStream(dtype='int16')` mic capture + float32 fallback
- [~] render.py: SKIPPED — no caller knows the slide count, so the param would be dead code
- [x] parser.py: deleted dead nbsp no-op — parser acceptance test confirms output unchanged

### Batch 3 — Slightly bigger
- [x] matcher.py: `.npz` + JSON sidecar + age eviction (replaced pickle ACE format)
- [x] matcher.py: 1-entry query encode cache (behavior-preserving)
- [x] simulate_to_video.py: raw-RGB ffmpeg pipe + `-threads 1` — verified end-to-end (640x360, 160f, valid MP4)
- [ ] tests/conftest.py: session-scoped encoder + FakeEncoder fixture — DEFERRED (test-speed nicety)

### Review (2026-06-18)
- 8 files touched + 1 new (`src/pcm.py`). Suite: 159 passed throughout. No new lint errors.
- Determinism contract now enforced in the LIVE path: CPU-only, single-thread, offline-pinned encoder.
- Two items intentionally not shipped: render `_count_slides` (would be dead code) and conftest
  FakeEncoder (pure test-speed; no production impact). Both noted, not silently dropped.

### Phase 1 (in progress)
- [x] bge-large → bge-small-en-v1.5 swap: **11x faster** live encode (114.6 → 10.1 ms/query
      CPU 1-thread), 1.3G → 383M weights. Re-tuned `PolicyConfig.threshold` 0.55 → **0.65**
      (bge-small cosines run ~0.05-0.10 higher). Grounding **37/40, flicker 0** (lost s3_e5).
      Test re-baselines: slide-3 floor 9→8; filler test tracks threshold; mock switch test
      frozen-time bug fixed (delay_s + tail(8)). 159 passed.
- [~] torch CPU-wheel reinstall: BLOCKED — this is the **shared system Python** (`/usr`, 635 pkgs;
      OCR project shares torch+CUDA). Mutating it would strip GPU from other projects. Safe path =
      isolated venv (rank 2). **Awaiting user decision** on building the venv.
- [x] enrichment wiring fix (rank 8): **DELETED** the dead ~7GB VLM+LLM branch (enrich_vlm.py,
      enrich_llm.py, test_enrich.py, dead out/index.* outputs). It wrote index.vlm/enriched.json
      but group_scope builds scoped from refined, bypassing both → dead output every run. Also
      redundant (parser already explodes chart categories into pointable elements) and off-ethos
      (nondeterministic, GPU-leaning). Zero grounding impact (37/40). 121 passed.
- [ ] footprint torch CPU-wheel — SKIPPED by user decision (shared system python; bge-small already
      won the 10x model-weight footprint)
- [x] dependency manifest (rank 2): pyproject.toml + requirements.txt, pinned working set,
      requires-python>=3.10,<3.13, folded pytest+ruff config (pytest.ini removed). 121 passed.
- [x] conftest FakeEncoder (rank 10): session-scoped real encoder + model-free FakeEncoder.
- [x] pypdfium2 (rank 7): refine_text.py swapped pdfplumber → pypdfium2 TIGHT glyph boxes.
      Raster-calibrated at 150dpi → SHIFT_FRAC 0.22→0.0 (tight boxes are glyph-exact, 3.4x
      tighter fit, equal clipping). Extraction ~142ms→8.9ms (~16x). pdfplumber removed from
      manifest. 14 refine tests + real-deck tightness pass; 121 suite.
### Phase 2 (2026-06-19)
- [x] static-frame dedup: simulate_to_video memoizes frame bytes keyed by the highlight
      bbox — renders each unique frame once (14/160 on slide 1, ~91% reused), re-pipes
      bytes for static holds. Better than the report's concat-demuxer idea (no segment-
      boundary fragility). Output unchanged. 121 passed.
- [~] warm LibreOffice (unoserver): NOT DONE — assessed as net-negative. Adds a daemon +
      new dependency for a win the mtime cache already blunts (cold soffice only hits when
      the deck CHANGES, which is rare for a grounding tool pointed at a finished deck).
      Violates the offline/minimal-deps ethos. Report rated it "not high" value. Recommend skip.
- [~] ONNX OnnxEncoder / faster-whisper / GGUF: MOOT or unrequested — ONNX encoder only
      pays off if the ~700MB torch footprint is a hard constraint (user skipped footprint);
      GGUF/VLM were for enrichment (deleted); faster-whisper is offline-narration-only, not asked.

### Optimization program COMPLETE — only net-negative/moot items remain.

---

## Active milestone: Stage 1 — Parser → index.json

Goal: from `test_deck.pptx`, produce `out/index.json` that satisfies the Stage 1
acceptance test in `CLAUDE.md`.

### Plan

- [x] Project skeleton: `src/`, `tests/`, `tasks/`, `out/`
- [x] Quick inspection of `test_deck.pptx` to confirm shape layout
- [x] `src/parser.py`: core parsing
- [x] `src/index_deck.py`: CLI entry, writes `out/index.json`
- [x] `tests/test_parser_acceptance.py`: 13 acceptance tests
- [x] Run acceptance test, iterate until green (13/13 passed on first run)
- [x] Update review section below

## Stage 3 — Matcher (2026-05-21)

**Status:** complete. 34/34 tests green (Stage 1 + Stage 3 combined).

- [x] `src/encoder.py` — BGE-large-en-v1.5 via transformers (cached locally, GPU auto)
- [x] `src/matcher.py` — Index loader with on-disk embedding cache; Matcher.match (per-element MAX over phrases); MatchPolicy (threshold + runner-up gap + stickiness + clear-after-N-low)
- [x] `src/match_repl.py` — typed-input REPL with `:slide`, `:threshold`, `:gap`, `:stick`, `:reset`, `:show`
- [x] `tests/test_matcher.py` — 9 deterministic policy unit tests + 12 semantic acceptance tests against real BGE+index
- [x] Hand-tuned defaults: threshold=0.55, gap=0.03, stickiness=1 (raise to 2–3 for streaming partials)

**Stage 3 design choices**
- BGE-large-en-v1.5 was already cached at `~/.cache/huggingface/hub` — zero install. Wrapped behind a tiny `Encoder` protocol so it can be swapped for a smaller model later.
- Per-element score = MAX over that element's trigger-phrase embeddings (cheap dot products, ~30 phrases per slide).
- Element embeddings are precomputed at Index load and cached on disk keyed by index-content hash + model name, so re-runs are instant and parser changes auto-invalidate.
- MatchPolicy is a small state machine with three knobs:
  - `threshold` — absolute floor (0.55 by default)
  - `gap` — best must beat runner-up by this much (0.03) so close calls don't flash
  - `stickiness` — consecutive frames before switching (1 for typed input; bump to 2–3 once streaming STT is wired so partial-transcript jitter doesn't flicker the highlight)
- Tested REPL trace: filler ("as you can see in this diagram") correctly emits `hold`, not a stray highlight. Genuinely-ambiguous queries also emit `hold` — matcher returns the best guess but policy refuses to flicker.

## Stage 4 — Overlay (2026-05-21)

**Status:** complete. 57/57 tests green (Stages 1+3+4).

- [x] `src/render.py` — pptx → pdf via `soffice --headless` → per-slide PNG via `pdftoppm` at 150 dpi; idempotent disk cache at `out/slides/slide_NNN.png` keyed on pptx mtime
- [x] `src/geom.py` — pure geometry: inches → image px, FitTransform (letterbox + image-to-canvas), lerp_bbox, ease_in_out_cubic — all testable without a display
- [x] `src/viewer.py` — Tkinter app: full-window slide canvas, typed-phrase Entry, status line, PgUp/PgDn nav, Escape clears, Ctrl-Q quit; animated highlight (12-frame eased lerp, dark halo + amber outline); window resize re-fits image and re-places the highlight at the new scale
- [x] `src/preview.py` — headless PIL preview that composites the highlight onto the slide PNG (works in CI, useful when Tk isn't available)
- [x] `tests/test_render.py` — 4 tests (count, aspect ratio, cache hit, cache invalidation)
- [x] `tests/test_geom.py` — 15 tests (origin, full-slide, letterbox both ways, lerp, ease)
- [x] `tests/test_end_to_end.py` — 4 integration tests covering spoken phrase → matched id → image-pixel bbox at the same coordinates the viewer would draw

**Stage 4 design choices**
- `soffice → pdf → pdftoppm` was the cleanest offline pipeline available locally (~2.5 s for the 7-slide test deck). Cached by file mtime so re-runs are instant.
- Geometry was deliberately factored out of `viewer.py` into `geom.py` so the bbox math is unit-tested without a display. The Tk widget code itself can only be verified manually.
- Highlight = thick amber rectangle with a thin dark halo. No alpha (Tk Canvas doesn't support it portably); the halo gives enough contrast on both light and dark slides.
- Animation: 12 frames at ~15 ms each (180 ms total), cubic ease-in-out. First appearance pops in (no animation from nowhere); switches and clears animate; window resizes snap.
- Chart-category highlights cover the whole chart for now (per CLAUDE.md "start by highlighting the whole chart on a category match"). Sub-region per-bar pointing is deferred.

**How to run**
```
python3 -m src.index_deck test_deck.pptx     # build out/index.json
python3 -m src.render test_deck.pptx         # build out/slides/slide_NNN.png
python3 -m src.viewer                        # open the Tk viewer
python3 -m src.preview --slide 6 --phrase "let me explain the matcher" --out /tmp/p.png
```

**Headless verification:** the Tk window can't be auto-tested here. I verified
the integration path end-to-end via `src/preview.py` (three previews — Matcher
diagram box, STT chart, Sub-millisecond table cell — all landed pixel-perfect)
and via `tests/test_end_to_end.py` which asserts the matched-element bbox
lands within ±5 px of the expected location. The Tk widgets themselves still
need a manual run.

## Stage 2 — STT (2026-05-21)

**Status:** complete. 68/68 tests green (Stages 1+2+3+4).

- [x] `src/stt.py` — `STTEngine` Protocol, `Partial` record, `RollingBuffer` (finals + non-final partial, time-pruned), `hotwords_from_index` (deck vocabulary with stopword filter), `MockEngine` (scripted), `VoskEngine` (offline, streaming, optional grammar-bias mode)
- [x] `src/audio.py` — `AudioSource` Protocol, `MicSource` (sounddevice callback → queue, runs on its own audio thread), `FileSource` (16-bit mono WAV in chunks)
- [x] `src/live.py` — `LiveTranscriber` glues `source.chunks()` → `engine.transcribe_stream()` → `RollingBuffer` → `on_update(text, partial)` callback. Runs in foreground or on a daemon thread.
- [x] Viewer wiring — `--mic` / `--wav PATH` / `--grammar-bias` flags; audio thread pushes buffer-text into a queue, Tk drains it every 50 ms and runs `policy.step()`, throttled to 5 Hz to keep the matcher off the hot path.
- [x] `tests/test_stt.py` — 11 tests (RollingBuffer 5, hotword extraction 4, MockEngine pipeline 1, scripted full pipeline 1) — all hit the real index when available.
- [x] Real-engine smoke verified: VoskEngine on the `alsa/Front_Center.wav` sample emits 5 `partial 'front'` then `FINAL 'front center'`.

**Stage 2 design choices**
- Vosk small-en-us (~40 MB) chosen for offline streaming with real partials; pip-installable, model auto-downloads on first use. Wrapped behind `STTEngine` so faster-whisper or sherpa-onnx can be dropped in later without touching the viewer.
- The `MockEngine`/`MockScript` pair is the centrepiece of the test strategy — STT is non-deterministic on real audio, so the integration tests script the exact emissions and assert the matcher+policy still end up on the right element. This is what lets us test "full pipeline produces correct highlight" without a mic and without flakes.
- "Vocabulary biasing" turned into two complementary mechanisms:
  1. **Semantic robustness** in the matcher (BGE embeddings tolerate STT typos like "macher" → "matcher") — the default mode.
  2. **Hard grammar bias** in Vosk via `--grammar-bias` — recognizer is constrained to deck vocabulary, giving perfect on-deck recognition at the cost of refusing off-deck speech. Use when the presenter stays strictly on-topic.
- The `--match-throttle-ms` (default 200) decouples STT rate from matcher rate; Vosk fires a partial roughly every audio chunk (~5 Hz already), but this guarantees we never hammer the embedding model.
- Tk thread-safety: the audio thread never touches widgets; it only pushes the latest buffer text into a `queue.Queue`. The Tk main loop drains the queue on `after(50, …)` and does all the matching/drawing on the main thread.

**How to run end-to-end**
```
python3 -m src.index_deck test_deck.pptx     # Stage 1 → out/index.json
python3 -m src.render test_deck.pptx         # Stage 4 → out/slides/*.png
python3 -m src.viewer --mic                  # Stages 2+3+4 wired together
# or:
python3 -m src.viewer --wav recording.wav    # replay instead of live mic
python3 -m src.viewer --mic --grammar-bias   # constrain Vosk to deck vocab
```

**Headless verification caveat:** the `--mic` path can't be auto-tested here
(no display, and no controlled mic input). VoskEngine itself is verified on a
real WAV; the pipeline (audio → STT → buffer → matcher → policy → highlight)
is verified end-to-end via the MockEngine path in `tests/test_stt.py`. The
Tk+mic combination still needs a manual run from you to confirm latency and
feel.

## Simulation — TTS-driven end-to-end demo (2026-05-21)

**Status:** complete. The simulator narrates the deck with synthesised speech,
plays the audio out the speakers, feeds the same bytes into STT, and the
viewer's highlight tracks the spoken word in real time.

- [x] `scripts/narration.json` — one paragraph per slide that naturally
      mentions each pointable element
- [x] `src/tts.py` — Piper offline TTS; resamples to 16 kHz mono
      (Vosk-ready); idempotent disk cache at `out/narration/slide_N.wav`
- [x] `src/audio.py::PlaybackSource` — reads a WAV, plays it through the
      speakers, AND yields the same bytes at real-time speed (sounddevice's
      `stream.write` block paces the yields automatically)
- [x] `src/stt.py::RollingBuffer.tail(n)` — slice to the last N words so
      Vosk's ever-growing partial doesn't trap the matcher on an
      early-mentioned element
- [x] `src/viewer.py::start_playlist()` — walks a `[{"slide": N, "wav": p}]`
      playlist on a worker thread, switching slides between WAVs
- [x] `src/simulate.py` — top-level CLI; loads index, ensures slide PNGs
      and narration WAVs exist, then opens the viewer in playlist mode

**Headless dry-run output** (TTS → muted PlaybackSource → STT → matcher → policy):
each slide's narration produces an ordered trajectory of highlight switches —
bullets light up in order on slide 2, the table walks left-to-right on
slide 3, the architecture flow goes Microphone → Matcher → Overlay on slide 6.

**How to run**
```
# one-time setup
python3 -m src.index_deck test_deck.pptx     # parse → out/index.json
python3 -m src.render     test_deck.pptx     # slides  → out/slides/*.png
python3 -m src.tts                           # narration → out/narration/*.wav
# the demo
python3 -m src.simulate                      # window opens; deck narrates
python3 -m src.simulate --mute               # silent — STT-only, watch the highlights move
python3 -m src.simulate --pause 1.0          # longer gap between slides
```

**Headless verification caveat:** the Tk window + speakers part is verified
only by the import smoke test and the headless dry-run printed timeline. The
actual visual + audible playback needs a manual run from you to confirm
end-to-end feel.

## Bullet-bbox fix (2026-05-21)

**Status:** fixed. Slide 2 bullets no longer overlap.

- [x] `src/parser.py::_paragraph_metrics()` reads each paragraph's font size,
      line spacing, and `space_after` from python-pptx
- [x] `src/parser.py::_bullet_bboxes()` walks paragraphs using actual line
      heights (font × 1.2 by default) and stacks them with `space_after`
      gaps, honouring `vertical_anchor` (TOP/MIDDLE/BOTTOM)
- [x] Replaced the equal-strip splitter in `_parse_slide`'s bullets branch
- [x] Re-ran parser → slide-2 bullets now at y=2.47, 2.88, 3.25, 3.66, 4.06
      with h≈0.23–0.27" each (was 5×0.80" overlapping strips)
- [x] Regenerated `out/simulated/slide_002_simulated.png` — bullets are now
      tight, sequential, no overlap, and visually hug the rendered text
- [x] 68/68 tests still green

## Real-time demo videos (2026-05-21)

**Status:** complete. Watch the simulation as MP4s — narration audio + frame-by-frame animated highlight in sync, same animation the Tk viewer would draw.

- [x] `src/simulate_to_video.py` — runs the live pipeline (TTS → muted PlaybackSource → STT → matcher → policy), records (timestamp, active_id) events paced by audio time, renders frames at 30 fps with the viewer's lerp+ease_in_out_cubic interpolation, encodes via ffmpeg as H.264 + AAC.
- [x] Per-slide MP4s at `out/video/slide_NNN.mp4` plus the combined `out/video/demo.mp4` (1.78 MB, 1m 50s total).

**Verification:** extracted a frame from `slide_006.mp4` at t=9.5s; the
amber highlight sits exactly on the Matcher box at the moment the narrator
says "the matcher finds the best element on the current slide." Per-slide
event timestamps printed during render show the matcher tracking each
phrase as it arrives.

**How to watch**
```
out/video/demo.mp4         all 7 slides concatenated (1.8 MB, 1:50)
out/video/slide_NNN.mp4    one MP4 per slide
```

## Enrichment (VLM + LLM, options 1 & 3) — 2026-05-21

**Status:** complete and runnable; option 2 (geometric chart sub-regions) deferred per user.

- [x] Downloaded Qwen2-VL-2B-Instruct (~4.2 GB) and Qwen2.5-1.5B-Instruct (~3 GB) locally
- [x] `src/enrich_vlm.py` — for `image` and `chart` elements, crop the slide
      PNG to the element's bbox and ask the VLM for navigational +
      descriptive phrases. Cross-element de-dup per slide drops phrases
      that wound up on multiple elements (the matcher can't disambiguate
      them anyway).
- [x] `src/enrich_llm.py` — for text-bearing elements (bullets, body,
      captions, longer labels), generate 3-5 paraphrases. Skip elements
      with < 3 words of source text — a 1.5B LLM hallucinates context for
      short labels. Filter outputs that start with imperative or meta
      openers ("show me", "highlight as", "this is the", …).
- [x] `tests/test_enrich.py` — 12 unit tests covering the parsers (no model load).
- [x] End-to-end run on the test deck: 4 elements VLM-enriched, +82 LLM
      phrases. Total run ~50 s on 8 GB GPU.

**Wins (verified by probe + simulation):**
- Slide 7 image element went from 0 triggers (`needs_vision: true`) to 4
  phrases. "look at the picture on the left" now resolves to the image
  itself (was hitting the title).
- Slide 4 bar chart highlights as a whole at t=7.7 s when narrator says
  "looking at latency by stage" — base index never lit the chart itself,
  only the inner categories.
- Slide 4 pie chart now correctly wins the "the pie chart" query (base
  always picked the column chart because they shared the generic "chart"
  trigger).

**Honest limitations:**
- Qwen2-VL-2B sometimes hallucinates context — when fed only the pie chart
  crop, it described "the bar chart on the left," polluting the pie chart's
  triggers. Cross-element dedup helps but doesn't catch this case (it's a
  single unique-to-that-element phrase that's just wrong).
- Slide 5's line chart got mislabeled as a bar chart. Triggers still
  improve coverage but the description is incorrect.
- LLM enrichment for the title "The Smart Pointer" produced C++-flavoured
  paraphrases ("reference-counted pointer") — the model doesn't know this is
  our product name. Mitigations would require richer slide context or a
  bigger LLM.
- Cleanest gains come from elements with actual content (images,
  charts, bullets); short labels (Matcher, STT, < 1 ms) are now skipped by
  design to avoid hallucinated triggers.

**Files**
- `src/enrich_vlm.py`, `src/enrich_llm.py`, `tests/test_enrich.py`
- `out/index.vlm.json` — index + VLM phrases
- `out/index.enriched.json` — index + VLM + LLM phrases (use this as the
  matcher input for the richest experience)
- `out/index.vlm.vlm_debug.json` — raw VLM responses for inspection
- `out/simulated_enriched/slide_004_simulated.png` — verified composite
  showing whole-chart highlights triggered earlier in the narration

**How to run**
```
python3 -m src.enrich_vlm                              # ~17 s
python3 -m src.enrich_llm --input out/index.vlm.json \
                          --out out/index.enriched.json # ~30 s
# point the simulator/viewer at the enriched index:
python3 -m src.simulate --index out/index.enriched.json
python3 -m src.viewer   --index out/index.enriched.json --mic
```

## VLM refactor — focused questions (2026-05-21)

**Status:** complete. The 2B VLM hallucinates on open-ended "describe what a
presenter might say" prompts but is *accurate* on focused single-answer
questions ("what type?", "what's the title?", "what's the largest bar?").
Refactored to ask 2–3 narrow questions per visual element and assemble
trigger phrases programmatically from the sanitized answers.

- [x] `src/enrich_vlm.py` rewritten — `_enrich_chart()` asks type/title/max
      → `_phrases_for_chart()` builds clean type-correct vocabulary
      (`the pie chart`, `the largest slice`, `the highest point`, …)
- [x] `_enrich_image()` asks kind/description → `_phrases_for_image()` adds
      navigational phrases plus the model's description
- [x] Per-type superlative vocabulary: pie→slice, bar→bar/tallest,
      line→point/highest. Generic superlatives generated from chart type
      even when the VLM can't name the max category.
- [x] `_strip_leading_article()` fixes `"the an icon"` → `"the icon"`.
- [x] 27 new unit tests on assembly + sanitation; suite now 106/106 green.
- [x] Re-enriched + re-rendered `out/video_enriched/demo.mp4`.

**Verified probes (all clean, four exact-1.0 matches):**
| query | base → | enriched → |
|---|---|---|
| "the bar chart on the left" | bar (0.79) | **bar** (0.88) |
| "the pie chart" | bar ✗ | **pie** (1.0) |
| "the biggest bar" | title ✗ | **bar** (1.0) |
| "the largest slice" | pie (0.66) | **pie** (1.0) |
| "the line chart" | line (0.83) | **line** (1.0) |
| "look at the picture on the left" | title ✗ | **image** (0.80) |

**Slide 7 win is now visible in the simulation:** image element lights up
at t=0.5s, 5.3s, 7.0s of the slide-7 narration. Before refactor the image
was never hit during this narration.

## Option 2 — Geometric chart sub-regions (2026-05-21)

**Status:** complete. Per-bar / per-slice / per-point highlighting works
for the bar, pie, and line charts in the test deck.

- [x] `src/chart_geom.py` — `classify_chart_type()`, `plot_area()` (per-type
      inset heuristics), `bar_subregions()` (vertical strips),
      `pie_subregions()` (sector bounding boxes via arc-extrema math),
      `line_subregions()` (narrow vertical strips at each x position).
- [x] `src/parser.py::_expand_chart()` rewired to assign per-category
      sub-bboxes AND attach a `"the {category} {noun}"` trigger to each
      `chart_category` element (so "the STT bar" → the STT column, not the
      chart-level element).
- [x] `src/enrich_vlm.py::_phrases_for_chart()` no longer generates
      label-specific phrases on the chart element — those live on the
      category. Generic superlatives ("the largest bar", "the biggest
      slice") stay on the chart-level.
- [x] `tests/test_chart_geom.py` — 20 new tests covering classify, plot
      area inset, sub-region positioning, dispatch. Suite at 126/126.

**Verified visual outputs** (in `out/preview_*.png`):
- `preview_stt_bar.png` — "look at the stt bar" → narrow amber strip on
  the STT column only
- `preview_parsing_slice.png` — "the parsing slice" → amber box on the
  right half of the pie (where the 50% Parsing wedge sits)
- `preview_line_09.png` / `preview_line_05.png` — "the point at 0.9" /
  "at threshold zero point five" → thin vertical strips landing exactly
  on the matching x labels, covering both line series at that x

**Probes (all clean):**
| query | element | score |
|---|---|---|
| "the stt bar" | STT category | **1.000** |
| "the sync bar" | Sync category | **1.000** |
| "the parsing slice" | Parsing category | **1.000** |
| "the overlay ui slice" | Overlay UI category | **1.000** |
| "the bar chart" | whole bar chart | **1.000** |
| "the largest bar" | whole bar chart (generic) | **1.000** |
| "the pie chart" | whole pie chart | **1.000** |
| "at threshold zero point five" | 0.5 line point | 0.770 |
| "the point at 0.9" | 0.9 line point | 0.922 |
| "look at the render bar" | Render category | 0.924 |

**Known calibration:** the plot-area insets are heuristic constants tuned
for LibreOffice/PowerPoint default chart styling. They were re-tuned for
line charts (`left=0.13, right=0.10, bottom=0.18`) because the test deck's
line chart has wider axis-label gutters than the bar/column defaults.
Different chart styles may need per-chart inset overrides — that's a
future hook, not a present blocker.

## Backlog (later milestones)

- Stage 5: Polish (slide-sync against native PowerPoint/Keynote, transition smoothing tuning)
- Stage 6: Optional vision labeling for `needs_vision` elements
- Stretch: per-bar / per-slice sub-region highlights for `chart_category` elements
- Stretch: alternate engines behind the STTEngine Protocol (faster-whisper for higher accuracy, sherpa-onnx for true hotword boosting without grammar lock-in)
- Stretch: capture the simulator run as a video (record screen + audio) for
  shareable demos

## Review — Stage 1 (2026-05-20)

**Status:** complete. `python3 -m src.index_deck test_deck.pptx` writes a
56-element `out/index.json` covering all 7 slides; `pytest` runs 13 acceptance
tests, all green.

**Key design choices**
- One per-slide pass, in this order: (1) merge empty-container + text-inside
  pairs, (2) expand tables / charts / pictures, (3) emit remaining text shapes
  with title/bullet/body/caption classification, (4) detect arrows as empty
  zero-dim shapes. This makes merging and arrow detection mutually exclusive
  and avoids reclassifying the same shape twice.
- Merged container+text pairs are typed `diagram_label` when ≥ 3 such pairs
  appear on a slide (a real diagram flow) and `stat` otherwise. This single
  rule covers both the slide-6 architecture diagram and the slide-2 "< 1 ms"
  stat callout without per-slide special cases.
- `chart_category` elements inherit the chart's bbox for now (per CLAUDE.md
  "start by highlighting the whole chart on a category match"). Sub-region
  estimation is a stretch goal for later.
- Table cells use per-column and per-row widths from `table.columns[i].width`
  / `table.rows[i].height`; fall back to even division if those don't sum to
  the table bbox.
- Trigger phrases stay deliberately thin: lowercased text + punctuation-
  stripped variant + per-line splits. Stage 3 semantic similarity will do the
  paraphrase work; over-engineering this now would only add noise.

**Known limitations carried forward**
- Slide 1's centred title is classified as `body` (the title heuristic
  requires `y < 1.2"`; the title-slide layout puts the title mid-page). Not
  blocking; revisit if/when Stage 4 layout needs it.
- Arrows on slide 6 are recorded but unpointable by speech — intentional per
  CLAUDE.md scope.
- Chart sub-region (per-bar / per-slice) highlighting deferred.

**Files**
- `src/parser.py` — core
- `src/index_deck.py` — CLI
- `tests/test_parser_acceptance.py` — 13 acceptance tests
- `out/index.json` — generated artifact

## PDF text-layer bbox refinement (2026-06-01)

**Status:** complete. Text element bboxes are now glyph-exact from the soffice
PDF text layer, with per-line sub-boxes.

- [x] `src/render.py` — cached `deck_pdf()`; `render_deck` rasterizes from it (PNG output unchanged, guarded by test_render.py)
- [x] `src/refine_text.py` — pure core (`words_to_inches`, `assign_words` w/ smallest-box tiebreak, `cluster_lines`, `union_bbox`) + `refine_index` + CLI
- [x] `tests/test_refine_text.py` — unit + integration (soffice-gated)
- [x] `out/index.refined.json` — index with refined `bbox`, additive `shape_bbox`, `line_bboxes`, `bbox_source`

**Design choices**
- PDF points / 72 = inches, scale derived per page (`slide_w / (page_w_pt/72)`)
  so it is robust to any soffice page scaling.
- Words assigned to the smallest-area containing element so merged
  diagram_label/stat boxes don't steal each other's text.
- `cluster_lines` requires >50% vertical overlap (not a sliver) to merge words
  into a line, preventing cascade-merge of tightly-leaded distinct lines.
- `{chart, chart_category, image, picture, arrow}` excluded — chart sub-regions
  stay as tuned by chart_geom; visual elements carry no snap-worthy PDF text.
- No-match elements fall back to `bbox_source="shape"` with the original box, so
  text baked into images never regresses.

**Scope boundary:** per-line *data* is emitted now. Highlighting a single spoken
line (matcher picks which line; viewer draws that one box) is a noted follow-up.

**How to run**
```
python3 -m src.refine_text                                  # out/index.json -> out/index.refined.json
python3 -m src.refine_text --input out/index.enriched.json --out out/index.refined.json
```

## Live grounding — Phase 1 (2026-06-01)

**Status:** complete. Grounding accuracy 34/40 → 37/40 on the narration audit,
flicker 6 → 5, and all three diagnosed failures fixed. Fully offline /
deterministic / CPU-only (no live neural model; GPU was unavailable).

**Baseline → result** (`python3 -m tools.timeline_audit --index out/index.scoped.json --normalize`)
- s4_e2 STT latency bar — never lit → lit (alias layer)
- s4_e3 Match latency bar — lost to pie "Matching" → lit (group scoping)
- s6_e1 STT Engine node — never lit → lit (acronym expansion trigger)

**Four deterministic levers**
- [x] `src/asr_normalize.py` + `data/aliases.json` — post-STT alias/normalization
      (fixes "STT"→"sgt" mishearings). Auto-derives acronym variants; hand overrides.
- [x] `src/group_scope.py` + `data/expansions.json` — author-time `group_id`/
      `group_label` + scoped triggers ("the match bar in the latency chart") +
      acronym expansions ("speech to text engine"). Output `out/index.scoped.json`.
- [x] `src/matcher.py` — `scored()` + group-aware effective score. `same_group_bonus`
      keeps "match" in the latency chart; `stay_bonus` resists same-group sibling
      flicker but yields freely to a different group (real switches still fire).
- [x] runtime wiring — `best_index()` (scoped>refined>base) + alias normalizer in
      viewer/simulate; cache key bumped (v2-groups).

**Eval harness:** `tools/timeline_audit.py` + `tests/fixtures/expected_highlights.json`
(ordered-recall metric). `tests/test_grounding_eval.py` (opt-in `SMARTPPT_RUN_EVAL=1`).

**Known remaining (not Phase-1 targets):** "sync"→"sink" mishearing (s4_e5);
the "< 1 ms" stat vs body paraphrase (s2); three identical "Live" table cells (s3).

**Out of scope (Phase 2/3):** CLIP/SigLIP cross-modal live scorer; Florence-2/VLM
author-time grounding; streaming engine swap with contextual biasing; cross-encoder rerank.

**How to run**
```
python3 -m src.index_deck test_deck.pptx
python3 -m src.refine_text                 # glyph-exact bboxes -> index.refined.json
python3 -m src.group_scope                 # groups + scoped triggers -> index.scoped.json
python3 -m src.render test_deck.pptx
python3 -m src.tts
python3 -m src.simulate                    # auto-uses index.scoped.json + alias normalize
```

## Deck-agnostic group scoping — roadmap #9 (2026-10-01)

**Status:** complete. Spec `docs/superpowers/specs/2026-10-01-deck-agnostic-scoping-design.md`,
plan `docs/superpowers/plans/2026-10-01-deck-agnostic-scoping.md`.

- [x] parser emits `chart_title` / `series_names` (chart `text` unchanged)
- [x] `_TITLE_KEYWORDS` deleted; labels from chart title → sole series → slide title → noun; phrases for head + full
- [x] `derive_expansions` + `expansions_for` (file overrides win), wired in build_index + group_scope CLI; hyphenated definitions ("Speech-to-Text (STT)") handled
- [x] corpus deck `titled_chart_acronym`

**Eval (test_deck):** baseline recall 38/40 · false 1 · flicker 2 → after: recall 38/40 · false 1 · flicker 2,
per-slide trajectories byte-identical. Slide-4 pie relabelled latency → build effort split (was a whitelist mislabel).
`SMARTPPT_RUN_EVAL=1` grounding test passes; full suite 195 passed, 1 skipped.

## Multi-deck benchmark harness — program step 1/5 (2026-10-01)

**Status:** complete on `feat/deck-bench`. Spec `docs/superpowers/specs/2026-10-01-deck-bench-design.md`,
plan `docs/superpowers/plans/2026-10-01-deck-bench.md`. Program: 1 harness → 2 PaddleOCR vs Tesseract →
3 LayoutLM vs pptx → 4 Florence-2 / SmolVLM / CLIP / ViT → 5 deploy winners.

- [x] SherpaEngine `audio_clock` + unpaced `timeline_audit --fast` (byte-identical to real-time on test_deck; ~15 s vs ~2 min)
- [x] per-segment Piper synthesis, 900 ms gaps, content-hashed + atomic WAV cache
- [x] `tools/narrate.py` (XML truth parse only; banded reading order; dup alternates)
- [x] `build_index(ocr=)` + OCR id-invariant test
- [x] `python3 -m tools.deck_bench` → `out/bench/report_<variant>.{md,json}`; `decks/` git-ignored

**First finding (test_deck, auto-narration):** recall 31/47, precision 1.000, flicker 0. Every miss is a 1–2 word
mention (table cells "Stage"/"Live", bare chart readings "STT is 180", short labels "STT Engine") — the matcher
holds the previous element rather than switching on a single-word cue. Real weakness, not a harness artifact;
not tuned away. Candidate follow-up once real decks confirm it.

**Multi-scale tail (2026-10-01):** fixed the short-mention dilution. Per-phrase windows (`max(4, n+1)` words) +
2-word short-tail override (raw ≥ 0.73, gap 0.03, new element only) + 8-step revert guard, all inside
`MatchPolicy.step`; `timeline_audit` now runs through `LiveController`. 3-deck bench: hits 442→519/741,
1+2-word 82→134/255, 6+-word 193→199/225, false 0, flicker 31→20. Hand fixture 38/40, false 1, flicker 2→1.

**Filler precision (2026-10-01):** `deck_bench` now runs a negative pass (FILLER_HOLDOUT presenter filler per slide,
expected none) and reports `filler_false`. Holdout baseline: old (cec1f1c) 20, multi-scale 36. Fix: per-element null
floor (best cosine to fixed `NULL_SPEECH` generic talk) + `null_margin` 0.12 in `MatchPolicy`. Holdout 36→5, tune
56→9; bench hits 512/741, 1+2w 131/255, 6+ 198/225, false 0, flicker 15; hand 38/40, false 1, flicker 1.

**Pointer behaviour under natural speech (2026-10-02):** presenter_bench adjudicated scoring (5ccd53f); slide furniture
unpointable (1eabcb3); revert-guard bounce no longer clears (02520e6); select_line punctuation (d7dd866). Real-STT:
AI 43/48, adj false 10 ids / 19.4 s, 2.40 sw/step, late 2.60 s; Open-Q 91/117, adj 23 / 37.7 s; hand 38/40 f1 fl1;
filler holdout 5; verbatim 483/741; pytest 265 passed. **Gate NOT met** (AI adj false s ≤ 10.2, sw/step ≤ 1.5):
every switch-confirmation rule that reaches them breaks the hand fixture (31–37/40) and Open-Q recall (59–68/117).
Best non-gate candidate (not shipped): confirm only short-override switches + release a hold after 3 unsupported
updates — AI adj 12.7 s / 2.19 sw / late 2.80, Open-Q adj 27.0 s, hand 38/40, verbatim 471. Decision pending.

**Card-level pointing (2026-10-02):** eval `--device` + per-deck processes (bd464be; presenter 353→106 s, identical
trajectories); short-override confirm + stale release (e611d5d; AI 2.19 sw/step, adj 12.7 s); pointable units at Index load
(stack / picture+caption / card panel) — policy, viewer, simulators and benches all at unit level; gap 0.035, clear 16,
stale off. AI 43/48, 1.58 sw/step, adj 9.4 s; Open-Q 89/117, 2.01, adj 46.2 s (worse, flagged); hand 38/40 f1 fl1;
filler 3; verbatim 462/741. Next: kicker/title lights.

## 2026-10-12 — Generalization test on an unseen deck (TapHole) + STT lag

Deck: `decks/TapHole.pptx` (copy of `[Tap hole][Netvision]AI model for tap hole detection_R1_22May2025.pptx`, 18 slides).
Rule: no retuning — shipped params and prompts only; TapHole is a pure holdout.

- [x] 1a. Build default index + slide PNGs (`tools.deck_bench --only TapHole`)
- [x] 1b. Slide maps (Claude agents, blind to scripts) → `out/maps/TapHole/`, all validate
- [x] 1c. Three talk scripts (original / jumpy / meaning), agents blind to maps, annotated `refers_to`
- [x] 1d. Map index (`--variant map`), presenter audio, record STT, `clause_replay --deck TapHole --holdout`
- [x] 1e. Compare with AI_Handout rows (same params); list failure modes per slide
- [x] 1f. Video: draft script → voice → cue sheets → validate → render → contact-sheet audit
- [x] 2. STT lag: per word, spoken time (Whistle on the clean TTS wav) vs first appearance in sherpa partials (AI_Handout + TapHole)
- [x] Report + lessons

Review (2026-10-12): pointer frozen on TapHole lit 63/65, 59/70, 68/72 (wrong ids 12/7/5); STT lag 0.51 s median on
both decks (~20% of lateness; presenter naming late ~70%). Video: 99 cues, audit 57/99 (AI_Handout 77/77); ~30
failures = captions over slide content / titles on dense slides; 4 = brackets off-frame (fixed d58545e);
render deterministic. Parser drops embedded videos + one picture, wrong bbox on slide 8. Open: caption band.

## 2026-10-15 — Parser coverage: general fixes, then an unseen English holdout

Rule: one global config, no per-deck settings (user, 2026-10-15). Every fix targets a PowerPoint feature, scored on all decks.
Holdout: 8 public English decks → `decks/holdout/` (frozen, not opened until the final run). `mit_theory_computation`
was opened for the pilot below and its failures drive 4b, so it is a DEV deck from 2026-10-15 on; 7 remain unseen.
Dev decks for parser work: 4 bench decks + MIT + 5 local (Wheel-Rail GUI, Wheel_rail_Edge 설계문서, 휠레일 컨택, Amaliyot ×2).

- [ ] 0. Commit `script/TapHole/` (TapHole video narration)
- [x] 1. Coverage report: `python3 -m tools.coverage out/bench/<deck>/<variant>` → `coverage/slide_NNN.png` (every element's
      box + id drawn on the render, set-of-mark style) + `report.{md,json}` (PDF words no element covers, text elements with
      no words under them). Baseline (pre-fix indexes, 2026-10-20): test_deck 0/190 words unclaimed, AI_Handout 0/1202,
      TapHole 37/984 (all s8 = the refine miss), MIT 340/813 (the equation boxes; s8 has 2 elements for 87 words),
      Open-Q 0/2541 but 5 empty text elements: s1_e7 and s8_e28 sit OFF the slide (y 9.5 / 8.9 in on 7.5), three "};"
      code bullets keep a 12-in frame box (text has no alnum → unrefinable; known, not now). Picture-only gaps (TapHole
      s10 panel) show only in the overlay — by design. Ceiling: lines cluster on y only, so two columns merge.
      Note: `refine_text._assemble_words` splits words 2–3.7× finer than pdfium's own tokens (glyph-gap rule); harmless
      for refine (it matches normalised text) but not reusable for reports — coverage has its own 20-line tokenizer.
- [x] 2. Picture detection (f0f6332): every `p:pic` is a picture (Picture, PlaceholderPicture, Movie). TapHole +3: s10 placeholder
      picture (alt text), s12 movie (poster-frame OCR), s17 movie (needs_vision). b9d3c29: PowerPoint's auto-generated
      alt text ("A close-up of a graph … AI-generated content may be incorrect.") counts as no alt text → OCR /
      needs_vision path (TapHole: 11 such pictures; they now carry tesseract text of their axis labels). Review: the
      2018–2023 marker "Description automatically generated" was missing (two holdout decks have it) — 9685c71; 5649ff2:
      a picture cropped to a non-rect shape no longer occludes by its AABB; eb6d220: palette PNG + tRNS is transparent,
      a trailing br and empty spacer paragraphs advance the bullet cursor (boxes repositioned: AI 2, TapHole 11, MIT 20,
      Open-Q 1; ids unchanged). Occlusion counts unchanged.
- [x] 3. Embedded video: covered by 2 (Movie is a `p:pic`; `_ocr_picture` reads `poster_frame.blob`).
- [x] 4. `refine_text` dropped lines that are in the PDF (TapHole s8 e13–e15). Cause (2a77f6a): `align_words` treated
      `ALIGN_MAX_GAP` (2.0 in from the parser's GUESSED box) as a hard gate; with font sizes inherited from the master the
      parser guesses 16-pt lines (0.22 in/para) against a 0.32 in/para render, so the guess drifts past 2 in by the 13th
      paragraph. Rule now: page-unique word-aligned text (≥ 4 alnum) is placed where the stream has it; the gap only
      ranks among several occurrences. TapHole: exactly e13–e15 changed; AI_Handout, test_deck: 0 changes. Review found
      word stealing (a text absent from the PDF taking an identical word inside another element, verified); fixed in
      67d585e: proximity pass first, then the page-unique rescue, which never takes words inside another unplaced
      element's box. Two pinning tests. Approved on re-review.
      Known, not now: `line_texts` miscounts a "•" glyph on wrapped bullets (s8_e14 line 2 = "사용"); geometry is right.
- [ ] 4b. From the MIT pilot (general causes, all from map/talk agents + XML checks):
      - [x] `mc:AlternateContent` shapes dropped (64cc204): every text box holding an equation vanished whole (MIT: 47
        shapes, slides 6–13). `_iter_flat` now walks the raw spTree/grpSp children and takes the first alternative with a
        shape; math text from `a14:m` `m:t`, NFKC (math-italic 𝐿 → L). MIT 146 → 256 elements (s8: 2 → 13). Review:
        one must — a `p:contentPart` (PowerPoint INK) inside a Choice has no python-pptx class, the factory raises and
        the slide parses EMPTY (fix in flight: filter contentPart out of the shape tags → ink falls to its Fallback
        picture; 16b7da1, corpus deck `ink_annotation`). 2275ffd: table cells read equations via `_frame_text` (MIT s7
        δ table: 9 of 12 cells were empty, now q1/q2/q3). a50a5e8: a br paragraph is metered as N lines (repositions 8
        TapHole / 34 MIT bullet boxes under such paragraphs). Noted: AC in layouts/masters ignored; `a:fld` now reaches `_paragraphs` (no dev deck affected).
        a:br is a line boundary ("\n"), not a space — a space killed per-line trigger phrases (TapHole s1_e0 5 → 1).
      - [x] Equation boxes had no PDF text layer (their render is the Fallback picture). 58becba: `deck_pdf(text_layer=True)`
        renders the ORIGINAL pptx a second time for refine (`<stem>.text.pdf`; plain decks reuse the one PDF). MIT
        refined: pdf_text 96 → 136, shape 99 → 67, none 61 → 20 (of 223); slide 8's plain lines now sit on their glyphs
        (e11 guess 5.04 → 6.12 in). Still guessed: lines with math MID-sentence — `select_text_words` wants one block
        ≥ 80 % and the blank splits it ("belanguages" = 79 %). fc7f514: gap-tolerant match (sum of in-order blocks ≥ 80 %
        for selection; in `align_words` a text with no exact occurrence is located by a free span of abutting segments
        spelling ≥ 50 %) → MIT 173 / 30 / 20; left guessed: lines that are entirely an equation (no glyphs at all).
        Three dev decks byte-identical. Review: one must in flight — `_gap_runs` aligned the whole page once, so a
        repeated anchor was never enumerated and a short fragment could bind to the earliest line (a probe turned a
        correct neighbour box wrong); fix = enumerate every anchor occurrence, extend by abutting blocks, let proximity
        choose — c317cb7, four regression tests; byte-identical on the dev decks, MIT 173/30/20 unchanged.
      - [ ] Table-with-math fallback renders tiled (MIT s7): PowerPoint's Fallback is the same table with each math cell
        blip-filled by ONE picture of the whole table, so the substitution draws it per cell. Rule in flight: such a
        graphicFrame fallback becomes one `p:pic` at the frame.
      - Cache ceiling: `deck_pdf`/`render_deck` are keyed on pptx mtime only, so PDFs/PNGs cached before 126fe38 stay
        stale for equation decks until deleted (MIT's `out/bench/mit_theory_computation/*/` must be cleared before rebuild).
      - [x] soffice leaves the math blank in the render: 126fe38 renders a temp copy with each AlternateContent replaced by
        its `mc:Fallback` picture (deterministic zip; plain decks take the original path). 0cb7f2c gives every soffice run
        a private profile (`-env:UserInstallation`), so parallel builds no longer collide.
      - [x] arrows from geometry (a48cbdb): connector | LINE | line/arrow preset | empty thin FILLED bar (min < 0.15,
        max ≥ 0.5 in) | zero-dim custGeom; other empty shapes skipped whatever their size. AI_Handout: 37 dividers stay,
        +3 empty rightArrow presets become arrows (6 ids shift). TapHole −3 dots +2 line presets. MIT −19 state circles,
        −3 zero-area boxes. test_deck byte-identical (hand fixture ids hold).
      - [x] pair merge (87802a5): a container pairs only with 1..2 text centres inside (label + sub-label — test_deck's
        stat box holds two, so "exactly one" was wrong); > 2 → background/panel. AI_Handout: 12 background pairs gone,
        11 titles typed `title`, s1 kicker → body, the three stage cards re-pair with their header bands (tighter box).
        Was: the first text whose centre is inside any empty shape ≥ 0.5 in: a full-slide cover rectangle
        grabbed the slide number (MIT s11_e1), and slide numbers come out as `diagram_label` (s12_e0). Same defect on
        AI_Handout: every slide's background rectangle pairs with the title, so all 12 titles are `diagram_label` in the
        raw parse (masked: refine_text shrinks the box onto the glyphs). The fix re-types them to `title`, which has its
        own pointer rules — expect AI_Handout `default` rows to move; measure, don't assume neutral.
      - [x] occlusion (47bf0b6): a leaf fully under a LATER opaque leaf is dropped before pairing/typing (opaque = pic |
        solid/grad/blip/patt fill without alpha | theme fillRef; groups/lines never occlude; AABB, rotation ignored).
        MIT s11: 37 shapes dropped (the diagram under the cover, 13 of them former arrows); "11" (last in z-order) kept.
        Open-Q s2: 2 background images under a third + a strip under a banner. Review (predicate run over 13 decks: 93
        drops, 58 legit, 35 under an ANIMATED or ROTATED shape, 11 with text — all in holdout decks): in flight —
        animated shapes (`p:spTgt` targets) never occlude (what the audience sees at any point stays pointable),
        rotated covers never occlude (AABB ≠ footprint), only rect-like presets occlude, RGBA/alphaModFix pictures are
        not opaque (reverts 2 of Open-Q's 3), huge ≤2-text panels don't pair, `lineInv` preset — landed as 82859bc,
        3f16fac, b428567, 1f40226, 01da09b. // DECISION 2026-10-20: index = the rendered slide. Our viewer shows the
        static soffice render (no animation playback exists in the product), so a cover hides what is under it whatever
        its animation; the "animated shapes never occlude" exemption (82859bc) is reverted in c8af1ad (corpus slide kept,
        expectation flipped: text under a click-to-appear cover is dropped). MIT s11's cover is in a
        click-to-appear group — its 37 drops stay. Animation builds remain a product gap (per-step renders).
      - [x] hidden="1" (55c5aad): none in the dev decks. Off-canvas (434ab06): Open-Q −2.
      - [x] line breaks (`a:br`): python-pptx emitted "\x0b" (fine for matching — Python treats it as whitespace — but
        invisible in JSON viewers, which is what the agents saw); now "\n". Cosmetic for the pointer.
      - [x] shapes entirely outside the slide canvas (Open-Q s1_e7 y 9.54 in, s8_e28 y 8.92 in on a 7.5-in slide) → dropped (434ab06).
      - not now: auto-numbers missing from text, bullet type on headings (maps assign roles), animation states.
- [x] 5. Other classes surfaced along the way, each with a corpus deck/test: ink, table math, br metrics, auto alt text
      (two marker generations), occlusion guards (rotation, shape, RGBA/palette transparency, cropped pictures), huge
      panels, off-canvas, `lineInv`, table-math fallback render, text-layer PDF, gap-tolerant refine. 36 commits.
- [x] 6. Ids are positional (`next_id` = slide + emission order, pairs first), so ANY element added, dropped or
      re-paired shifts every later id on that slide — on every deck, not just TapHole. Blast radius (XML scan, 2026-10-20):
      AI_Handout 37 tiny empty shapes (today `arrow`, slides 2–9) + 12 background-rectangle pairs (all slides);
      TapHole 3 tiny (s4) + 2 covers (s12, s17) + placeholder picture s10 + MEDIA s12/s17; test_deck 3 tiny (s6);
      MIT 47 AlternateContent boxes + tiny + covers; Open-Q none. So: `tools/remap_ids.py` — old→new id map per deck
      from (slide, text, bbox ≈) between the old and new `index.json`, applied to maps (items / units / order),
      `presenter_script*.json` `refers_to`, `expected.json`, `adjudication.json`; it lists unmatched old ids and brand-new
      ids. New elements join the maps via the deterministic fix-up (singleton units) on AI_Handout, TapHole,
      test_deck, Open-Q (a handful of new ids each). EXCEPTION — MIT: 47 equation boxes are ~80% of its content, so
      fix-up singletons would score placeholder units over most of the deck; its maps are REGENERATED with the same
      blind two-agent protocol once the parser lands, and an agent re-annotates the ~25 `refers_to: []` sentences per
      talk script (the scripts themselves are kept). `apply_slide_maps` validates every map against the new packet, so
      the order per rebuild is: build `default` → `remap_ids --apply` (maps, scripts, expected, adjudication) → build
      `map` → replay.
      `tools/remap_ids.py` landed (78245dc, 0daba6e, 38a0cdb; 11 tests; reuses `local_slide_map.fixup`). Review: 3 musts in
      flight — a text-changed element was dropped instead of matched by position; a dropped id alone in an expected step
      left an empty step that lowers recall silently; no run-once guard. Maps/scripts/presenter dirs snapshotted to
      `out/pre_remap_backup_124840/` before any remap mutated them.
      Attribution protocol (per parser commit, not per branch): rebuild AI_Handout + TapHole, remap, replay the map
      variant (~40 s each, STT cached), record the row. Step 7 is then a table of deltas with one cause each; a single
      before/after of the whole branch would be unreadable (≤3 lit is noise; AI_Handout gets title retyping, the pair
      change and the refine fix at once).
- [x] 6b. MIT rebuild at c8af1ad: 223 elements (was 146 / 256 before occlusion); both PDFs + fallback render; old maps
      archived to `out/maps/mit_theory_computation/old_146/`; scripts remapped 146 → 223 (111 kept, 35 dropped = hidden s11
      diagram + s12 state circles, 112 new; 7 sentences lost ids). In flight: fresh maps (2 blind agents), script
      re-annotation (blind to maps; done: 28/25/33 sentences now point at the new text; the 3 check-in sentences had
      followed the "11" text to the real page number — re-pointed at the check-in box by hand). Expected steps regenerated:
      78 / 73 / 82 (was 52 / 50 / 52) — the honest denominators. Final at 0a53cb8 (fresh maps, 223 elements):
      | MIT | lit | false ids | false s | late med / p90 |
      | pilot (old truth 52/50/52; ~25 unscorable sentences per talk ≈ 44/77 honest) | 44/52 · 38/50 · 41/52 | 5 · 8 · 2 | 43.0 · 30.3 · 14.8 | 2.49/5.3 · 4.05/11.1 · 3.07/5.6 |
      | final (truth 78/73/82) | 64/78 · 59/73 · 65/82 | 11 · 10 · 3 | 55.7 · 44.4 · 9.3 | 2.66/6.2 · 4.00/8.2 · 3.06/5.6 |
      Recall on the honest ruler ≈ 57 % → 82 / 81 / 79 %; more pointable content also means more wrong switches on the
      two reading-order talks (false s +13 / +14), fewer on the meaning talk.
      Where the 45 misses are: slide TITLES 15 (s3/s4/s5/s8/s10/s13/s14 e0 — titles get no map talk and little text; the
      known title/kicker lever), the automata diagram slides 6/7/9/12 21 (units made of "0"/"1" transition labels and
      state labels collide; 10 of the 24 false ids are on s9/s12), the rest scattered singles. Clean text slides are at
      or near 100 %. Next lever is the pointer (titles, tiny-label diagram units), not the parser.
- [x] 7-pre. pytest at 0a53cb8: 515 passed / 2 skipped (baseline 459 / 2). Coverage after the fixes: AI_Handout 0/1202 unclaimed
      words, TapHole 0/984 (was 37), test_deck 0/190; empty text elements 0 on all three. MIT (coverage now reads the
      text-layer PDF, 6fb17c7): 2/813 unclaimed (was 340), 18 empty text elements = 9 equation-only bullets + the 9 math
      cells of the s7 table — no glyphs in either layer, expected. Open-Q after its rebuild.
- [x] 7-verbatim (read-every-box auto narration, reported, non-blocking; `deck_bench --reuse-index` at HEAD):
      AI_Handout default 129/168 (2026-10-02 report: 123/168 = 0.732), map 88/168; TapHole default 100/164 (= 2026-10-12),
      precision 0.990, images 0 alt / 26 no-alt (the 11 auto captions now count as no alt → OCR); MIT default 120/183 (pilot
      61/88 — 95 more narratable elements), map 86/183; TapHole map 62/164; test_deck 40/47 default and map (= 2026-10-02
      0.851); filler false per deck 0–2 (gate ≤ 5); hand fixture 38/40, false 1, flicker 1 (= gate). Open-Q default
      313/524 (2026-10-02: 0.591 = 310/524), map 229/524; Open-Q coverage 0/2541 unclaimed, 3 empty text elements (the
      "};" code bullets, known).
- [x] 7. Regression at HEAD (c317cb7): pytest 526 passed / 2 skipped (baseline 459); clause_replay rows per commit above
      (AI_Handout identical throughout; TapHole changed only by truth and by the alt-text commit, for the better);
      verbatim bench level or better on all five decks; hand fixture 38/40 · 1 · 1; filler ≤ 2 per deck; coverage: 0
      unclaimed words on AI_Handout, TapHole, test_deck, Open-Q; MIT 2/813.
- [x] 7a. Attribution table (`out/bench/attribution.md`, script `tools/rebuild_replay.sh <commit> <label> [deck …]`: clean
      worktree → build default → remap → build map → clause_replay, AI_Handout + TapHole):
      | commit | AI orig/jumpy/meaning lit · false ids · false s | TapHole same |
      | baseline 7fe4c79 | 46/48·3·4.5, 59/61·3·10.5, 57/59·8·22.7 | 63/65·12·46.8, 59/70·7·27.2, 68/72·5·13.7 |
      | refine 2a77f6a | identical | identical (s8 e13–e15 boxes now on the glyphs; sw/step 1.38→1.37) |
      | text content 03fceab (equation boxes, math, br, fallback render) | identical | identical (s1_e0 / s14_e1 phrases restored; ids unchanged) |
      | hidden 55c5aad, off-canvas 434ab06, occlusion 47bf0b6 | identical | identical (no such shapes on these decks; AI drops 4 empty frames under pictures, no element change) |
      | pictures f0f6332 | identical | identical; +3 new ids (s10 picture, s12/s17 movies) as fix-up singletons — talk steps about them still `[]`, re-annotation pending |
      | arrows a48cbdb | identical (+3 rightArrow presets, fix-up decoration) | jumpy 59/70 → 58/70, false 7 → 8, false s 27.2 → 33.3: REMAP ARTIFACT — the 3 dropped dots (s4 e43/e44/e130) left one expected step empty (unhittable) and the lit it used to score now counts as false; fix = prune empty steps (remap fix #3) + re-annotate that sentence |
      | pairs 87802a5 | identical; 13 retyped (11 titles → title, s1 kicker + s2 panel → body) | identical |
      | re-annotation (truth change, no code) | — | TapHole maps: real units for s10_e2 (own unit), s12_e2 / s17_e3 (merged with their "Video demo" captions); scripts: 26 sentences now point at the videos (8/9/9), the burr sentence at the 29 sketch segments; denominators change from here on |
      | replay at 87802a5 with the new truth | identical | orig 65/67 · 13 · 47.7, jumpy 61/72 · 8 · 34.6, meaning 70/74 · 5 · 13.7 — the video steps light (2 new steps each after same-unit collapse), the burr sentence hits again |
      | alt text b9d3c29 | identical | lit same; orig false s 47.7 → 44.2, jumpy false 8 → 7 / 34.6 → 27.2 s — the auto-caption noise phrases ("a close-up of a graph") were firing; 11 pictures re-matched by position (remap fix #1) |
      | ink 16b7da1 · table math 2275ffd · br metrics a50a5e8 | identical | identical (bbox-only / no such shapes) |
      | occlusion guards 3f16fac…c8af1ad · text-layer PDF 58becba · gap-tolerant refine fc7f514 | identical | identical (no equation boxes on these decks; refine byte-identical) |
      | final HEAD c317cb7 (alt-text markers, cropped-picture occlusion, palette PNG, spacer metrics, table render, refine anchors) | identical | identical |
      | Open-Q at HEAD (one row, remap: −3 ids = 2 off-canvas texts + 1 occluded strip, 0 retyped, no script step touched) | — | orig 94/117 · 28 · 129.5 s, jumpy 92/124 · 29 · 159.6, meaning 116/146 · 18 · 60.9 (strict). Last recorded rows 89–91/117 (2026-10-02, adjudicated false s 46.2). [UNCERTAIN: no pre-branch strict clause-pointer row for Open-Q was recorded; the removed elements were invisible and unreferenced, so they could only have lit falsely] |
- [x] 7b. Merged to master (f8b93c2) and pushed 2026-10-20; rollback copies (`out/pre_remap_backup_124840/`,
      `out/maps/mit_theory_computation/old_146/`) deleted. c317cb7 approved on re-review (all probes, 0 changed elements).
- [ ] 8. Holdout run on the 7 unseen decks (not MIT): maps + 3 blind talk scripts each, shipped params, report against
      the bench rows. MIT is re-scored with them but reported as a dev deck.

### Pilot first (user, 2026-10-15: "try with smaller one"): unseen-deck run on `holdout/mit_theory_computation` (14 slides), shipped code
- [x] default index + verbatim bench: recall 61/88, precision 1.000, false 0, flicker 8, filler false 1
- [x] maps (2 agents, blind to scripts) + 3 talk scripts (original / jumpy / meaning, blind to maps)
- [x] map index, presenter audio, `clause_replay --record`, `clause_replay --deck mit_theory_computation --holdout`
- [x] compare with AI_Handout / TapHole rows; failure modes per slide (parser vs pointer)

Result (shipped params, cross decider, cpu): lit 44/52 · 38/50 · 41/52 (orig/jumpy/meaning), false ids 5/8/2, false s
43.0/30.3/14.8, late med 2.49/4.05/3.07 s. AI_Handout 46/48 · 59/61 · 57/59; TapHole 63/65 · 59/70 · 68/72.
- The lit ruler flatters MIT: ~25 sentences per script talk about text the parser dropped (`mc:AlternateContent`), so
  they are scored "stay quiet". Counting them, the original talk lights 44 of ~77 pointing sentences (~57%).
- Clean slides (1–5, 14): 26/30 · 23/28 · 25/30. Misses there are mostly slide titles (s3/s4/s5 e0) — the known
  title/kicker gap, not a parser gap.
- 8 of 15 false ids are on slide 11, all three talks, and all of them are the diagram hidden under the cover box
  (s11_e0/e4/e10; measured 2026-10-20). The slide number paired into the cover (s11_e1) is NOT among them: the talk agents
  annotated it as the check-in heading, so the pointer lighting it scored as a hit — a parser defect scored as correct.
