# smartppt Improvement Research — 2026-06-24

**Premise.** The performance/footprint/determinism audit (`docs/optimization-research-2026-06-18.md`) is essentially *done*: bge-small encoder, sherpa-onnx streaming STT, CPU/thread/offline determinism pinning, dependency manifest, `.npz` cache, raw-RGB video pipe, dead VLM/LLM branch deleted. This report covers the **next frontier** — what makes the tool *more accurate, more general, more maintainable, and more of a real product* — not speed.

**Hard constraint (unchanged, overrides every "best tool" instinct):** fully **offline**, **deterministic** (same input → same highlight), **CPU-only**, **permissive license** (no AGPL). Every recommendation below is installable+runnable offline on CPU and determinism-preserving, or is explicitly flagged otherwise.

**Method.** Four parallel read-only research passes over the live tree (grounding accuracy, generalization, architecture/testability, product/UX+STT). Cross-validated claims spot-checked against source.

---

## 1. Executive summary

The single loudest signal: **the streaming STT engine already in use exposes native contextual biasing that the code explicitly throws away** — `src/stt.py:318` does `del hotwords  # the matcher's embedding handles deck-term robustness`, while `hotwords_from_index()` already computes the deck vocabulary and `timeline_audit.py` already plumbs it through. *Two independent research passes (accuracy and product) ranked this #1.* It fixes mishearings like "sync"→"sink" (`s4_e5`) **at the source**, before the embedding ever sees corrupted text — strictly better than the post-hoc `data/aliases.json` band-aid. No new dependency, deterministic under the existing single-thread greedy decode. **Ship first.**

The biggest *latent* risk is **generalization**: the entire system is tuned and tested against exactly one deck (`test_deck.pptx`, 7 slides). Grouped shapes / SmartArt **vanish without error** (`parser.py` walks `slide.shapes` only, never recurses `GROUP`), image elements dead-end at `needs_vision` (the VLM that handled them was deleted), charts collapse to 4 single-series archetypes, and group/expansion vocabulary is hand-authored per deck. A real user's deck will silently lose whole regions. The unlock that *gates* fixing any of this safely is a **code-generated diverse deck corpus** — the only thing that catches "works on deck A, breaks on deck B."

The highest *visible-quality* win is **line-level highlighting**: `refine_text` already emits per-line `line_bboxes`, the viewer just never uses them. Whole-box highlighting on a tall bullet body is the most obvious "too coarse" complaint a real presenter will have, and the data is already on disk.

The structural debt worth paying: the index build is **three disk-staged CLIs joined only by docstring prose** with a silent `best_index()` selection footgun, and `viewer.py` (515 LOC) **couples the testable live-decision loop to Tk**, which is exactly why the four `simulate*.py` entry points had to re-implement that loop (and drifted — the PNG path forgot the alias normalizer the video path has). One `LiveController` seam fixes the testability gap *and* the duplication *and* a latent correctness divergence.

Two free bug fixes fell out: an **undefined-name `VoskEngine`** in two type annotations (`simulate_to_png.py:65`, `simulate_to_video.py:59` — only `load_default_stt_engine` is imported; survives solely because `from __future__ import annotations` never evaluates it), and **stale "BGE-large"/"Vosk" log strings** that now lie about the running engine.

---

## 2. Top priorities (ranked by impact / effort)

| # | Recommendation | Axis | Impact | Effort | Key evidence |
|---|---|---|---|---|---|
| 1 | **Wire sherpa-onnx hotword/contextual biasing** (stop discarding deck vocab) | accuracy+STT | High | Low | `stt.py:318` `del hotwords`; `hotwords_from_index` `stt.py:139`; plumbed in `timeline_audit.py:41` |
| 2 | **Code-generated diverse deck corpus + acceptance harness** | generalization | High (enabling) | Medium | one deck only; floor table in `test_grounding_eval.py` is literally this deck's per-slide counts |
| 3 | **Recurse GROUP shapes / SmartArt + read speaker notes + shape rotation** | generalization | High | Med–Large | `parser.py:347` walks `slide.shapes` only; `GROUP` dropped at `:427`; `rotation` ignored `:60` |
| 4 | **Line-level / sub-element highlighting** (use `line_bboxes`) | product | High | Medium | `refine_text` emits `line_bboxes`; viewer draws whole-element `bbox` only |
| 5 | **`build_index` orchestrator** (one in-process build; warn on stale `best_index`) | architecture | High | Small–Med | 3 CLIs joined by prose `simulate.py:10-14`; silent selection `matcher.py:68-82` |
| 6 | **`LiveController` seam** (headless live loop out of Tk) | architecture+test | High | Medium | `viewer.py:449-466` decision logic unreachable without `tk.Tk()`; re-implemented in both simulators |
| 7 | **Deterministic lexical fusion** (hand-rolled token-set score blended into cosine) | accuracy | Medium | Medium | cosine-only `matcher.py:241-249`; recovers `s3_e5` "offline" without touching `gap` (which spikes flicker) |
| 8 | **Image elements: alt-text → tesseract OCR → needs_vision fallback** | generalization | Medium | Medium | `parser.py:408-416` dead-ends raster content; alt-text (`descr`) ignored |
| 9 | **Auto-derive expansions + drop `_TITLE_KEYWORDS` whitelist** | generalization | Medium | Small | `expansions.json` 1 entry; `group_scope.py:51-55` hardcodes this deck's titles |
| 10 | **Partial-vs-final highlight commitment** (viewer keeps the `partial` it's handed) | product | Medium | Low–Med | `viewer.py:433` discards the `partial` object `live.py` already passes |
| 11 | **Fix `VoskEngine` undefined-name + stale engine logs** | architecture | Low (bug) | XS | `simulate_to_png.py:65`, `simulate_to_video.py:59`; `viewer.py:82` "BGE-large" |
| 12 | **One-command onboarding** (`prepare`/`smartppt deck.pptx` + `doctor` preflight) | product | Medium | Low | 5-step manual pipeline; steps already idempotent; auto-aliases already work |

---

## 3. Per-axis findings

### 3.1 Grounding accuracy (the product metric: ~37–38/40)

**1 — Hotword/contextual biasing (rank 1, do first).** `stt.py:318` discards the `hotwords` argument that flows all the way from `hotwords_from_index()` → `LiveTranscriber` → engine. The vendored streaming-zipformer is BPE (`tokens.txt` is `▁`-prefixed), and the installed sherpa-onnx 1.13.x `from_transducer(...)` accepts `hotwords_file` + `hotwords_score` + `modeling_unit="bpe"` + `bpe_vocab` — currently passed *none* (`stt.py:295-309`). Boosting deck terms raises their decode probability before corruption, fixing the whole "sync"→"sink" mishearing class that aliases can only patch one-by-one. **Deterministic** (fixed additive beam score under single-thread greedy). **Caveat:** confirm the model dir ships `bpe.model`/`bpe_vocab` (one-line add to the fetch script if not); transcripts change so re-run the eval and re-pin the floor. `// DECISION:` start `hotwords_score≈1.5–2.0`, sweep on the eval. ~15 lines.

**2 — Lexical fusion (rank 7).** Scoring is purely `embs @ q` (`matcher.py:241-249`). bge-small compressed the cosine range (threshold was raised 0.55→0.65), eroding discrimination on short exact words. The `s3_e5` "offline" loss is a single exact token; recovering it via `gap=0.01` spikes flicker 0→7 (noted in `test_grounding_eval.py`), but a small lexical term (`α·cosine + (1-α)·token_set_ratio`, α≈0.2–0.3) pins it **without** touching the gap, and disambiguates the s2 "< 1 ms" stat vs body paraphrase. **Hand-roll the token-set score (~20 lines, stdlib)** rather than add rapidfuzz — honors zero-new-dep and is equally deterministic. Risk: α is a new global tunable; keep it a tie-breaker, never dominant, and co-calibrate with threshold/gap.

**3 — Identical "Live" cells (s3): merge, don't disambiguate.** The three cells share text/phrases but have distinct `bbox.y`. Honest finding: `expected_highlights.json` already accepts any of the three, so positional disambiguation yields **zero eval gain** and the narration gives no positional cue to bind to. The real win is a `group_id` merge to stop the matcher oscillating between them — a *flicker* reduction, which the metric does count. Skip positional NLU until real decks contain "the second row" phrasing.

**4 — Better partial handling (low/med).** `RollingBuffer.tail(8)` was tuned for Vosk's never-finalizing partials, but sherpa *does* finalize on endpoint (`stt.py:330-334`), so the rationale is partly stale; sweep `tail_words ∈ {6,8,10,12}` and the endpoint rules (`stt.py:306-308`) alongside rank 1's re-baseline.

**Rejected:** **cross-encoder rerank** — reintroduces 5–20× the per-frame cost bge-small was adopted to kill, on a loop that runs every partial, to fix failures that are *lexical/acoustic* not semantic-subtlety (wrong tool, wrong cost). **Standalone phonetic matching (metaphone)** — strictly dominated by rank 1, post-hoc, new dep, determinism foot-gun. **Per-type thresholds** — promising but would overfit the 40-step eval; revisit only after rank 2 grows the corpus.

**Eval-harness gap.** `timeline_audit.py` loads exactly one deck/fixture; flicker is computed but not in the headline metric (loose `≤10` assert); the metric is recall-only and can't see *false* highlights (precision). Fold flicker + a false-positive penalty into the number before tuning α or per-type thresholds.

### 3.2 Generalization to real decks (the biggest latent risk)

*Corrected assumption (good news):* `asr_normalize.build_alias_map` **auto-derives** acronym→spoken-variant aliases from deck text; `data/aliases.json` is only a 2-line override. So ASR acronym normalization already generalizes. The bespoke files are `expansions.json` and `group_scope`'s keyword tables.

Ranked by likelihood of biting a real user:

1. **Grouped shapes / SmartArt vanish (rank 3 above).** `parser.py:347` iterates `slide.shapes`; a `GROUP` has no text frame and is dropped at `:427`. SmartArt process diagrams — extremely common — produce *nothing*, no error. Fix: recurse `GROUP` with accumulated child offsets (python-pptx gives relative `left/top`). Also currently ignored and free: `slide.notes_slide` (rich phrase source), `paragraph.level` (bullet structure), `shape.rotation` (rotated shapes get an axis-aligned, wrong box at `parser.py:60`).
2. **Image elements dead-end (rank 8).** `parser.py:408-416` → `needs_vision=True, trigger_phrases=[]`, never revisited since the VLM was deleted. Deterministic offline ladder: **pptx alt-text** (`shape._element` `descr`/`title` — author-written, zero cost, currently ignored) → **tesseract OCR** on the already-rendered crop (Apache-2.0, deterministic with pinned `--oem/--psm` + pinned traineddata) → `needs_vision` only if both empty.
3. **Charts = 4 single-series archetypes (rank 4, Large).** `chart_geom.classify_chart_type` folds area→line, doughnut→pie, and radar/combo→default "bar" (`chart_geom.py:40`); `_expand_chart` reads only `plots[0]`/`series[0]` (`parser.py:286-287`). Stacked bars break the height-to-value matching (`chart_detect.py:104-113`); pie is never measured-from-render (`chart_detect.py:308-314`). Fix incrementally: multi-series iteration + color-segment measurement for stacked; measure pie center/radius from pixels; explicitly mark radar/combo `needs_vision` instead of silently mislabeling.
4. **`≥3 pairs ⇒ diagram_label else stat` is a deck-specific slide-wide guess** (rank 9 above, `parser.py:20,363`). 3 KPI callouts → mislabeled `diagram_label` (loses stat phrases); 2-node diagram → `stat` (nonsense numeric phrases). Fix: classify each pair by its *own* text against the stat regex, not the slide's pair count.
5. **Hand-authored vocabulary (rank 9).** `expansions.json` has one entry and no derivation; `group_scope.py:51-55` hardcodes this deck's slide titles. Auto-derive expansions from `ACRONYM (Expansion Words)` patterns (first-letter check, pure regex — mirror the proven alias design); derive group labels from the chart's own title / slide title verbatim, drop the whitelist.
6. **Geometry magics** — `title y<1.2"` (`parser.py:333`): prefer `placeholder_format.type == TITLE` (authoritative, ignored today), constant as fallback only. `font×1.2`/16pt default: largely cosmetic *when* refine_text runs (PDF glyph layer overwrites), risky only if skipped. `SHIFT_FRAC=0` and `150 dpi`: ratio-relative, low risk. `chart_detect` band thresholds (`_MIN_BAND_PX=4`) are small absolutes — make dpi-relative.
7. **Error handling / scale.** `parse_deck` has no try/except (corrupt pptx throws raw); `_expand_chart` swallows failures into an un-pointable empty chart; render `timeout=120s` will fail a 100-slide deck; `LETTER_SOUNDS`/word-assembly are English/LTR-only. Wrap per-slide (warn, don't abort), scale timeout with slide count, treat page-count mismatch as a refinement-disable warning not a crash.
8. **Single test deck (rank 2 — the enabling unlock).** Build `tools/make_corpus.py` that **generates** decks with python-pptx covering each gap (grouped/SmartArt-like, multi-level bullets, stacked/combo/scatter/doughnut, rotated, image-only ±alt-text, 100-slide scale, non-English, 4:3). Because they're code-generated, ground-truth element type/bbox is **known by construction** — no manual labeling, fully deterministic. Assert: every intended element parsed (no silent drops), bbox lands in the rendered region, page-count invariant holds, no exception on a deliberately-corrupt fixture. Gate behind the existing `SMARTPPT_RUN_EVAL` pattern. **This is what keeps fixes 1–7 from regressing.**

All fixes use python-pptx (MIT) / numpy / tesseract (Apache-2.0) — permissive, CPU, deterministic with pinned inputs.

### 3.3 Architecture / code quality / testability

1. **No `build_index` orchestrator (rank 5).** `index_deck → refine_text → group_scope`, three `python -m` calls each re-reading the prior JSON, wired only by docstring prose. `best_index()` (`matcher.py:68-82`) silently picks scoped>refined>base, so a stale `index.refined.json` beside a fresh `index.json` is ignored with no warning — a real footgun. Add `src/build_index.py` calling the three already-pure functions in-process; keep the CLIs as single-stage debug shims; have `best_index` warn when it picks a file older than a sibling it skipped. Pure refactor, outputs unchanged.
2. **`LiveController` seam (rank 6).** The live *decision* (`viewer.py:449-466`: tail-slice → `policy.step` → status) is a method on a class whose `__init__` calls `tk.Tk()`, so it's untestable headless — which is precisely why `simulate_to_png.py`/`simulate_to_video.py` re-implement the loop, and why the PNG path silently **forgot the alias normalizer** the video path has (a latent correctness divergence). Extract a Tk-free `LiveController.feed(text) -> event`; viewer and all simulators consume it. Unlocks a deterministic `MockEngine → LiveController → policy` test of the live path.
3. **Config sprawl.** `window_seconds=4.0` magic-duplicated at 4 sites; `tail(8)` at 3; `ANIM_FRAMES` defined **twice with different values** (viewer=10, video=8 — a silent animation mismatch between the Tk view and the baked video); determinism env pins are import-time side effects in `encoder.py:23-26`; policy defaults re-plumbed as argparse defaults in 3 entry points. Centralize in `src/config.py` (frozen dataclasses + one explicit `apply_determinism_env()`), preserving `setdefault`/import-order semantics.
4. **Observability.** Zero `logging` in `src/` — all bare `print()`. The status line shows the *winner* only; when a highlight doesn't fire the user can't see runner-up/gap/threshold margin, though `policy.step` already computes them (`matcher.py:361-363`) and throws them away. Add a `MatchTrace` + `--explain`; convert prints to module loggers. Biggest debuggability gap, read-only, deterministic.
5. **Tooling.** ruff `[tool.ruff]` selects *no* rule families; no mypy. The `Encoder`/`STTEngine` Protocols are defined but call sites are untyped, including a `setattr`+`# type: ignore` stash in `matcher.py:361-363`. `F` alone catches the `VoskEngine` undefined-name bug. Add mypy (lenient, scoped to encoder/matcher/stt/live first) and ruff `E,F,I,UP,B`.
6. **Free bug fixes (rank 11).** `VoskEngine` undefined annotations (`simulate_to_png.py:65`, `simulate_to_video.py:59`) → use `STTEngine`. Stale logs: `viewer.py:82` "BGE-large" (now bge-small), four "Vosk" labels in `viewer.py:453-490` missed by commit `0910a40`. Orphan `tests/__pycache__/test_enrich.*.pyc` with no source — remove + gitignore `__pycache__`.
7. **No README.** Build steps survive only in scattered module docstrings; the `SMARTPPT_RUN_EVAL`/`SKIP_SEMANTIC` switches are buried in skip-reason strings. A minimal README (documenting the `build_index` command + env switches) compounds with rank 5.

Sequence: bug fixes (#6, minutes) → `build_index` + shared simulate setup (#1) → `LiveController` (#2, unlocks the trace and the threading test) → config (#3) → mypy/ruff (#5) to lock it in. Every step behavior-preserving.

### 3.4 Product / UX + STT frontier

**Tier 1 — high value, feasible now.**
- **Line-level highlighting (top pick, rank 4 above).** `line_bboxes` already emitted; highlight the spoken line, not the whole box. Index-side work mirrors `_expand_chart` (explode into per-line pointable children sharing the parent's `group_id` so existing `stay_bonus`/`same_group_bonus` prevents intra-element flicker); viewer change ~15 lines. The most visible quality jump.
- **One-command onboarding (rank 12).** `smartppt deck.pptx` runs the idempotent pipeline then launches; gate TTS behind `--demo`; add a `doctor` preflight (sherpa model? soffice on PATH?). Auto-aliases already work, so the "go" path needs zero hand-authoring to *function*. Removes the adoption wall, ~½ day.
- **Partial-vs-final commitment (rank 10).** `viewer.py:433` discards the `partial` object `live.py` already passes. Queue `(text, is_final)`; let partials drive a *provisional* highlight (extend an already-active group only) and finals confirm. Deterministic, snappier feel, plumbing already exists.
- **Spotlight / dim-the-rest.** Tk has no alpha, but PIL pre-compositing (already used in `preview.py`/`simulate_to_video.py`) gives a dimmed slide with the active region punched bright. Behind `--spotlight`; keep the outline as default. Reads better than a thin line for a back-of-room audience.

**Tier 2 — the strategic bet (real build).**
- **Presenter-mode overlay onto the *actual* running slideshow (#5)** — a transparent, always-on-top, **click-through** window over the presenter's real PowerPoint/LibreOffice fullscreen, instead of showing our own PNGs. Confirmed feasible on *this* box: X11 with Composite + SHAPE + XFIXES and Mutter compositor. Click-through = empty X SHAPE **input** region (clicks pass to the slideshow; clicker still advances). Needs `python-xlib` (pure-Python, permissive, **not installed**) or a tiny ctypes `libXext` shim; highlight drawing ports almost unchanged (same `geom.py`). **Honest limit:** Wayland has no portable always-on-top click-through-over-another-app API — `[UNCERTAIN]` for the broader user base. `// DECISION:` target X11 first, keep the self-rendered viewer as universal fallback, document Wayland as unsupported for overlay mode.
- **Slide-change detection (#6, prerequisite for overlay).** In overlay mode we don't control nav. Strong deterministic option: **screen-capture + perceptual hash (dHash, a few lines of numpy) against the precomputed hashes of our rendered slide PNGs** — exact, content-independent, with a `MatchPolicy`-style debounce for animated builds. Vocabulary-from-speech is a weak tiebreaker only. Also removes manual `PgDn` friction in self-rendered mode.

**Tier 3 — solid, lower urgency.**
- **Mic selection (real podium failure).** `MicSource` has a `device` param **nothing exposes** — a USB lapel mic loses to the laptop mic. Add `--device`/`--list-devices` + startup picker (sounddevice already enumerates). Also: `LiveTranscriber` thread death just prints and freezes the highlight — surface it to the status line and attempt one restart.
- **Confidence indicator + feel presets.** `score`/`why` already computed; map to a 3-state dot, add `--stable` vs `--responsive` presets flipping `stickiness`/`clear_after_n_low`.
- **Highlight event emitter.** `simulate_to_video` already records `(timestamp, active_id)` transitions; expose them as live JSON-lines for OBS browser-source / second-screen presenter view. Ship the emitter primitive first.

**STT frontier.** Contextual biasing is §3.1 rank 1 (cross-validated here). Endpointing rules are exposed, tunable, deterministic. Multilingual *STT* is cheap plumbing (engine is model-path-parameterized via `SMARTPPT_SHERPA_MODEL`), but multilingual *grounding* needs a multilingual embedder (e.g. bge-m3, permissive) — a real project. Speaker adaptation: out of scope for the deterministic offline ethos.

---

## 4. Considered & rejected (or deferred)

| Idea | Verdict |
|---|---|
| Cross-encoder (MiniLM) rerank in the live loop | **Reject** — 5–20× per-frame cost on the hot path bge-small was adopted to lighten; fixes lexical/acoustic errors a cross-encoder is the wrong tool for. |
| Standalone phonetic (metaphone/soundex) stage | **Reject** — dominated by hotword biasing (fixes the error *before* corruption); post-hoc, new dep, determinism foot-gun. |
| Positional-NLU to split the three "Live" cells | **Reject** — ground truth already accepts any; narration has no positional cue. Merge into one group instead (flicker win). |
| Per-element-type thresholds | **Defer** — would overfit the 40-step eval; revisit after the corpus (rank 2) grows. |
| rapidfuzz for lexical fusion | **Defer to hand-rolled** — MIT/offline but a new runtime dep; a ~20-line stdlib token-set ratio is equally deterministic. |
| Warm LibreOffice / unoserver, ONNX encoder, GGUF LLM, faster-whisper | Already assessed net-negative/moot in the 2026-06-18 program; unchanged. |
| Wayland overlay mode | **Out of scope** — no portable click-through-over-another-app API; X11-first, document the limit. |
| Multilingual grounding | **Defer** — needs a multilingual embedder; multilingual *STT* is cheap, grounding is a project. |

---

## 5. Suggested roadmap

**Phase A — high-leverage, low-risk (days each).**
- Hotword biasing wired in `stt.py` + re-baseline the eval (§3.1.1). *Cross-validated #1.*
- Free bug fixes: `VoskEngine`→`STTEngine`, stale logs, orphan pyc (§3.3.6).
- `build_index` orchestrator + `best_index` stale-sibling warning + minimal README (§3.3.1).
- Mic `--device`/`--list-devices` + engine-restart-on-death (§3.4 Tier 3).

**Phase B — quality + testability.**
- `LiveController` seam → headless live-loop test + shared simulate setup (kills the normalizer drift) (§3.3.2).
- Line-level highlighting via `line_bboxes` + per-line sub-elements (§3.4 Tier 1). *Most visible win.*
- Hand-rolled lexical fusion + fold flicker/precision into the eval metric (§3.1.2).
- Partial-vs-final commitment; config centralization; ruff/mypy (§3.3.3/5, §3.4).

**Phase C — generalization (gated by the corpus).**
- `tools/make_corpus.py` + acceptance harness (§3.2.8). **Do before shipping C2–C4.**
- Recurse GROUP shapes / SmartArt + notes + rotation (§3.2.1).
- Image elements: alt-text → tesseract OCR (§3.2.2).
- Auto-derive expansions / drop title whitelist; per-pair stat classification; placeholder-type title detection; per-slide error isolation (§3.2.4/5/6/7).
- Multi-series / stacked / pie-measured charts (§3.2.3, Large).

**Phase D — strategic product bet.**
- Presenter-mode X11 click-through overlay + perceptual-hash slide detection (§3.4 Tier 2). The ceiling-raiser; honest Wayland limit; self-rendered viewer stays the fallback.

**Determinism discipline (unchanged):** every new lever pins device=CPU + single-thread, pinned model/tool versions and revisions, and folds model_name + version sentinels into content-hash cache keys. Hotword files, OCR traineddata, and corpus decks are deterministic inputs — pin them like any other.
