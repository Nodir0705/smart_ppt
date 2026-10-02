
## 2026-06-18 — Adversarial verify of render+viewer tool recs
- x264 output bytes change with thread count (verified threads=1/2/4 → 3 distinct SHA-256). Current `_encode_video` sets NO `-threads`, so it inherits core-count default = accidentally machine-dependent. Any "deterministic encode" MUST pin `-threads 1` / `stream.thread_count=1`.
- Installed PyAV 12.3.0 bundles its OWN ffmpeg (libavcodec 60.31 ≈ ffmpeg 6.1), NOT system ffmpeg 4.3.2. Swapping to PyAV changes MP4 bytes vs the current pdftoppm→system-ffmpeg path even at threads=1. This is fine here ONLY because the MP4 is a terminal artifact (no golden-hash test, never re-read into a decision loop) and simulate_to_video.py is an OFFLINE batch CLI, separate from the live grounding loop.
- Lesson: "deterministic" must be scoped — per-backend reproducibility ≠ cross-backend byte-identity. For artifacts not feeding a decision loop, only per-backend matters.

## 2026-06-24 — sherpa hotword biasing: empirically REFUTED for this deck (negative result)
The 2026-06-24 improvement research ranked "wire sherpa-onnx hotword biasing (stt.py:318 `del hotwords`)" as the #1 accuracy lever, claiming it fixes "sync"→"sink" (s4_e5) at the source. Hands-on probing on the vendored streaming-zipformer-en-2023-06-26 int8 model refuted this:
- **It works mechanically** (no crash, deterministic 2× identical) but ONLY under `decoding_method="modified_beam_search"` + `modeling_unit="bpe"` + `bpe_vocab=<text vocab>`. `bpe_vocab` is NOT the binary `bpe.model` — it's a two-column `token logprob` text file you must export from the sentencepiece model (`spm.id_to_piece/get_score`). Passing `bpe.model` → "Each line in vocab should contain two items"; per-stream `create_stream(hotwords=...)` under greedy → **segfault** (no validation).
- **It does not fix the target.** With hotword `sync`, "and sink is four" stayed "sink" at score=2/3/6. At score=6 it *degraded* an unrelated word ("asty t"→"z d t"). Contextual biasing can't overcome a confident acoustic "sink"→"sync" confusion here.
- **Forces a decoding-method change.** greedy→modified_beam_search alters transcripts even with no hotwords → would require a full grounding-eval re-baseline for zero demonstrated gain.
- **The alias-layer alternative is net-zero too.** Adding `"sink":"sync"` to data/aliases.json DOES correctly transcribe it and lights s4_e5 — but then s4_e3 (Match) misses instead (adjacent "Match is one. And sync is four."), so OVERALL stays 37/40. It relocates the miss rather than removing it, and risks legit "sink" on other decks. Reverted.
- **Lesson:** verify a recommended STT/accuracy lever against the SPECIFIC failing utterance before wiring it — a plausible mechanism (contextual biasing) can be mechanically real yet empirically inert, and "fix it at the source" can just relocate the error when the failing elements are adjacent. The existing post-STT alias layer remains the right tool, but only for mishearings where the correct word doesn't collide with an adjacent element. s4_e5 sync/sink is genuinely hard on this deck; deprioritize over generalization work (Phase C).

## 2026-06-27 — lexical fusion hypothesis REFUTED; real cause was STT word-split (s3_e5)
Goal: recover s3_e5 ("Offline" cell, slide 3) — the one element bge-small misses. The plan was "lexical fusion": add an additive bonus in `MatchPolicy._ranked.effective()` when a deck phrase appears verbatim in the STT tail, to break the near-tie with s3_e4. A synthetic probe (hand-built 8-word windows over the *script text*) seemed to confirm s3_e5 clears threshold (raw 0.71–0.76) and loses a gap-tie → "reachable, implement a ~0.04 bonus."
- **The advisor blocked it on three checkable gaps**, all of which proved decisive: (1) my "wins" math skipped the policy's `gap≥0.03` confidence check; (2) the same-group stay_bonus assumption was unverified (s3_e4/e5/e6 each have their OWN group → NO stay_bonus ever applies, so s3_e5 should emit with ZERO bonus); (3) the synthetic windows aren't what the live `buf.tail(8)` actually samples.
- **Instrumenting the REAL slide-3 live run** (deterministic on the fixed WAV) exposed the truth: the STT transcribes "offline" as **"off line"** (two tokens). The token "offline" never appears in any tail → s3_e5's phrase `["offline"]` can't match lexically AND its embedding stays below s3_e4, which holds active the whole row. A lexical bonus on "offline" would do literally nothing — the token isn't there.
- **The correct fix was the existing deterministic layer**: one hand override in `data/aliases.json`, `"off line": "offline"`. Recovers s3_e5 in correct order (s3_e4→s3_e5→s3_e6), recall 37→38, flicker unchanged at 2, precision unchanged (1 pre-existing false on s6). Met every pre-committed bar.
- **Lesson:** a synthetic probe over *clean script text* is NOT the live trajectory — STT segmentation (compound-word splits like offline→"off line") is invisible to it and is often the actual root cause. Before designing a re-ranking mechanism to break a near-tie, instrument the real pipeline and read the actual tails; the fix is frequently upstream (normalization) and far simpler than the mechanism you were about to build. Also: always run the policy's own confidence math (threshold AND gap), and verify group membership, before concluding "reachable."

## 2026-06-28 — OCR fallback for baked-in image text (tesseract): precision-gated
Added an offline OCR step so a picture with text rendered INTO the raster and no alt text becomes pointable (`parse_deck(deck, ocr=True)` → tesseract subprocess, TSV). Self-contained Phase-C win.
- **Determinism scope = the ENCODER precedent, NOT the MP4 one.** OCR output feeds trigger_phrases → embeddings → matches → highlights, so it's a *pinned build-time input* in the same class as bge-small and the soffice PDF layer — reproducible only per (tesseract version, eng.traineddata). The soffice/PyAV "per-backend is fine" permission did NOT apply: that held only because the MP4 was terminal and fed no decision loop. tesseract + traineddata now belong in the dep-manifest gap the optimization audit already tracks ([[smartppt-optimization-audit]]).
- **Verify PRECISION, not just recall — OCR's real risk is hallucination.** A true-positive test ("NET REVENUE survives") proves nothing about the failure mode that matters: OCR emits spurious high-conf tokens on text-LESS but visually structured images → a false pointable that lights up (the exact thing that got hotword biasing and lexical fusion rejected). Built a NEGATIVE corpus deck (geometric shapes, no text) and it caught it: `--psm 6/11` hallucinated "OS" at conf 64; `--psm 3` emitted nothing. Filter = conf≥60 AND ≥3 alphabetic chars (the ≥3-alpha rule kills "OS" even under psm 6). A solid-fill image would have proven nothing — shapes are what make tesseract hallucinate.
- **known_gap would have become a lie.** The deck was marked `known_gap` ("no OCR"); once OCR landed via a *gated step the parse-only fast test can't run*, that marker would permanently assert a gap that's actually fixed. Resolution: a `requires_ocr=True` deck flag — the fast no-silent-drops test SKIPS those, a slow OCR test asserts survival. Retired the known_gap.
- **Landmine:** threading a new positional arg through `_parse_slide` breaks the monkeypatch stub in `test_parse_deck_isolates_a_failing_slide` (`def boom(slide,i,w,h)` → TypeError). Made `ocr` keyword-only and updated the stub to `*, ocr=False`. Don't weaken the isolation test to "fix" the failure.
- **Lesson:** any feature that injects text into the match decision loop must clear the precision bar with a negative test BEFORE shipping, exactly like the levers that were rejected for failing it. `--psm 3` is the robust default but is under-determined by one centred label (scattered diagram labels may want psm 11) — noted as a knob, not tuned to the single image.

## 2026-10-01 — eval pacing was wall-clock coupled; Piper is not bit-stable
- `Partial.timestamp` defaulted to `time.monotonic()` and `RollingBuffer` prunes its 4 s window by it, so the audit HAD to replay audio in real time. Stamping partials with audio-time (opt-in `SherpaEngine(audio_clock=True)`) + unpaced `FileSource(chunk_ms=100)` reproduced the real-time trajectories byte-for-byte, 8× faster. Lesson: before parallelising a slow eval, find what couples it to the wall clock — one injected clock beat the whole parallel-runner design.
- Piper TTS output differs run-to-run (VITS noise). Narration audio is therefore a cached artifact keyed on content (sha of segments+gap+voice), not something to regenerate per run — deleting `out/bench/*/audio` can shift scores slightly. Compare variants on the SAME cached audio.
- Auto-narration reads only the XML truth parse (`ocr=False`); narrating from the index under test would score an OCR/VLM hallucination as a hit.

## 2026-10-01 — multi-scale tail: the 3-5 word band needed per-phrase windows, not a short override
Problem: 1-2 word mentions ("Stage", "8 GB") rarely lit — the 8-word STT tail is diluted by lingering older words (title "Pipeline Stages" 0.886 beats cell "Stage" 0.763 while "stage" is being said).
- **Record once, replay many.** The policy only ever sees `buf.tail(8)`, so recording per-update buffer texts (2.5 min of STT) and replaying policy variants with a cached encoder made each 3-deck sweep ~30 s instead of ~4 min, and reproduced the harness baselines exactly. Final numbers still certified on the real harness.
- **Split recall by phrase-length band.** Every "long tail primary + 2-word override" variant fixed 1-2w but left 3-5w flat (167→170) and plateaued at ~490 hits; half of tail-4's gain was in 3-5w. Different bands need different scales → score each trigger phrase against its own window `max(4, n_words+1)` of the tail (`Matcher.scored_tail`). That alone took 6+ word recall 193→209.
- **Fleeting mentions kill stickiness.** stickiness=2 / "sustained N updates" cut recall hard (mentions last ~1 update). Dropping the in-progress last word (fragment hypothesis) barely moved flicker (71→65). What worked: a revert guard — the element a switch just left can't come back for 8 steps — flicker 54→20 with no hit cost.
- **Result** (short_words=2, short_threshold=0.73, revert_guard=8): bench hits 442→519/741, 1+2w 82→134/255, 6+ 193→199/225, false 0, flicker 31→20; hand fixture 38/40, false 1, flicker 2→1. Plateau: Ts 0.71–0.75 and guard 6–9 all pass the gate.
- **Live = bench:** the multi-scale logic lives in `MatchPolicy.step` (after normalization) and `timeline_audit.record_trajectory` now runs through `LiveController.feed`, the same path the viewer and simulators use.

## 2026-10-01 — filler precision: the bench only measured narration that NAMES things
- **Measurement gap.** Every eval narrated deck elements, so `false` could only count wrong-element highlights while
  the presenter was naming something. Nobody measured talk that names nothing ("as you can see here, this is really
  important"). A negative pass (filler only, expected `[]`) showed most slides lit something: holdout 20 false (old)
  → 36 after the multi-scale tail. Ablation: the 2-word override adds ~1 (off 35, Ts 0.78 35); the per-phrase windows
  add the rest, because a 4-word window of filler resembles a short label more than an 8-word one does.
- **Diagnosis decided the mechanism.** Filler raw scores sat at 0.65–0.71 (true mentions: median 0.73–0.80) and
  spanned all label lengths (1w 30 / 2w 16 / 3w 10 / 4+ 18 of 74 switches). So a "≤2-word needs lexical support" rule
  would have left ~40%, and a global threshold of 0.68 cost ~10% of true switches.
- **Mechanism: per-element null floor.** `Matcher.null_floor[id]` = the element's best cosine to a fixed list of
  generic presenter lines (`NULL_SPEECH`); `MatchPolicy` requires raw ≥ max(threshold, floor + `null_margin`) on both
  the full-tail and short-override paths. "When"/"2"/"Display" sit ~0.6 from any sentence, so they get a higher bar
  than specific labels. One parameter, no new model, one extra encode at load. `NULL_SPEECH` is a decision-loop
  input like aliases: changing it shifts every bar.
- **Tune/holdout discipline.** Margin picked on FILLER_TUNE + bench (0.12: tune 9; passing window 0.10–0.13; 0.14
  drops bench hits to 504) BEFORE recording the holdout. Holdout at 0.12: 5 (old 20, current 36). NULL_SPEECH shares
  phrasing with FILLER_TUNE ("look at this", "next slide") — tune is optimistic. That's why the holdout shares no
  content word with tune, NULL_SPEECH or any deck (unit-tested).
- **Cost:** bench hits 519→512/741, 1+2w 134→131/255, 6+ 199→198/225, flicker 20→15, false 0; hand fixture unchanged
  (38/40, false 1, flicker 1). Residual: "2" lit by "a couple of things" (0.73–0.74) — semantically fair, left alone.
- **Lesson:** a precision metric that only runs while the target is being named is blind to the most common real
  failure (talk that names nothing). Every recall lever needs a negative pass, and a held-out filler set, before it
  ships.

## Pointer behaviour under natural speech (2026-10-02)

- **Adjudicate before tuning.** presenter_bench now also scores false ids / false seconds with tolerated ids as
  acceptable. Baseline adj: AI_Handout 13 ids / 20.4 s, Open-Q 24 / 37.5 s — strict (30 / 69 s) overstates the
  problem ~3x.
- **Lateness median rewards hops.** 6/40 AI and 12 Open-Q baseline "hits" lit *before* their sentence started
  (accidental early lights credited by the ordered walk). Any filter that removes hops raises the deck median even
  when every real switch is only one update (+0.30 s) later. Report the paired per-step delta next to the median.
- **Switch-count and fast-succession recall are one tradeoff here.** Every confirmation rule tried (stickiness 2,
  k-of-n, score-conditioned instant path, confirm-only-short-override, confirm-only-while-active-supported,
  stable-last-word tail) either kept AI switches/step > 2.0 or broke the hand fixture (38 → 31–37) and Open-Q recall
  (91 → 59–68). Hops and legit single-update wins have overlapping score/gap/support distributions — no feature
  separated them. Don't re-run that sweep without a new signal.
- **Furniture is a free win, decided from the deck itself:** running text on ≥ half the slides or own page number,
  in the top/bottom 15% margin, titles exempt. Marked unpointable post-load (cache untouched).
- **Harness hygiene:** replay scripts that `os.chdir` into the repo write their outputs there — always write to an
  absolute scratch path.

## Card-level pointing (2026-10-02)

- **Eval speed: find the dominant cost before parallelising.** 79% of replay was the bge-small encoder. `--device cuda`
  (eval only) + one spawned process per deck: presenter 353 → 106 s, verbatim+filler 791 → 226 s, with per-slide
  trajectories identical to sequential CPU on every deck. CUDA kernels aren't bit-identical, so non-CPU encoders get
  their own embedding-cache key. `deck_bench` rebuilds `<variant>/` unless `--reuse-index` — never forget it on a
  shared bench tree.
- **Point at what the audience sees.** 80–85% of presenter switches went between parts of one card. Grouping
  elements into pointable units at load (stack / picture+caption / card panel; titles, charts, tables never) and
  letting the policy compete units took AI_Handout 2.19 → 1.88 switches/step with lit and false seconds unchanged
  or better — the intra-card hops are gone. A panel holding several multi-element units is a board, not a card.
- **Unit level changes which policy knobs pay.** The rest of the gate (≤ 1.6) came from policy knobs re-tuned on
  AI: gap 0.03 → 0.035, stale_release 3 → 0 (a released card was re-lit moments later, ~1 in 10 switches), and
  clear_after_n_low 8 → 16. Holdout Open-Q paid for holding longer: adj false s 27.0 → 46.2. The remaining false
  switches are mostly kicker labels and slide titles — next lever, not a grouping problem.
- **Score at the level you point.** Map expected ids AND highlights to units; collapse consecutive same-unit steps
  but credit every original step, so lit stays on the old ruler (48/117/40/741) and stays comparable.

## Using the slide map's meaning (2026-10-02)

- **A phrase pile poisons the null floor.** The null floor is a per-element MAX over its phrases, so one generic map
  talk phrase ("the system i want to show you", "what you get at the end") raised the bar of the whole unit — 78
  elements' floors rose (s1_e1 0.56 → 0.83, bar 0.95 = unlitable). That one mechanism produced the jumpy misses and
  most of the new "flicker", which was A → None → A (every map-variant clear was a stale release), not A → B → A.
  Before adding phrases to an element, check what they do to its floor, not just to its score.
- **Keep only specific map phrases, decided at load:** a map phrase (talk, terms, summary) joins its anchor only if its
  NULL_SPEECH similarity ≤ the anchor's own text floor + 0.03 and it sits closer to the anchor's text than to any
  other unit's text on the slide. Title / section-label units get no talk (their talk is slide-intro speech said
  without pointing). Summaries pass the same filter. Result (orig, jumpy): lit 45/48, 58/61; false s 34.1, 21.8
  (default 40.8, 25.1); flicker 12 (default 9; gate ≤ 11, missed by one), 8.
- **Replay harness:** STT is identical across index variants (sherpa ignores hotwords), so cache the STT tails once and
  replay only the policy — about 40 s per variant instead of a full bench, and it reproduced the bench exactly.
- **Know the noise band before you tune.** Near the locked point, orig flicker moved 11 ↔ 12 and sw/step ±0.02 with
  any small change (slack ±0.01, map-phrase weight 0.97/0.98/0.99). The one setting that passed every cell
  (weight 0.98) passed by giving back 6 s of the false-seconds win. Don't choose the setting that happened to pass;
  report the miss.
- **The holdout disagreed:** meaning lit 52/59 vs default 53 (the unfiltered map got 55). False ids, flicker, sw/step
  and false seconds all improved. The filter drops recall phrases as well as the generic ones. That costs recall on
  the speech that most needs the map (meaning-style references).

## Fix the root cause, not the phrases (2026-10-02, follow-up)

- **The filter treated a symptom.** The nearest-unit rule plus the 0.03 null filter fixed the tune scripts, but it
  removed recall phrases along with the generic ones. On the holdout, lit fell to 52/59 (s6_e13 lost every map
  phrase). The root cause was the floor rule, so fix the floor rule.
- **Each map phrase gets its own null bar.** An element's null floor now comes from its slide-text phrases only. A
  map-phrase row is scored minus (its null similarity − that floor, if positive) + MAP_DISCOUNT 0.04. It clears only
  by beating its own null similarity + null_margin. Adding phrases can't make a unit harder to light any more. On
  unmapped decks every row is text and nothing shifts, so decisions are byte-identical (hand fixture 38/40, 1, 1).
- **A per-phrase bar alone over-lights.** With no discount: lit 106 but sw/step 1.77–1.83 (default 1.745) and
  filler 3. The discount band 0.03–0.05 passes every summed gate. 0.02 fails on filler and flicker. I locked 0.04,
  the centre of the band. The null filter only touches the far tail now (slack 0.06; None is nearly the same).
- **Flow bonus off.** 0 was equal to or better than 0.03 and 0.05 in every row; few real transitions follow map flow.
- **Sum the gate across tune scripts.** ±1 flicker or lit per script is noise at 48–61 steps. Summing over the two
  tune scripts gave a gate that stable settings pass and that doesn't reward picking a lucky setting.
- Result, default → map: tune sums lit 100 → 105, false ids 34 → 26, false s 65.9 → 61.7, flicker 23 → 22,
  sw/step mean 1.745 → 1.695. Holdout (meaning): lit 53 → 57, false 22 → 19, false s 56.9 → 47.9. Holdout
  sw/step is worse (1.81 → 1.86, not gated). Filler 2 → 2.

## Local LLM decider for the clause pointer (2026-10-02)

- **Letter-scoring prompts carry a letter prior; calibrate it away.** Raw Qwen2.5-1.5B picked the last item (G) 62×
  vs 24 gold and N 43× vs 7. Subtracting the scores of an empty clause on the same slide prompt (contextual
  calibration, one cached call per slide) took oracle agreement 54% → 59% (3B: 52% → 61%).
- **Put the fixed text in the cached prefix.** Moving the question above the clause slot cut the per-call suffix to
  the clause + chat tokens: 298 → ~160 ms on CPU, and moving `terms` (ASR-style spellings) into the item lines lifted
  agreement 34% → 54%. Prompt layout mattered more than model size.
- **The previous clause hurt (−5 pts), so `use_context` defaults off.** Measure inputs instead of assuming context helps.
- **Bigger wasn't better:** 3B ≈ 1.5B in agreement (61–66% both, CPU/GPU numerics move it ±3), 2× slower on CPU.
- **`Llama.save_state()` copies the whole n_batch×vocab logits buffer (~300 MB/state).** Use
  `llama_state_seq_get_data/set_data` for seq 0 instead: ~14 MB per slide, restore ~100 ms, bit-identical scores.
- **Thread count is measured, not assumed:** i5-14400F (6P+4E) best at 12 threads (131 ms); 16 threads thrashes
  (p90 1.5 s).
