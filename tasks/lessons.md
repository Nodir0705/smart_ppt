
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

## Clause pointer: decide per clause, point and hold (2026-10-02)

- **Segmenting into clauses fixed hopping; the decider fixed wrong ids.** The same bge-small cosine as today,
  run per clause (bi decider), cut lateness 2.59 → 1.75 s but kept 1.9 switches/step. A cross-encoder that reads
  clause and unit description together cut false ids 16/10 → 6/6 and switches to ~1.2/step.
- **Re-ranker logits are bimodal and absolute.** Unrelated pairs score -9..-12 and a named unit -5..+5 on every
  slide, so one fixed NONE bar (-7) works. The pass band was -8..-5.5; it fails at -5 on lateness, not on recall.
- **CPU cost is per token, so measure it on real calls.** MiniLM-L-6 took 387 ms/call on one thread (6-10 units x
  ~150 tokens); int8 quantization only gave ~1.5x. TinyBERT-L-2 (128 wide) took 15 ms in the live loop and passed the
  same gate. bge-reranker-base was best on quality (jumpy false ids 2, lateness 2.63 s) but took 2 s per call.
- **The local LLMs lost recall inside the engine:** lit 41-50 on the tune talks, against 46-59 for the cross-encoder.
  A tie-break hybrid (LLM only when the cross top-2 are close) never beat the cross-encoder alone.
- **Put the transcript state on the object that travels.** Every consumer hands `controller.buffer` to
  `LiveTranscriber`, and the viewer swaps in a fresh buffer per slide. The cumulative words and the pause boundaries
  therefore live on `RollingBuffer`, and `LiveTranscriber` shows every chunk to it before STT. No consumer can forget
  the hook.
- **The replay harness must reproduce known rows before it ranks anything.** `tools/clause_replay` matched today's
  map system (46/48 1.85 ... on all 3 talks) and the oracle rows exactly. A first parity check against my faster
  sweep loop caught an off-by-a-partial bug: the pointer was created after the first partial, which dropped that
  partial's words.
- **Don't re-run a pipeline stage on its own output.** Re-applying `apply_slide_maps` to the scoped index changed
  `map_phrases` (scope had added a trigger phrase afterwards). Patch in only the new key and diff the result.
- **Holdout kept honest.** I locked params on original + jumpy, committed them, and only then ran meaning:
  57/59 (=), false 11 (19), 1.28 sw/step, dwell 5.4 s, lateness 2.47 s (2.68). All gates passed.

## Wrong topic jumps: jump gap, title word gate, position terms (2026-10-03)

- **Replay the pointer on recorded inputs, not on audio.** Record what `ClausePointer.update` sees on every feed
  (t, words, boundaries) once (13 s for both decks) and cache decider scores by (slide, unit descriptions, clause).
  After that a policy variant costs about 2 s for all 4 tune runs instead of minutes, and the baseline matched
  `clause_replay` and the bench row for row. Check that match before ranking anything.
- **Damping buys precision with recall, and lit is gated per run.** In-clause EMA (alpha 0.3-0.6), a raised global
  gap, a higher NONE bar, lexical-evidence switching and titles-never all cut wrong moves 20-45%, but each one lost
  6-20 lit. In this data a lost hit is usually a short correct stop that damping smooths away.
- **Gate only the move that causes the jump.** Most wrong moves were new-clause commits away from a box that was
  already lit. A separate `jump_gap` (2) for that case only, with first lights left at gap 1, kept every run's lit
  and lateness within 0.08 s of the old values. The global gap 2 had cost OQ +0.19 s lateness.
- **Titles: gate on the title's own words.** A title unit is a candidate only when the clause contains a word of
  4 or more letters that no other unit on the slide has. With 3-letter words allowed, "so **how** does it actually"
  lit "How does it work?". Titles-never cost 5 lit; the word gate costs 0 and cut title false seconds 105 -> 50.
- **Position terms ("on the right") add recall:** +3 to +6 lit for about +7 false s; they cover the lit the gates cost.
- **A bigger teacher would not fix what is left.** bge-reranker-base at matched lit scored the same OQ false seconds
  (318 vs 309), so I did not try distillation. The remaining OQ false seconds are mostly "other" (158 s: a sibling
  entity named in the same sentence, e.g. "reads the key over I2C" -> I2C controller), units linked by part_of /
  explains to the referenced one (49 s), and sentences with no `refers_to` (49 s, several plainly name a unit:
  "Battery life and performance are the usual tradeoff" next to Battery Life / Performance cards). Those need
  human adjudication (see `candidate_adjudication.json` in the diagnosis scratch), not tuning.
- **ASR aliases from map terms: no.** None of the garbles ("esa memu", "codak", "linox") appear in any map's terms.
- Result, 4 tune runs summed (before -> after): lit 291 -> 291 (each run equal), false ids 82 -> 63 (-23%),
  sentence-level wrong moves 117 -> 95 (-19%), false s 364.6 -> 304.1 (-17%; AI 36.1 -> 15.0, OQ 328.5 -> 289.1),
  sw/step lower on every run. **The 25% gate missed on both counts, because Open-Q's floor sits above it.**
  Holdout (meaning x2): lit 173 -> 173, false ids 34 -> 26, false s 116.0 -> 83.5, lateness 2.49 -> 2.45.
- **Check the adjudicated columns too.** Open-Q original has a human adjudication. On it, false adj went 14 -> 16 and
  false s adj 48.9 -> 56.8 (AI unchanged at 1 / 0.3). The adjudicator had tolerated intro title lights. The title
  gate removes those, so it gains nothing on the adjudicated measure, and the unit that lights in the title's place
  is sometimes an error. Position terms add two of those errors: "on the right are the four ..." lights the
  right-hand card. Without position terms it is 16 / 51.5 and costs 3 lit. The raw false-seconds win is partly
  tolerated title time.

## 2026-10-03 — blind whole-map refinement REGRESSED (maps_v2 not adopted)
A per-slide "review and refine" pass (43 agents, blind to scripts: merge duplicates, presenter-angle rewrites, exclusive terms first, drop generic phrases) made both decks WORSE on the clause pointer: AI tune false s 15.0→29.5, jumpy lit 59→56; Open-Q original lit 94→89, meaning false s 60.9→93.7, flicker 2→12. The refined maps carried ≈2–3× more `terms`/`talk` per slide and changed unit partitions on 3 AI + 5 OQ slides; losses concentrate on the re-partitioned slides, and false time moved between slides rather than disappearing.
- **Lesson:** "more and richer descriptions" over-trigger a cross-encoder that scores short noisy clauses; a map rewrite must be MINIMAL and targeted (one failure class at a time, ablated), not a wholesale re-authoring. Keep v1 maps as the baseline; test candidate edits as hybrids (partition-only, exclusive-terms-only, summary-only) before adopting anything.
- Adjudication of 27 "non-pointing" sentences that plainly name a unit: 21 tolerated → Open-Q original adjudicated false s 56.8→39.9 (measurement only; strict ruler unchanged).

## 2026-10-03 — maps_v2 ablated by component: nothing adoptable (out/maps unchanged)
Hybrids of v1 + v2 (H1 partition-only, H2 exclusive-terms-only: ≤3 v2 terms that no other unit on the slide uses, prepended; H3 summary-only; H4 v2 trimmed to terms[:3]/talk[:5]), 4 tune runs each, same win rule (false ids AND false s lower per deck, lit within 1 per run, lateness within 0.15 s). None won on either deck; H5 (per-slide partition pick) was skipped because H1 beat v1 on no re-partitioned slide on both tune scripts (AI 5/7 and OQ 6 got worse, OQ 5 was mixed: −1.3 s original / +1.8 s jumpy, AI 1 and OQ 18/21/23 did not change).
- **The closest was H2** (AI identical to v1 on lit/false/false s, sw/step +0.03; OQ false ids 57→55 but false s 289.1→295.5: slide 20 +9 s, slide 12 −5 s). It is not the minimal edit it was meant to be: exclusivity against the other v1 units' text/talk/terms let nearly every v2 term through (3 added on almost every unit; AI 230→409 terms, OQ 826→1404). A stricter filter (summaries too, or v2's own siblings) was not tested. **Partition merges cost false s wherever they changed anything** (AI 7 +3.8 s, OQ 6 +15.4 s). **v2 summaries alone (H3) lose lit** (AI 46/59→44/58, OQ jumpy 92→88) and move false time between slides (OQ 3/6/10 −6..−10 s; 11/20/28 +9..+13 s).
- **Truncation is not the mechanism.** The cross decider cuts the unit doc at 256 tokens (clause + summary | text | talk | terms, terms last), but only 13/287 v2 Open-Q units and 0/84 AI units exceed ~230 tokens, and H4 (trimmed back to v1's size) is the worst hybrid on lit (OQ 88/117, 83/124). The richer wording itself re-ranks units; "exclusive terms first" reached the model and still did not help.
- **A per-slide pick of v2 summaries would show about −41 false s on tune** (H3 on OQ 3/6/8/10/12/13/14/15, lit +1 on 8/10/13), but that is 8 free choices fitted on the tune set; it is only worth testing against the meaning holdout, and was not built.
- **Bench on CPU reproduces the cuda rows exactly** (all 4 v1 tune runs row-for-row), so `--device cpu` is the fallback when another job holds the GPU (cuda OOM'd with the VLM worker resident).
- **Per-slide false seconds come out of `trace_<variant>.json` + the manifest wav lengths** with the `false_seconds` rule; the reconstruction matched every report total before it was used for deltas. `report_presenter_<variant>_<script>.md` holds one deck per `--only` run, so copy it after each run. Hybrid indexes built with the deck path relative to the repo (`decks/<deck>.pptx`) are byte-identical to `map/` on index.json / index.refined.json; an absolute path changes only the `deck` string.

## 2026-10-03 — Sound repair + card lever + stricter bar: tune gate nearly met, holdout failed on false ids
Three levers built in order on the clause pointer (commits 7e1c571, 8c9392d, 52f27ac), each measured with the cached-decider `clause_replay` on the 4 tune runs, then the meaning holdout scored once after locking.
- **Repair by sound, not by spelling.** Per slide, map words get CMU phonemes (`data/cmudict.txt.gz`), letter names for acronyms (plural LEDs = letters + Z) and a rule fallback. Homophone spans of dictionary words need exact phonemes (colonel -> kernel, knowed -> node); spans with an unknown word or a spelled letter may be 0.25 off; spelled letters only match acronyms ("u sees" -> uses was the main false repair before that rule). Script-checked precision 332/345 (0.96), recall 308/864 misheard slide words, 0 correct words lost; p90 0.04 ms per update. The rest is "g u" for GPU and deletions, which no sound match can fix.
- **A correct repair still costs false ids on the strict ruler.** At the old bar it bought Open-Q tune lit +13 for false ids +10, false s +30, sw/step +0.16: the decider now lights units on sentences that name them but carry no `refers_to` (slide openers: "and finally the display node", "next the kernel which is the heart"). Repair pays for a stricter bar (commit 0 -> 1, jump_gap 2 -> 3); without repair that bar loses 5-11 lit and 0.25 s lateness.
- **Card lever: the gain is all `explains`.** Lighting the part_of / explains target when it is within 4 of the winning child cut Open-Q false s 319 -> 286 and lifted lit 199 -> 204 (tune, on top of the repair); part_of alone did nothing. `explains` points both ways in these maps (node -> card on s17, bullet -> node elsewhere), so "parent = larger box" is not a safe reading. Titles must be excluded as cards (many part_of/explains point at them).
- **Rejected with numbers:** opener titles (ungated title on a slide's first light) added false ids on both decks and no lit; min_words 4 gave +4 lit for +3 false ids and +0.13 sw/step.
- Tune result (orig + jumpy): Open-Q lit 186 -> 199 (+13, gate asked +15), false ids 57 -> 57, false s 289.1 -> 275.7; AI lit 105 -> 106, false ids 6 -> 5, false s 15.0 -> 14.0; sw/step <= +0.05, lateness <= +0.06 s. Bench on CPU = replay row for row.
- **Holdout (meaning) failed:** Open-Q lit 116 -> 123 but false ids 18 -> 23, false s 60.9 -> 64.3 (adjudicated 14 -> 19 / 48.8 -> 51.5); AI 57/8 unchanged, false s 22.7 -> 23.1. About 75 configs were swept on 4 runs and the fid count of 57 was fitted: card delta 3/4/5 gave +1/0/+2 false ids, a knife edge. **Lesson: when a lever trades lit for false ids, a retuned bar that zeroes the false-id delta on tune is fitting noise; expect the holdout to show the trade.** The new false ids are mostly openers naming an intro/overview unit on an unlabeled sentence, plus a few real wrong units (s11 "rpmh" -> Connectivity). Way back: revert the three commits, or `ClauseParams(repair=0, card=-1, commit=0, jump_gap=2)`.
- Gap to the ceiling after (Open-Q, all styles, 489 oracle clauses): misses 148 -> 119. ASR garble misses 29/94 s -> 11/24 s, garbled-and-clean-fails 17/52 s -> 9/24 s, child/parent 20/89 s -> 15/61 s; title gate 17/60 s and paraphrase under NONE 20/55 s -> 25/67 s remain (the stricter bar added 5 there). Wrong switches the oracle didn't make 118 -> 130 (intro 14 -> 20).

## 2026-10-03 — gap-closing levers failed the holdout → shipped OFF (defaults restored)
Sound-based ASR repair (repair 0.25) + card preference (card 4) + a stricter bar (commit 1, jump_gap 3) were locked together on the 4 tune runs (Open-Q lit +13, false ids ±0 — fitted after ~75 settings) and FAILED the held-out meaning talks: Open-Q false ids 18→23, sw/step 1.01→1.16, flicker up on 3 runs. Independent verification also found the repair's "0 correct words changed" claim false: spelled 4-letter acronyms are cut by the 3-token span cap ("r p m h" → "rpm h") and "l p d d r" fuzzes to "led" (~5% wrong rewrites, 0.3% on clean text).
- Defaults restored to repair 0 / card −1 / commit 0 / jump_gap 2 (bit-exact with the pre-round rows); the code stays for re-tuning.
- **Lesson:** three coupled levers tuned jointly on 4 runs over ~75 settings is a fitting exercise — the holdout caught it. Ship one lever at a time, each holdout-checked alone. Repair needs ≥4-token spans and an exact-letter-sequence guard before it is reconsidered; on its own at the old bar it bought lit at a high false cost (+13 lit, +10 false ids, +30 false s).

## 2026-10-05 — Offline slide maps from a local model: fix-up, prompts v2-v4, spoken terms (AI_Handout only)
Local maps from qwen3.5 via Ollama, scored with `clause_replay` (tune = original + jumpy, meaning held out). Best local
total 148 lit (both v4 finalists) against the 155 target and Claude's 162. False ids and false seconds stayed within
Claude's (4b v4: 9 / 32.9 against 14 / 37.7).
- **Fix-up beats repair for bookkeeping.** The deterministic fix-up rescued 3/12 slides for 9b and 6/12 for 4b (v1
  prompt). No slide falls back now; 4b used to lose 2. It never costs a model call, and the repair turn mostly echoed its
  input. The gain is reliability, not lit. On every fixed 9b slide the per-slide lit was identical. On 4b, jumpy slide 1
  went from 3/5 to 1/5, because there the fallback's all-singleton map beat the model's merged one.
- **Ollama's `format` grammar does not hold integer enums, required keys or minItems/maxItems.** A key skeleton with
  `"slide": 0` sent all 12 v2 slides to repair (177 s/slide). Items for arrows went missing, and talk came out with 3-4
  phrases against a schema minimum of 5. Fix-up therefore sets the slide number and fills in missing items. Put counts in
  the prose, and never write a value into a skeleton that the model could copy.
- **The partition was not the bottleneck; summary wording was.** v2 lifted pair-F1 against Claude from 0.76 to 0.91, yet
  tune lit went from 95 to 94. On slide 4 the partition matched Claude's and still lost 2 lit. The TinyBERT decider puts
  the summary first in its unit text. A meta summary ("This panel shows...", "A feature card highlighting...") scored
  -7..-10 on the clause that names the unit, while a content summary scored +3. The 9b kept writing meta summaries after
  three different wordings.
- **Prompt wording is fragile. Screen before a full run.** Asking for 6-8 talk phrases made the 9b pad with repeats (one
  phrase 4x). Asking for "key terms" made it copy slide text into terms. A `"<key words>: ..."` summary shape collapsed
  summaries into the slide text. A cheap screen: the decider margin (target logit minus best other) on the tune scripts'
  written sentences, over 4 hard slides (~6 min of GPU). Use it to rank prompts. It is not a replacement for lit.
- **Richer talk buys lit with false ids.** 9b v4 (560 talk phrases, close to Claude's 608) had tune lit 96 against 94-95,
  but false ids 7 against 3-5 and false s 23 against 7-19. This is the same over-trigger seen with maps_v2.
- **Spoken terms belong in code, not in the prompt.** Both models ignored "no digits" (11-25% of terms kept them).
  `speak_terms` brings that to 0%, and tune lit and false ids did not change.
- **Byte-determinism holds only on an idle GPU.** Isolated runs of 4b reproduced 4/4. The 9b reproduced 2/3 on slide 2,
  diverging at a near-tie: close the talk list, or add a phrase. Slide 5 reproduced 3/3. Runs that shared the GPU with
  another CUDA process, or with an aborted Ollama request still decoding, changed. Generate the maps once and keep them.
  Do not count on a regeneration matching.
- **Noise floor:** two 9b runs of the same prompt moved jumpy slide 2 by 2 lit, through wording drift alone. The v1 run
  was unpinned, and the rerun shared the GPU with a scoring job, so the cause is not established. A tune difference of
  3 or less between variants is not a result.

## 2026-10-05 — Whistle STT and sherpa hotwords: better transcripts, no pointer gain (AI_Handout only)
Built `WhistleEngine` (Cactus Whistle in a subprocess; VAD segments end at 150 ms of silence, cut at <= 5.12 s,
re-decoded every 0.5 s of audio) and `SherpaEngine(hotwords_score)` (commit 0741ee2). All streams were recorded
from the same input (the wav + 1 s of zero PCM) and replayed with Claude maps (`map`). Tune = original + jumpy.
| engine | WER / slide terms | tune lit | false ids | false s | sw/step | late med | holdout meaning lit · false · false s |
|---|---|---|---|---|---|---|---|
| sherpa greedy (default) | 7.99 / 84.9% | 105/109 | 6 | 15.0 | 1.12, 1.09 | 2.35, 3.06 | 57/59 · 8 · 22.7 |
| sherpa hw 1.5, per-slide terms | 7.68 / 88.4% | 106 | 6 | 15.0 | 1.15, 1.11 | 2.35, 2.95 | 57 · 9 · 25.6 |
| sherpa hw 3.0, per-slide terms | 7.80 / 89.7% | 105 | 6 | 15.0 | 1.15, 1.09 | 2.32, 2.98 | 57 · 9 · 25.6 |
| sherpa hw 1.5, deck-wide terms | 7.59 / 87.9% | 106 | 7 | 18.2 | 1.17, 1.11 | 2.35, 2.95 | 57 · 9 · 26.6 |
| whistle | 5.02 / 92.7% | 105 | 9 | 28.3 | 1.19, 1.12 | 2.02, 2.47 | 57 · 12 · 29.2 |
- **WER is not this pointer's bottleneck.** Whistle cut WER by 37% and term misses by half. Lit did not move on any
  talk. Whistle's words arrive earlier, so lateness improved 0.33/0.59/0.45 s, but the first 5-word decision now
  lands on a fragment: "on the left is a" lights the ✓ card through its position term, and "now scale it up to"
  lights the title. Or it lands on an intro sentence that names a unit, which the strict ruler counts as false.
  False ids went +3 on tune and +4 on holdout.
- **Hotwords lift term accuracy but not the pointer.** The +1 tune lit at 1.5 is inside the noise floor, and the
  holdout gained 1 false id and 2.9 false s. Per-slide terms are the best case. Live passes one deck-wide list
  (per-slide terms would need a stream restart at each slide change), and that list was worse on tune: +1 false id
  and +3.2 false s. The WER rows match the offline bench exactly, so the option is the bench config.
- **Replay drops partials stamped after the wav ends**, so trailing padding changes WER only. Re-recording greedy
  sherpa with the pad gave the old streams' partials on all 36 slides. That check proves the default path did not
  change. It also explains why the old file's WER is 8.41: it lost the tail words.
- **Audio-clock stamps hide decode time.** Whistle's paced live lag is p90 71 ms. Shifting its stamps by +0.1 s
  raised lateness by 0.1 s, cut false ids by 1, and left lit unchanged. It is still 0.2-0.5 s earlier than sherpa.
- **Live cost (paced, with the cross decider, Ollama idle):** Whistle keeps up with real time. Emit lag was p50 38,
  p90 71, max 110 ms (sherpa 16/17/22) and did not drift. The first partial came 0.49 s after speech onset (sherpa
  0.69 s). It averaged 1.0 core (sherpa 0.1) across 12 server threads that burst over every core.
- **Next:** Whistle's lateness headroom could pay for a stricter first decision (min_words / jump_gap). The AI
  meaning holdout has now been seen for Whistle, so that test needs a fresh holdout.

## 2026-10-05 — Local maps: the summary was not the bottleneck, talk is (AI_Handout only)
Derived map sets from the qwen35_4b_v4_sp maps (`local_slide_map derive`, no regeneration), scored with `clause_replay`
on tune (original + jumpy). Baseline 96 lit / 4 false ids / 17.6 false s. None of the variants beat it, and none
reached the 155 / 3-talk target.
- **Blanking every summary changed nothing** (96 / 4 / 17.6). A patched decider that drops the summary from the doc,
  so the validator never sees it, gave 95 / 5 / 18.6. Summary := the member texts gave 93 / 5 / 19.5. The model
  rewrite gave 95 / 5 / 20.8: one content sentence per unit, 69/74 accepted, 0 meta summaries left. The brief's
  diagnosis ("the summary is read first, so meta summaries sink the clause") did not hold up when measured.
- **The swap test locates it.** On the slide 4 command unit, for "it's just one command, pointed at the scanned
  handout": Claude summary + local talk scored −6.9, local summary + Claude talk scored +6.5. The talk carries the
  score. Claude's talk says what you DO with the unit ("you point it at the pdf", "you just run one script"). The local
  talk names its kind ("the command", "the code", "the syntax").
- **A context-free talk rewrite makes things worse.** The 4b got only the unit's text and invented meaning for short
  units ("8 GB / GPU" → "your machine will feel like a true gaming beast", "14/15 tense-chart page" → "this diagram
  helps us see how things change"). Tune went to 92 / 7 / 29.8. A talk pass needs slide context or the image, and the
  4b may simply be too small for it.
- **The gap to Claude (tune 105 against 96) is mostly partition.** Slide 1 −4 (e2+e3 merged), slide 7 −2 (e4+e5
  merged), slide 4 −3 (same partition as Claude, so the loss is talk), slide 3 −1, slide 8 +1.
- **Talk cleanup is neutral and safe.** Digits spoken, in-unit duplicates dropped, phrases on 3 or more slides dropped
  (3 of them: "how it works", "the first step", "the main topic"), refilled up to 3 twice. All 3 talks scored
  identically (148 / 9 / 32.6 against 32.9), and digits in talk went from 22 to 0.
- **Under `think=false`, qwen3.5:4b ignored a small `format` schema** and answered in plain text. Every summary was
  rejected until the parser accepted plain text. Use plain one-per-line output for short answers.
- A meta-word check needs an own-text exemption. "a consumer card", "instantly shows" and "phone photos" are content.
- Both rewrite passes reproduced byte-identically (2/2) on an idle GPU: summary 3.8 s/slide, talk 8.5 s/slide.

## 2026-10-05 — First-decision timing per STT engine: Whistle's head start does not buy precision (AI_Handout only)
Swept min_words / min_content (new, default 0 = off) / gap / jump_gap / commit with cached-decider replay on tune =
original + jumpy + meaning (Claude maps), rule pre-registered before any sweep row: max lit s.t. false ids <= 14 and
false s <= 37.7 (sherpa-default sums), sw/step <= sherpa + 0.05 per talk, all +-1-step neighbours passing, fallback =
defaults. Then the sealed `casual` talk once. Nothing adopted; no default changed.
- **One lever at a time, Whistle (162 lit / 21 false ids / 57.4 false s / mean late 2.16 s):** min_words 6-7 → 19
  false ids, 8 → 17 (161 lit, late 2.93); commit 1.5-2 → 18-17 (161 lit); min_content 2 → 18, 3 → 17 (160 lit,
  late 2.52). gap and jump_gap hardly move false ids (jump_gap 3.5: false s −1.9). Lit barely moves on any lever.
- **Combined, inside the grid (576 configs): 0 pass C1 or C2.** Frontier 161 / 16 / 42.4 at late 2.27 s
  (min_content 2, jump_gap 3.5, commit 1). The gates remove about 5 of Whistle's 7 extra false ids, not all.
- **Past the grid (exploration, not lockable): matching sherpa's false bar costs the head start.** min_words 8-9,
  commit 1.5-2, min_content 2-3 reach 9-13 false ids / 29-37.5 false s, but at 154-157 lit and mean late 2.96-3.08 s,
  later than sherpa-default (2.62). Whistle's 0.3-0.6 s lead is worth less than its extra early fragments cost.
- **min_content works as intended on fragments** ("on the left is a" now holds) and is plateau-safe for sherpa (mc 2
  at min_words 6: 12 false ids / 33.1 s, lit 162), but it is not needed for any locked row; it ships OFF.
- **Plateau artifact on sherpa:** the rule picked min_words 6 only because the defaults' min_words-4 neighbour fails
  (17 false ids). On tune it ties on lit, −1.4 false s, +0.25 s late. Casual: 52 / 5 / 13.2 against 52 / 6 / 13.8,
  late 2.56 against 2.26. Not a gain worth +0.3 s.
- **Casual (sealed, scored once): Whistle-as-is 55/56 · 5 · 18.2 s · sw 1.20 · late 1.95 against sherpa 52/56 · 6 ·
  13.8 s · 1.06 · 2.26.** It fails on false s, so not adopted. +3 lit is at the noise floor; the extra false time
  and sw/step are the same pattern as on tune. The casual verdict tested Whistle with defaults (no eligible config),
  not the stricter first decision.
- **Method:** a dict cache on `decide` keyed by (clause, context, unit uid+text+summary), warmed once at the lowest
  min_words, makes every config ~4 s for 3 talks (0 misses afterwards); the cached default row matched
  `clause_replay` exactly. `clause_replay --styles casual` now works (SCRIPTS has it; `--record` and the defaults
  still cover only original / jumpy / meaning).
