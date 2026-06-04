# Live grounding accuracy — Phase 1 design

**Date:** 2026-06-01
**Status:** draft for review (brainstorming → spec)
**Depends on:** the matcher/STT/policy stack; the refined index (`out/index.refined.json`)

## Goal

Improve **live grounding accuracy** — which element gets highlighted as the
presenter speaks — fixing the three failures measured in the 2026-06-01 timeline
audit, while keeping the runtime **fully offline, deterministic, and CPU-only**
(the GPU is currently unusable: `nvidia-smi` reports a driver/library mismatch).

No generative model runs in the live loop. No cloud. No script/cue-sheet
(usage is improvised/live). All Phase 1 intelligence is either deterministic or
precomputed offline without a neural model.

## Measured failures this must fix (from the headless audit)

Locations (bboxes) are already correct everywhere; the defect is *element
selection over time*. Audit verdict: slides 1, 2, 3, 5, 7 correct; slides 4 and 6
wrong.

| # | Failure | Evidence |
|---|---|---|
| 1 | Acronym mis-recognition | "STT" heard as `"sgt"`/`"s g t"`; the STT latency bar (`s4_e2`) and STT Engine node (`s6_e1`) **never** highlighted |
| 2 | Near-duplicate across groups | "Match is one" lit the effort-pie **Matching** slice (`s4_e8`) instead of the latency **Match** bar (`s4_e3`); "STT" drifted to **Sync** (`s4_e5`) |
| 3 | Flicker | slide-2 `e3↔e2` on a `"text"→"taxed"` mis-hear; transient blips on slides 5/6 |

## Why these happen (root cause)

The live match is **text-embedding similarity only** (`matcher.py:148-155`): embed
the STT tail, dot-product against precomputed trigger-phrase embeddings, per-element
MAX, then a threshold/gap/stickiness policy. It ignores recognition robustness,
group structure, and discourse history. So garbled acronyms match nothing useful,
and two charts with near-identical labels are decided on raw string similarity.

## Approach (Phase 1 — four deterministic levers)

### 1. ASR alias / normalization layer (fixes #1)

A deterministic, engine-agnostic text normalizer applied to the rolling-buffer
text **before** matching. It rewrites known mis-hearings and phonetic/spelled
forms to canonical deck terms.

- New module `src/asr_normalize.py`:
  - `build_alias_map(index) -> dict[str, str]` — auto-derives seeds from the deck:
    every all-caps token (acronym) maps its letter-spelling and common phonetic
    confusions to the canonical token, e.g. `STT → {"s t t", "sgt", "es tee tee",
    "speech to text"}`. Expansions are pulled from nearby element text when present.
  - `load_aliases(path) -> dict` — merges a hand-maintained
    `data/aliases.json` (override + additions the auto-derivation can't guess).
  - `normalize(text, alias_map) -> str` — longest-match phrase replacement over
    the transcript tail; case-insensitive; word-boundary safe.
- Wired into the match path (`MatchPolicy.step` / the simulator + viewer feed) so
  every spoken tail is normalized before `matcher.match`.

Deterministic, offline, CPU, engine-independent — works with Vosk today and any
future engine. Engine swap (sherpa-onnx / faster-whisper contextual biasing) is
explicitly **out of scope** for Phase 1 (Phase 2 candidate).

### 2. Author-time group metadata + scoped triggers (supports #2)

Add group structure and group-scoped trigger phrases to the index **without a
neural model** — derived deterministically from existing `parent_id` + chart/title
text.

- New transform `src/group_scope.py` (CLI like the other index transforms):
  - For each element with a `parent_id`, attach `group_id = parent_id` and
    `group_label` = the parent chart's inferred label (chart title text, else
    chart type → "latency chart" / "effort chart" via a small type/keyword map
    seeded from the chart's own text). Title/standalone elements get
    `group_id = self`, `group_label = None`.
  - Add **scoped** trigger phrases to chart-category elements, e.g.
    `"the {category} {bar|slice|point}"` and `"{category} {group_label}"`
    (`"match latency"`, `"the matching effort slice"`), so the embedding for the
    correct item sits closer to scoped utterances. Existing phrases are kept
    (additive).
- Output `out/index.scoped.json` (input defaults to `out/index.refined.json`);
  composes with refinement/enrichment like the other transforms.

### 3. Group-then-item scoping in the matcher (fixes #2)

Change scoring so item selection is conditioned on a group decision instead of a
flat argmax over all elements.

- In `src/matcher.py`:
  - Compute per-element scores as today, then aggregate a **per-group score**
    (`group_id` → max over its items' scores, including the group/chart element's
    own triggers).
  - Pick the best group (subject to the same threshold/gap idea at group level),
    then pick the best item **within** that group.
  - Return the chosen item plus the group decision in `MatchResult` (additive
    fields) so the tracker/telemetry can use it.
- Elements with no group (`group_id = self`) form singleton groups, so behavior
  for titles/bullets/diagram-labels is unchanged unless they share a group.

### 4. Group-sticky temporal tracker (fixes #3, reinforces #2)

Replace the bare stickiness in `MatchPolicy` with a lightweight transition prior
(HMM-lite) computed over the *effective* score.

- `effective_score(e) = match_score(e) + stay_bonus·[e == current]
  + same_group_bonus·[group(e) == group(current)]
  + order_bonus·[e is the discourse-next sibling of current]`
  where "discourse-next sibling" = the next ordered bullet, or the next chart
  category in reading order, within the same group.
- The existing `threshold` / `runner-up gap` then operate on effective scores;
  switching still requires beating the gap, so genuine moves still happen but
  jitter and cross-group hops are damped. After STT→Render in the latency group,
  "Match" stays in that group (fixes the #2 residue) and a one-frame mis-hear no
  longer flips the highlight (fixes #3).
- Bonuses are small, explicit constants (config on `PolicyConfig`), tuned against
  the eval below. Deterministic.

## Eval / acceptance (measurable, Karpathy-style)

Phase 1 is judged by a metric, not vibes.

- Promote the audit harness to a committed tool `tools/timeline_audit.py`
  (the headless TTS→STT→matcher→policy timeline, muted, real-time paced).
- Add ground truth `tests/fixtures/expected_highlights.json`: per slide, a list of
  `{utterance_substring, expected_element_id}` cues covering each pointable
  mention in `scripts/narration.json` (≈30 cues across 7 slides).
- **Primary metric — clause accuracy:** fraction of cues where the highlighted
  element at that clause equals `expected_element_id`.
  - Baseline: measured first on the current pipeline (record the number).
  - Target: fix the 3 specific failures (slide-4 STT bar `s4_e2`, slide-4 Match
    bar `s4_e3`, slide-6 STT Engine `s6_e1`) AND **zero regressions** on the
    cues currently passing (slides 1/2/3/5/7).
- **Secondary metric — flicker:** count of switches that revert within 2 updates;
  target strictly lower than baseline.
- A pytest test asserts: clause accuracy ≥ baseline+Δ and the three target cues
  pass and no currently-passing cue regresses. (Gated on soffice/TTS assets; skips
  cleanly when absent.)

## Components / files

- `src/asr_normalize.py` — alias map + `normalize()` (new)
- `data/aliases.json` — hand override seed (new)
- `src/group_scope.py` — group metadata + scoped triggers transform + CLI (new)
- `src/matcher.py` — group-then-item scoring; additive `MatchResult` fields (modify)
- `src/matcher.py` `MatchPolicy`/`PolicyConfig` — group-sticky effective score (modify)
- viewer/simulator feed — apply `normalize()` before match (modify; small)
- `tools/timeline_audit.py` — committed audit harness (new; from /tmp prototype)
- `tests/fixtures/expected_highlights.json` — ground-truth cues (new)
- `tests/test_grounding_eval.py` — accuracy + flicker assertions (new)

## Constraints & decisions

- **CPU-only / GPU broken:** no GPU or neural author-time model in Phase 1; all
  four levers are deterministic. Robust to the current `nvidia-smi` failure.
- **Engine kept (Vosk):** the alias layer fixes acronyms without swapping engines.
  Engine swap is deferred.
- **Additive schema:** `group_id`, `group_label`, scoped phrases, and new
  `MatchResult` fields are additive; existing consumers keep working.

## Out of scope (later phases)

- Phase 2: CLIP/SigLIP cross-modal (spoken phrase ↔ element crop) live scorer;
  Florence-2 / strong-VLM author-time grounding for images & visual deixis;
  streaming engine swap with contextual biasing.
- Phase 3: small deterministic cross-encoder re-ranker for residual ambiguity.
