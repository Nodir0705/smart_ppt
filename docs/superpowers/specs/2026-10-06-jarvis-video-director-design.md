# Jarvis video director — design

**Date:** 2026-10-06 · **Branch:** `feat/jarvis-video` (from `master` ee771a1)

## Goal

Turn a `.pptx` into a narrated video in which slide content is pointed out and
marked like a JARVIS-style co-presenter, not a box that chases the speaker.
Effects are choreographed to the exact words of the narration.

## Decisions (from the brainstorm, 2026-10-05/06)

- **Output:** a recorded video (MP4). There is no live talk, so the whole
  narration is known before any effect is drawn.
- **Input:** a `.pptx` plus a narration script. **Claude drafts the script and
  the user edits it.** The script is read by an offline TTS voice (default Kokoro-82M, see §3).
- **Brain:** Claude, via Claude Code agents with one agent per slide, as for the
  slide maps. The file formats are model-agnostic, so a local Qwen3.5 path can be
  added later and measured against Claude (v2).
- **Look:** a clean HUD style (JARVIS/Iron-Man), readable on light and dark
  slides.
- **Director:** approach 1, a script plus a separate cue sheet. The user edits
  plain prose. The director re-runs only for edited slides. A deterministic
  renderer plays the cues.
- **Controls:** none. Everything is automatic from the script.
- **Scope of v1:** deck `AI_Handout_Checker` end to end. Open-Q is out of scope.

## 1. Pipeline and artifacts

```
decks/<deck>.pptx
 ├─ prepare  (exists)  build_index → out/bench/<deck>/map/index.scoped.json, slide PNGs,
 │                     understanding maps out/maps/<deck>/slide_NNN.json
 ├─ draft    (new)     Claude, 1 agent/slide → script/<deck>/slide_NNN.md
 ├─ (user edits the .md files)
 ├─ voice    (new)     TTS per sentence + word timings → out/video/<deck>/voice/
 ├─ direct   (new)     Claude, 1 agent per changed slide → out/video/<deck>/cues/slide_NNN.json
 └─ render   (new)     deterministic HUD renderer → out/video/<deck>/slide_NNN.mp4, deck.mp4
```

- **Change detection.** Each step writes a content hash of its inputs next to its
  output and skips work when the hash is unchanged.
  - voice: sentence text + voice model.
  - direct: script + slide map + index slide + effect vocabulary version.
  - render: cues + voice timings + slide PNG + renderer version.
- **CLI.** `python3 -m tools.video <step|all> <deck> [--only N]`.
  - `all` runs every step after `prepare`.
  - `--only` limits the run to one slide, so a single slide previews in seconds.
- **Script files live in the repo** (`script/<deck>/`). They are the user's
  editable source. Everything under `out/` is derived.

## 2. Script format — `script/<deck>/slide_NNN.md`

- Plain prose with one sentence per line. The text is spoken exactly as written:
  numbers in words, no markdown, no symbols.
- Lines starting with `#` are comments and are ignored.
- The drafter writes 3–6 sentences per slide (~15–35 s spoken), in a natural
  presenter style: it paraphrases rather than reading the slide, explains, and
  uses transitions.
- **The drafter gets:** the slide image, the slide map, the element index, and
  the previous and next slides' map messages for continuity.
- **The drafter is blind to:** existing presenter scripts, traces and
  adjudications.

## 3. Voice and word timings

- Each sentence is synthesized separately and joined with a 350 ms pause. The
  voice is `--voice <name>` (registry in `tools/video/voice.py`: `piper:<model.onnx>`
  or `kokoro:<voice id>`). Default `kokoro-heart` = Kokoro-82M `af_heart`
  (Apache-2.0, 24 kHz, own venv `~/.venvs/smartppt-kokoro`); `amy-low` (Piper, 16 kHz)
  stays available. Bench 2026-10-06 (3 slide-4 sentences): Whistle WER 13/99 against
  amy-low's 15/99, CPU RTF 0.27, deterministic. The slide wav keeps the voice's own
  rate; Whistle gets each sentence resampled to 16 kHz. Piper is seeded
  (`onnxruntime.set_seed` + a fresh session per slide): unseeded, its VITS noise made
  every run different.
- **Word times** come from Whistle (`~/models/whistle/whistle.cact`, run through
  the existing Whistle server, extended to return the engine's word timestamps) run on each
  sentence's clean audio.
  - Whistle's words are aligned to the known script words by sequence alignment
    (normalized: lowercase, punctuation stripped, digits as words).
  - A script word Whistle missed gets its time interpolated between its aligned
    neighbours.
  - Output: `timings.json` = per sentence `[{word, start, end}]` in slide-audio
    seconds.
- Whistle is deterministic (verified). Timing quality is checked in §7.

## 4. Cue sheet — `out/video/<deck>/cues/slide_NNN.json`

```json
{"slide": 4, "version": 1, "cues": [
  {"effect": "focus",     "target": "u3",     "at": {"s": 2, "w": 3}, "spotlight": true},
  {"effect": "underline", "target": "s4_e11", "at": {"s": 2, "w": 7}, "line": 0},
  {"effect": "glide",     "target": "u4",     "at": {"s": 3, "w": 2}},
  {"effect": "ring",      "target": "s10_e1", "at": {"s": 1, "w": 10}},
  {"effect": "zoom",      "target": "s5_e5",  "at": {"s": 4, "w": 1}, "region": [0.1, 0.4, 0.5, 0.3], "until": {"s": 4, "w": 12}}
]}
```

- **`at` / `until`:** sentence index `s` (0-based) and word index `w` (0-based,
  in that sentence's script words). The effect starts at that word's start time.
- **`target`:** a map unit id (`u3`) or an element id (`s4_e11`) on this slide.
- **`line`:** an index into the element's `line_bboxes`. Underline uses it;
  without it the underline goes under the line whose `line_texts` share the
  most content words with the anchor sentence (ties go to the first line).
- **`region`:** zoom only. A sub-rectangle of the target as fractions
  (x, y, w, h) of its bbox. Without it the zoom shows the whole target. The
  renderer snaps it out to every text line it touches (whole lines, so no word
  is cut); on a photo (no lines) it pads it by 4 % of the target each side.
- **`until`:** optional. Without it, effects end by these defaults:
  - focus holds until the next focus or glide;
  - spotlight follows its focus;
  - underline and ring persist until the next focus;
  - zoom lasts 3.0 s.

**Validator (`tools/video/cues.py validate`).** The renderer accepts only a
valid sheet. The validator checks:
- every target exists on the slide and is pointable;
- `line` is within `line_bboxes`;
- every `at`/`until` points at an existing script word, and `until` comes after `at`;
- glide goes from the current focus to its target;
- at most 1 focus, 2 marks (underline/ring) and 1 zoom are active at once;
- spotlight only accompanies a focus;
- a ring needs a compact target: its union bbox at most 4:1 either way and at
  most 25 % of the slide width, else "ring needs a compact target — use
  underline". (4:1, not 3:1: this deck's stat targets "15–33", "~1 min",
  "30/33", "14/15" measure 3.1–3.9:1 and ring well; the two flat swooshes of
  the first build were 42:1 and 57:1.)
- **the "why" rule:** the anchor sentence must refer to the target. That means
  it shares at least one content word with the target's text, map `talk` or
  `terms` (stemmed, stopwords removed), OR the target is the next unit in a
  `flow` relation from the current focus and the anchor word is a transition
  word (`next`, `then`, `finally`, `after`, `second`, `third`, …). A failing cue
  is reported with the reason.

**Warnings** (printed by `validate`, never an error): **dead air** — the first
cue starts more than 3.0 s after the first word of the first sentence that
refers to a pointable unit (the why rule's word match).

Invalid sheets are returned to the director agent once with the validator
errors. If the sheet is still invalid, the failing cues are dropped and the
slide renders with the valid ones; this is logged.

## 5. Director — Claude, one agent per changed slide

- **Inputs:**
  - the slide image;
  - the map (units, summary, talk, terms, relations, order);
  - the element index for the slide (ids, roles, bboxes, line_bboxes,
    line_texts, values);
  - the script sentences, already tokenized with word indices so the agent
    can't miscount;
  - the effect vocabulary with its rules.
- **Output:** the cue sheet.
- **Guidance** (`tools/video/packets.py` GUIDANCE = `video-direct.js`): Focus on what each sentence is about, with spotlight on sentences that single one thing out. Open on the subject: the first effect lands on the first sentence that refers to something. Underline the specific line when a sentence quotes or paraphrases one line of a multi-line unit. Ring a stat or number on the word where it is spoken, but only a compact target (at most 4:1 and a quarter of the slide wide); a wide or sentence-long target gets an underline instead. Glide when the narration walks a flow. Zoom when the sentence describes a detail in a photo or text too small to read. Leave slide intros/outros and pure explanation unpointed. Prefer fewer, well-timed effects to many.
- **The director is blind** to presenter scripts, traces and benchmarks.

## 6. HUD renderer — `tools/video/render.py` (deterministic)

- **Frames:** 30 fps at 1920×1080 by default (`--size 1280x720` for previews).
  The slide PNG is letterboxed onto the frame. Each frame is
  `f(t, slide image, cues, timings, index)` with no randomness and no wall clock.
- **Accent:** one HUD colour (cyan `#38d4ff`) with a 2 px dark halo and a soft
  outer glow (the halo shape Gaussian-blurred, 6 px at 1080p, cached per
  stroke), so it reads on light and dark slides. Pixel sizes are at 1080p and
  scale with the frame height. The spotlight dims to 55 %
  brightness with a 6 px feathered edge around the focus.
- **Effect animations:**
  - focus: corner brackets snap in over 180 ms, scaling from 108 % to 100 %
    with a ~5 % overshoot ease ("back": 1 + 2.2(p−1)³ + 1.2(p−1)²); each arm
    is 0.18 × the box's short side but at least 18 px (one-line targets get
    "[ ]");
  - glide: the brackets move along the straight path between the two bboxes
    over 400 ms, with ease-in-out;
  - underline: draws left to right over 350 ms;
  - ring: an ellipse 1.3× the padded box (a circumscribing one is 1.41×;
    1.12× and 1.2× cut the digits of the deck's stats), kept inside the frame,
    drawn clockwise over 400 ms with the back ease and snapping in from 108 %
    like the brackets (a sweep past 360° draws nothing new), then held;
  - zoom: a magnifier lens with a HUD frame and a thin leader line to the
    source region; it fades in and out over 200 ms. Placement: never over the
    source or a focus/glide live during the zoom. Try 2.2×, then 2.1× … 1.6×,
    and take the first scale with a spot clear of the title and every
    pointable element (12 px gap), nearest the source. If none fits, the 1.6×
    lens covers the least title area, then the least other content
    (decoration and empty space are free), then sits nearest the source.
- **Units:** an effect on a unit uses the union bbox of its members, padded by
  6 px.
- **Encoding:** the static ffmpeg (`~/.venvs/ffmpeg-static/shim`) encodes the
  frames with the slide's voice track into `slide_NNN.mp4`, and all slides are
  concatenated into `deck.mp4` with a 0.6 s pause between slides. Rendering
  draws in PIL and reuses the encode helpers from `src/simulate_to_video.py`.

## 7. Quality checks

- **Unit tests:**
  - cue validation, including each rule and the "why" check;
  - word-time alignment (missing and extra words, interpolation);
  - each effect's geometry at chosen times (pure functions);
  - change-detection hashes;
  - a small render of a synthetic slide that is byte-identical across two runs.
- **Contact sheets:** `python3 -m tools.video sheets <deck> [--only N]` writes,
  per played cue, the frames at start + 0.1 / 0.5 / 1.0 s and its last frame
  as a 2×2 half-size grid with the cue summary and its sentence underneath, to
  `out/video/<deck>/sheets/slide_NNN_cK.png` (K = the cue's index in its sheet).
- **Timing check:** for every sentence, Whistle must have aligned at least 85 %
  of the script words. Any sentence below that is reported.
- **Frame audit (agent, at the end of the first build):** render the frame at
  each cue's start + 0.5 s and judge it. Does the effect sit on what the
  sentence talks about? Is anything important covered? Is it readable? Report
  pass/fail per cue with the frame path. Target: ≥ 90 % of cues pass.
- **Determinism:** rendering slide 4 twice gives identical MP4 bytes, or, if the
  encoder is not bit-stable, identical decoded frames.
- **Human review:** the AI_Handout `deck.mp4` goes to the user.

## Out of scope for v1

- the reveal effect (hidden until said);
- the callback effect (an inset of an earlier slide);
- the local-model path for drafting and directing;
- an editor UI (the user edits `.md` files);
- recorded human voice input;
- Open-Q and other decks beyond running the same commands.
