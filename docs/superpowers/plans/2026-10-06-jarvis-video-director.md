# Jarvis Video Director Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** turn `AI_Handout_Checker.pptx` plus an editable narration script into a narrated MP4 with JARVIS-style HUD effects (focus+spotlight, underline, ring, glide, zoom), each choreographed to an exact spoken word.

**Architecture:** a new package `tools/video/` with pure, separately tested modules:
- `script` parses and writes the narration;
- `timings` aligns words;
- `cues` holds the cue-sheet schema and validator;
- `geometry` and `effects` hold the drawing maths;
- `voice`, `render` and `__main__` handle I/O and the CLI.

The LLM steps (draft and direct) run as Claude Code agents, one per slide, from named workflows in `.claude/workflows/`. They read packets that `tools.video` writes and write files that `tools.video` validates.

**Tech stack:**
- Python 3.10 with stdlib, numpy and PIL;
- Piper TTS (existing `src/tts.py`);
- Whistle STT through the existing venv server (`tools/whistle_server.py`), extended to return word timestamps;
- the static ffmpeg (`~/.venvs/ffmpeg-static/shim`);
- pytest.

**Spec:** `docs/superpowers/specs/2026-10-06-jarvis-video-director-design.md`, read in full. **Branch:** `feat/jarvis-video`.

**Ground rules for every task:**
- Commit only your own files: run `git add <paths>`, never `-A`.
- Never use `--no-verify`.
- End every commit message with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Y4sBDPZVumMRXYxBZCFBF4
  ```
- Never pip install into the system python.
- Work on deck AI_Handout_Checker only.
- Inputs that already exist:
  - index: `out/bench/AI_Handout_Checker/map/index.scoped.json`
  - slide PNGs: `out/bench/AI_Handout_Checker/default/slides/slide_NNN.png`
  - maps: `out/maps/AI_Handout_Checker/slide_NNN.json`
  - Index slides carry `slide["map"] = {message, relations, order, units: {uid: [ids]}, info: {uid: {summary, talk, terms}}}`. Elements carry `map_unit`, `map_role`, `pointable`, `bbox` and, for text, `line_bboxes`/`line_texts`, all in inches. `slide_size` gives `{w, h}`.

---

## File map

| File | Responsibility |
|---|---|
| `tools/video/__init__.py` | package marker, `VERSION = 1` |
| `tools/video/script.py` | read and write `script/<deck>/slide_NNN.md`; `sentences(path) -> list[str]`; `words(sentence) -> list[str]` (the canonical tokenizer, shared by every step); `norm(word)` |
| `tools/video/timings.py` | `align(script_words, asr_words) -> list[{word,start,end}]`, including interpolation for missed words |
| `tools/video/cues.py` | cue schema; `resolve(cue, slide)` gives target ids, bbox and line; `validate(sheet, slide, sentences) -> list[str]` errors, including the "why" rule; `defaults` for `until` |
| `tools/video/geometry.py` | inch→pixel letterbox transform; target box (unit union, element, line) with padding; empty-region finder for the zoom lens |
| `tools/video/effects.py` | pure functions `state_at(t, plan) -> FrameState` and `draw(frame, FrameState, style)` covering brackets, spotlight, underline, ring, glide, zoom |
| `tools/video/voice.py` | per slide: Piper per sentence (350 ms pause) → `voice/slide_NNN.wav` plus sentence starts; Whistle word timestamps per sentence → `timings/slide_NNN.json`; hash-cached |
| `tools/video/render.py` | per slide: cues + timings + PNG → frames → ffmpeg → `slide_NNN.mp4`; `deck.mp4` concat; hash-cached |
| `tools/video/packets.py` | writes the agent input packets (draft and direct) and applies agent outputs, with validation |
| `tools/video/__main__.py` | CLI: `python3 -m tools.video {packets-draft,voice,packets-direct,validate,render,all} <deck> [--only N] [--size WxH]` |
| `tools/whistle_server.py` | add `"words": true` to a request so the reply includes `[{word,start,end}]` from `transcribe(..., word_timestamps=True)` |
| `.claude/workflows/video-draft.js`, `.claude/workflows/video-direct.js` | named workflows: one agent per slide |
| `tests/test_video_*.py` | unit tests per module |

---

### Task 1: Script format + word timings (pure) + Whistle word timestamps

**Files:**
- Create: `tools/video/__init__.py`, `tools/video/script.py`, `tools/video/timings.py`, `tests/test_video_script_timings.py`
- Modify: `tools/whistle_server.py`

- [ ] **Tests first.** `tests/test_video_script_timings.py`:
  - `sentences()` reads one sentence per non-empty line, skips lines starting with `#`, and strips whitespace.
  - `words("Stage one turns pixels into text, so the printed stuff — all in one go.")` == `["Stage","one","turns","pixels","into","text,","so","the","printed","stuff","—","all","in","one","go."]`. Splitting is on whitespace only; that is the canonical word index.
  - `norm()` lowercases, strips edge punctuation, and maps digits to words (`"8"` → `"eight"`); a pure-punctuation token normalizes to `""`.
  - `align()`:
    - exact match → each script word gets the matching asr times;
    - an ASR word missing for script word k → k's start is interpolated linearly between its aligned neighbours (end = next start);
    - an extra ASR word is ignored;
    - a leading or trailing unmatched script word clamps to the first or last aligned time;
    - pure-punctuation script words (norm == "") take the time of the previous word;
    - the result has the same length as `script_words` and starts are non-decreasing.
- [ ] Run the tests and confirm they fail.
- [ ] **Implement.** `align` runs `difflib.SequenceMatcher` over the normed word lists, then interpolates.
- [ ] **Whistle server.** If the request JSON has `"words": true`, call `model.transcribe(audio, language=..., word_timestamps=True)` and return `"words": [{"word", "start", "end"}]` (seconds relative to the request audio). Check the field names with a one-off run in `~/.venvs/smartppt-whistle` on one TTS wav (`out/bench/AI_Handout_Checker/presenter/audio/*.wav`). Keep the old reply unchanged when `words` is absent, because `WhistleEngine` depends on it.
- [ ] `python3 -m pytest -q tests/test_video_script_timings.py` is green, the full suite is green, then commit.

### Task 2: Cue schema + validator (pure)

**Files:** create `tools/video/cues.py` and `tests/test_video_cues.py`.

- [ ] **Tests first**, on a synthetic slide dict: 3 units (`u1` title, `u2` card with 2 elements, `u3` stat), element `line_bboxes`/`line_texts`, map `info` (summary, talk, terms), and a relation `u2 -flow-> u3`. Sentences are given as lists of words.
  - **valid sheet:** returns `[]`.
  - **unknown target, or a non-pointable target** → error.
  - **bad `line` index** → error.
  - **`at` sentence or word out of range, or `until` ≤ `at`** → error.
  - **concurrency:** a second focus is a new focus that replaces the old one, so it is allowed. A third mark (underline or ring) active at once → error. A second zoom overlapping the first → error.
  - **spotlight on a non-focus cue** → error.
  - **why rule:**
    - the anchor sentence shares a stemmed content word with the target's text, talk or terms → OK;
    - otherwise → error, unless the target is a `flow` successor of the current focus and the sentence contains a transition word.
    - Stemming: lowercase, strip a trailing `s`/`es`/`ing`/`ed` (len > 4), and drop stopwords from a small fixed list.
  - **`resolve()`:** a unit id resolves to its member ids; an element id resolves to itself; `line` defaults to the line whose `line_texts` share the most content words with the anchor sentence, with ties going to the first line.
  - **`effective_until(sheet)`:** fills the defaults:
    - focus ends at the next focus or glide;
    - spotlight follows its focus;
    - underline and ring end at the next focus;
    - zoom lasts 3.0 s, so its end is expressed in seconds given the timings.
- [ ] Implement, run the tests, then commit.

### Task 3: Geometry + effects (pure, deterministic)

**Files:** create `tools/video/geometry.py`, `tools/video/effects.py` and `tests/test_video_effects.py`.

- [ ] **Geometry.**
  - `Letterbox(slide_w_in, slide_h_in, png_w, png_h, frame_w, frame_h)` provides `.box_px(bbox_in, pad_px=6)` → `(x, y, w, h)` in frame pixels, and `.place(png)` → the letterboxed frame image.
  - `union(boxes)`.
  - `empty_region(frame_size, occupied_boxes, want_w, want_h)` → the top-left of the largest free rectangle on a coarse 16×9 grid, deterministic, preferring the right and then the top.
- [ ] **Effects.**
  - **Style:** `HUD = dict(accent=(56,212,255), halo=(10,14,20), dim=0.55, feather=6, bracket_len=0.18, line_w=4)`.
  - **Timeline:** a list of `Effect` items, each `(kind, t0, t1, box, extra)` built from cues + timings. `state_at(t)` returns the active effects with animation progress:
    - focus brackets snap from 108 % scale to 100 % over 180 ms;
    - glide is the bracket box lerped between the two boxes over 400 ms with ease-in-out (reuse `src/geom.ease_in_out_cubic`/`lerp_bbox`);
    - underline progress 0→1 over 350 ms;
    - ring sweep 0→360° over 400 ms;
    - zoom alpha 0→1 over 200 ms in, and 1→0 over 200 ms out.
  - **`draw(frame, state, style)`** with PIL:
    - spotlight: darken everything outside the focus box by `dim`, with a feathered edge (precompute a mask per focus box and cache it per box);
    - corner brackets: an L-shape of `bracket_len`×min(w, h) at each corner, accent colour with a 2 px halo;
    - underline: a line at the line box's bottom + 3 px, drawn left→right by progress;
    - ring: an ellipse arc around the padded box (`ImageDraw.arc`, start −90°, sweep by progress);
    - zoom: a lens that crops the region at 2.2×, pasted at `empty_region`, with a HUD frame and a thin leader line to the source region.
- [ ] **Tests.**
  - `state_at` at chosen times gives the expected progress values and active sets.
  - `empty_region` avoids the occupied boxes.
  - Drawing a synthetic 640×360 frame twice is byte-identical, and a pixel inside the spotlight is brighter than one outside.
  - A 0-length effect list leaves the frame unchanged.
- [ ] Implement, run the tests, then commit.

### Task 4: Voice step + renderer + CLI (integration)

**Files:**
- Create: `tools/video/voice.py`, `tools/video/render.py`, `tools/video/__main__.py`, `tests/test_video_render.py`

**Behaviour:**
- **`voice`.** For each `script/<deck>/slide_NNN.md`:
  - Run `PiperTTS(...).synthesize_segments(sentences, wav, gap_ms=350)` to get the sentence starts, writing `out/video/<deck>/voice/slide_NNN.wav`.
  - For each sentence, cut its audio span (start to next start minus the gap) and send it to the Whistle server with `"words": true`. The server is started like `src/stt.WhistleEngine` starts it; reuse that code.
  - `align()` to the script words, then offset by the sentence start → `out/video/<deck>/timings/slide_NNN.json`, `{"sentences": [[{word,start,end}...]...], "wav": ..., "duration": ...}`.
  - Report the fraction of script words Whistle matched exactly, per sentence. Flag any sentence under 85 %.
  - Skip slides whose hash (sentences + voice model path + module VERSION) is unchanged. The hash is stored in `*.hash`.
- **`render`.**
  - For each slide with cues and timings: validate the cues (refuse invalid ones and print errors), build the effect timeline, and render frames at 30 fps.
  - Output size is `--size` (default 1920x1080). The slide PNG is letterboxed on a near-black background.
  - Frames are piped to ffmpeg exactly like `src/simulate_to_video._render_and_encode`: `-threads 1` x264, `-crf 20`, aac 128 k, with the slide wav.
  - Unchanged frames are memoized by state.
  - `deck.mp4` = concat of the slides with a 0.6 s silent gap. Generate the gap clip once by rendering the last frame of each slide plus silence, or use ffmpeg `tpad`/`apad`.
  - Hash-cached.
- **CLI:** `python3 -m tools.video {voice,validate,render,all} AI_Handout_Checker [--only N] [--size 1280x720]`. ffmpeg is taken from PATH; the CLI prepends `~/.venvs/ffmpeg-static/shim` if it exists.

**Tests:**
- Render a synthetic 2-second slide (a small PNG, a hand-written cue sheet, synthetic timings, a silent wav) to MP4 at 320×180 twice. Decoded frames must be identical: extract frames with ffmpeg to raw and hash them.
- The MP4 duration ≈ the wav duration.

**Smoke test:**
- Write a hand-made cue sheet for slide 4 (focus u3 + spotlight, underline, glide to u4, ring on s4_e19) and a 3-line script `script/AI_Handout_Checker/slide_004.md`.
- Run `voice --only 4` and `render --only 4 --size 1280x720`.
- Extract 3 frames at cue times and look at them.

Remove the hand-made files afterwards; do not commit them.

### Task 5: Agent packets + named workflows (draft, direct)

**Files:** create `tools/video/packets.py`, `.claude/workflows/video-draft.js` and `.claude/workflows/video-direct.js`, and extend `tools/video/__main__.py` with `packets-draft` and `packets-direct`.

- [ ] **`packets-draft`** writes `out/video/<deck>/packets/draft/slide_NNN.json` containing:
  - the PNG path;
  - map message, units (summary, talk, terms) and relations;
  - pointable elements (id, role, text, bbox);
  - the previous and next slides' map messages.
- [ ] **`packets-direct`** writes `out/video/<deck>/packets/direct/slide_NNN.json` containing:
  - the PNG path;
  - map (units, summary, talk, terms, relations, order);
  - elements (id, role, text, bbox, line_texts with line indices, value);
  - script sentences as `[[ [w_index, word], ... ], ...]` using `script.words`;
  - the effect vocabulary and rules (from the spec §4–5);
  - a `hash` of all of these.
  `direct` reruns only slides whose packet hash differs from the hash stored in their cue sheet.
- [ ] **`video-draft.js`** (named workflow; `args = {deck}`) runs one agent per slide:
  - Each agent reads its draft packet and looks at the PNG.
  - It writes `script/<deck>/slide_NNN.md`: 3–6 sentences, one per line, spoken-ready (numbers as words, no symbols or markdown), in a natural presenter style that paraphrases and explains, with transitions. Continuity comes from the previous and next messages.
  - The agent must not read `out/bench/*/presenter*`, traces, adjudications or `tasks/lessons.md`.
  - Skip slides whose script file already exists unless `args.force`, so user edits are never overwritten.
- [ ] **`video-direct.js`** (`args = {deck, only?}`) runs one agent per slide whose packet hash changed:
  - The agent reads the direct packet, looks at the PNG, and writes `out/video/<deck>/cues/slide_NNN.json` (schema per spec §4) with `"hash": <packet hash>`.
  - Then it runs `python3 -m tools.video validate <deck> --only N` and fixes the sheet until it prints VALID, with at most 2 repair rounds.
  - The guidance from spec §5 is included verbatim.
- [ ] **Tests:** the packet contents and hashing (a changed sentence changes the hash; an unchanged one does not), and `validate` CLI exit codes. The workflows are not unit-tested.
- [ ] Commit.

### Task 6: First full build + quality checks (orchestrator runs this)

- [ ] `python3 -m tools.video packets-draft AI_Handout_Checker`, then run the `video-draft` workflow, which writes 12 script files. Commit `script/AI_Handout_Checker/` as the first draft for the user to edit.
- [ ] `python3 -m tools.video voice AI_Handout_Checker`. Report the alignment coverage.
- [ ] `python3 -m tools.video packets-direct AI_Handout_Checker`, then run the `video-direct` workflow, which writes 12 cue sheets, all VALID.
- [ ] `python3 -m tools.video render AI_Handout_Checker`, which produces 12 slide MP4s plus `deck.mp4` at 1920×1080.
- [ ] **Frame audit:** an agent extracts the frame at each cue start + 0.5 s and judges each one:
  - Is the effect on what the sentence talks about?
  - Is anything important covered?
  - Is it readable?

  Report pass/fail per cue; target ≥ 90 %.
- [ ] **Determinism:** render slide 4 twice, then compare decoded-frame hashes.
- [ ] Add a tasks/lessons.md entry; send the deck video to the user.
