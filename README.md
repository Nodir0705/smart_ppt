# smartppt — Smart Pointer

A live presentation **grounding** assistant: as a presenter speaks, it highlights
the slide element they're talking about. Fully **offline**, **deterministic**, and
**CPU-only** by design — no cloud, no GPU, no network at runtime. Same speech →
same highlight, every run.

Pipeline:

```
pptx ──parse──► index.json ──refine(PDF glyph boxes)──► index.refined.json ──scope(groups+phrases)──► index.scoped.json
live: mic ──STT(sherpa-onnx)──► RollingBuffer ──alias-normalize──► matcher(bge-small) ──policy──► Tk highlight
```

## Setup

```bash
pip install -e .                 # see pyproject.toml (requires-python >=3.10,<3.13)
# LibreOffice (soffice) and pdftoppm must be on PATH for rendering/refinement.
# The STT and embedding models are vendored/cached locally (offline).
```

## Build an index for your deck

One command runs all three stages in-process (the single-stage CLIs below still
exist for debugging one stage):

```bash
python3 -m src.build_index my_deck.pptx        # → out/index.scoped.json (+ intermediates)
python3 -m src.build_index my_deck.pptx --no-scope   # stop at refined
python3 -m src.build_index my_deck.pptx --no-refine  # parse only (no soffice needed)
```

Single-stage equivalents (for inspecting one stage):

```bash
python3 -m src.index_deck  my_deck.pptx        # → out/index.json
python3 -m src.render      my_deck.pptx        # → out/slides/slide_NNN.png
python3 -m src.refine_text                     # index.json → index.refined.json
python3 -m src.group_scope                     # index.refined.json → index.scoped.json
```

The live app auto-loads the richest index it finds (**scoped > refined > base**)
via `matcher.best_index`. If a richer file is older than a more-basic sibling, a
stale-index warning is printed — rebuild with `build_index`.

## Run it

```bash
python3 -m src.viewer --mic                    # live: highlight tracks the mic
python3 -m src.viewer --list-devices           # list input devices, then:
python3 -m src.viewer --mic --device 2         # pick a mic by index or name substring
python3 -m src.viewer --wav recording.wav      # replay a 16-bit mono WAV
python3 -m src.simulate                        # TTS-narrated end-to-end demo
python3 -m src.simulate_to_video               # render the demo as MP4s
python3 -m src.preview --slide 6 --phrase "the matcher" --out /tmp/p.png   # headless
```

## Tests

```bash
pytest -q                                      # fast suite (model-free where possible)
SMARTPPT_RUN_EVAL=1 pytest -q tests/test_grounding_eval.py   # opt-in grounding eval (needs soffice + audio)
```

## Environment switches

| Variable | Where | Purpose |
|---|---|---|
| `SMARTPPT_RUN_EVAL=1` | tests | enable the opt-in grounding-accuracy eval (otherwise skipped) |
| `SKIP_SEMANTIC=1` | tests | skip tests that load the real embedding model |
| `SMARTPPT_SHERPA_MODEL` | runtime | path to the sherpa-onnx model dir (else Vosk fallback) |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | runtime | pinned to 1 by `encoder.py` for determinism (override at your own risk) |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | runtime | forced on by `encoder.py` — no network at runtime |

## Design notes

- **Determinism contract:** CPU-only, single-thread, offline-pinned, fixed model
  revisions. See `docs/optimization-research-2026-06-18.md`.
- **Next-frontier research** (accuracy, generalization, architecture, product):
  `docs/improvement-research-2026-06-24.md`.
