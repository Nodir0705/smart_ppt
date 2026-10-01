# Multi-deck Benchmark Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `python3 -m tools.deck_bench` scores live grounding (recall/precision/flicker) on every deck in `decks/`, with auto-generated narration, in ~seconds per deck.

**Architecture:** Truth parse (`parse_deck(ocr=False)`) → `tools/narrate.py` (pure: script + expected order) → Piper per-segment synthesis with explicit silence gaps and content-hashed WAV names → `build_index` for a named variant → `timeline_audit` run unpaced on a sherpa audio-time clock → `out/bench/report_<variant>.{md,json}`.

**Tech Stack:** Python 3, python-pptx, Piper TTS, sherpa-onnx, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-01-deck-bench-design.md` · **Branch:** `feat/deck-bench` (created, spec committed).

**Verified premise (2026-10-01):** audio-clock + unpaced `FileSource(chunk_ms=100)` reproduces the real-time test_deck audit byte-for-byte (38/40, 1 false, flicker 2), twice, 14.8 s vs ~2 min. Baseline output: `/tmp/claude-1000/-home-netvision-smartppt/45ac6273-e22d-4a7a-81cb-ce58102ee7e7/scratchpad/baseline_audit.txt`.

---

## File map

| File | Change |
|---|---|
| `src/stt.py` | `SherpaEngine(audio_clock=False)`; `load_default_stt_engine(audio_clock=False)` |
| `tools/timeline_audit.py` | `fast` mode, `manifest_path`/`fixture_path`/`models` params, missing-slide skip, CLI flags |
| `src/tts.py` | `PiperTTS._pcm16k`, `synthesize_segments`, module fn `synthesize_slides` (hash-keyed cache) |
| `tools/narrate.py` | NEW — pure narration generator |
| `src/build_index.py` | `ocr: bool = True` kwarg |
| `tools/deck_bench.py` | NEW — CLI orchestrator + report |
| `.gitignore`, `decks/README.md` | user decks folder, git-ignored |
| `tests/test_narrate.py`, `tests/test_tts_segments.py`, `tests/test_deck_bench.py` | NEW |
| `tests/test_corpus.py`, `tests/test_timeline_metric.py` | id-invariant test; skip test |

---

### Task 1: Audio-time clock + fast, parameterized audit

**Files:** Modify `src/stt.py` (~289 `SherpaEngine.__init__`, ~311 `transcribe_stream`, ~340 `load_default_stt_engine`), `tools/timeline_audit.py`. Test: `tests/test_timeline_metric.py`.

- [ ] **Step 1: Failing test for the missing-slide skip**

Append to `tests/test_timeline_metric.py`:

```python
def test_eval_items_skips_slides_missing_from_fixture():
    from tools.timeline_audit import _eval_items
    manifest = [{"slide": 1, "wav": "a.wav"}, {"slide": 2, "wav": "b.wav"},
                {"slide": 3, "wav": "c.wav"}]
    gt = {1: [["s1_e0"]], 3: [["s3_e0"]]}
    items, skipped = _eval_items(manifest, gt, only=None)
    assert [i["slide"] for i in items] == [1, 3]
    assert skipped == [2]
    items, skipped = _eval_items(manifest, gt, only=3)
    assert [i["slide"] for i in items] == [3] and skipped == []
```

Run: `python3 -m pytest tests/test_timeline_metric.py -k skips -v` → FAIL (`ImportError: cannot import name '_eval_items'`).

- [ ] **Step 2: Audio clock in `SherpaEngine`**

In `src/stt.py`, `SherpaEngine.__init__` signature becomes
`def __init__(self, sample_rate: int = 16000, model_dir: Path | None = None, audio_clock: bool = False) -> None:`
and add right after `self.sample_rate = sample_rate`:

```python
        # audio_clock: stamp partials with seconds of audio consumed instead of
        # wall time, so an unpaced file replay sees the same 4 s RollingBuffer
        # window as real-time playback (bench/eval only; live stays wall-clock).
        self.audio_clock = audio_clock
```

In `transcribe_stream`, replace the loop body from `last = ""` to the end of the method with:

```python
        last = ""
        audio_t = 0.0
        for chunk in audio_chunks:
            if not chunk:
                continue
            audio_t += len(chunk) / 2 / self.sample_rate
            stamp = {"timestamp": audio_t} if self.audio_clock else {}
            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            stream.accept_waveform(self.sample_rate, samples)
            while rec.is_ready(stream):
                rec.decode_stream(stream)
            text = rec.get_result(stream).lower()  # match Vosk's lowercase contract
            if rec.is_endpoint(stream):
                if text:
                    yield Partial(text=text, is_final=True, **stamp)
                rec.reset(stream)
                last = ""
            elif text and text != last:
                yield Partial(text=text, is_final=False, **stamp)
                last = text
```

Replace `load_default_stt_engine` with:

```python
def load_default_stt_engine(sample_rate: int = 16000, grammar: bool = False,
                            audio_clock: bool = False):
    """Return the best available offline STT engine.

    Prefers the lower-latency sherpa-onnx streaming model when it's installed and
    the model is present; otherwise falls back to Vosk. `grammar` forces Vosk
    (its deck-vocabulary constraint mode has no sherpa equivalent here).
    `audio_clock` (unpaced bench replay) requires sherpa — no silent fallback.
    """
    if not grammar and _sherpa_model_present():
        try:
            return SherpaEngine(sample_rate=sample_rate, audio_clock=audio_clock)
        except Exception as e:  # missing dep / bad model -> fall back, don't crash
            if audio_clock:
                raise
            print(f"[stt] sherpa-onnx unavailable ({e}); falling back to Vosk")
    if audio_clock:
        raise RuntimeError("audio_clock replay needs the sherpa-onnx engine")
    return VoskEngine(sample_rate=sample_rate, grammar=grammar)
```

- [ ] **Step 3: Parameterize `tools/timeline_audit.py`**

Change the import line `from src.audio import PlaybackSource` to `from src.audio import FileSource, PlaybackSource`.

`record_trajectory` signature → `def record_trajectory(sno, wav, raw_index, engine, matcher, normalize=None, tail_words=8, fast=False):` and replace `src = PlaybackSource(str(wav), mute=True)` with:

```python
    if fast and not getattr(engine, "audio_clock", False):
        raise ValueError("fast replay needs an audio_clock STT engine")
    # fast: unpaced file replay; same 100 ms chunking as PlaybackSource
    src = FileSource(str(wav), chunk_ms=100) if fast else PlaybackSource(str(wav), mute=True)
```

Add above `run`:

```python
def _eval_items(manifest, gt, only):
    """Manifest slides to evaluate, and those skipped for having no expected steps."""
    items, skipped = [], []
    for item in manifest:
        sno = item["slide"]
        if only is not None and sno != only:
            continue
        if sno in gt:
            items.append(item)
        else:
            skipped.append(sno)
    return items, skipped
```

Replace `run` header and setup (down to and including `for item in manifest:` / `sno = item["slide"]` / the `only` check) with:

```python
def run(index_path, only=None, normalize=None, manifest_path=None, fixture_path=None,
        fast=False, models=None):
    raw = json.loads(Path(index_path).read_text())
    manifest_path = Path(manifest_path or ROOT / "out" / "narration" / "manifest.json")
    manifest = json.loads(manifest_path.read_text())
    gt = {s["slide"]: s["expected"]
          for s in json.loads(Path(fixture_path or FIXTURE).read_text())["slides"]}

    if models is None:
        print(f"loading models (BGE + STT)…  index={index_path}", flush=True)
        models = (load_default_encoder(), load_default_stt_engine(sample_rate=16000, audio_clock=fast))
    enc, engine = models
    idx = Index(Path(index_path), enc)
    matcher = Matcher(idx, enc)

    tot_hits = tot_steps = tot_flicker = 0
    tot_relevant = tot_false = 0
    per_slide: dict[int, dict] = {}
    items, skipped = _eval_items(manifest, gt, only)
    for item in items:
        sno = item["slide"]
        events, flicker = record_trajectory(sno, item["wav"], raw, engine, matcher,
                                             normalize=normalize, fast=fast)
```

(The rest of the loop body is unchanged.) In the final `return`, add `"skipped": skipped` inside `"overall"`. Before the `OVERALL:` print add:

```python
    if skipped:
        print(f"skipped (no expected steps): {skipped}")
```

`audit` becomes:

```python
def audit(index_path, normalize_index=True, **kw):
    """Programmatic entry: returns the structured result dict (see run())."""
    normalize = None
    if normalize_index:
        from src.asr_normalize import build_alias_map, normalize as _norm
        raw = json.loads(Path(index_path).read_text())
        amap = build_alias_map(raw)
        normalize = (lambda t: _norm(t, amap)) if amap else None
    return run(index_path, normalize=normalize, **kw)
```

In `main`, add:

```python
    ap.add_argument("--manifest", default=None, help="narration manifest JSON")
    ap.add_argument("--fixture", default=None, help="expected highlights JSON")
    ap.add_argument("--fast", action="store_true",
                    help="unpaced replay on the STT audio clock (bench mode)")
```

and call `run(args.index, only=args.only, normalize=normalize, manifest_path=args.manifest, fixture_path=args.fixture, fast=args.fast)`.

- [ ] **Step 4: Unit test passes**

Run: `python3 -m pytest tests/test_timeline_metric.py -v` → PASS.

- [ ] **Step 5: Equivalence check on test_deck (both modes)**

```bash
S=/tmp/claude-1000/-home-netvision-smartppt/45ac6273-e22d-4a7a-81cb-ce58102ee7e7/scratchpad
python3 -m src.build_index test_deck.pptx > /dev/null 2>&1
python3 -m tools.timeline_audit --index out/index.scoped.json --normalize --fast 2>/dev/null | grep -E '^slide|trajectory|OVERALL' > $S/t1_fast.txt
grep -E '^slide|trajectory|OVERALL' $S/baseline_audit.txt | diff - $S/t1_fast.txt && echo FAST_IDENTICAL
```
Expected: `FAST_IDENTICAL`. Then `SMARTPPT_RUN_EVAL=1 python3 -m pytest tests/test_grounding_eval.py -q` → 1 passed (real-time path, default paths).

- [ ] **Step 6: Full suite + commit**

`python3 -m pytest -q` → all pass.
```bash
git add src/stt.py tools/timeline_audit.py tests/test_timeline_metric.py
git commit -m "feat(eval): unpaced audit on an STT audio-time clock + parameterized paths

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Segment synthesis with silence gaps + hash-keyed cache

**Files:** Modify `src/tts.py`. Test: `tests/test_tts_segments.py` (new).

- [ ] **Step 1: Failing tests**

Create `tests/test_tts_segments.py`:

```python
import wave

from src.tts import TARGET_RATE, PiperTTS, synthesize_slides


class _FakeTTS(PiperTTS):
    """PiperTTS without a voice model: each segment is 0.1 s of a constant tone."""
    def __init__(self):
        self.calls = 0

    def _pcm16k(self, text):
        self.calls += 1
        return b"\x10\x00" * (TARGET_RATE // 10)


def _frames(path):
    with wave.open(str(path), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (TARGET_RATE, 1, 2)
        return w.getnframes()


def test_synthesize_segments_inserts_gaps(tmp_path):
    tts = _FakeTTS()
    wav = tmp_path / "s.wav"
    tts.synthesize_segments(["a", "b", "c"], wav, gap_ms=500)
    # 3 x 0.1 s speech + 3 x 0.5 s gaps (between segments and one trailing)
    assert _frames(wav) == 3 * TARGET_RATE // 10 + 3 * TARGET_RATE // 2


def test_synthesize_slides_caches_by_content(tmp_path):
    tts = _FakeTTS()
    slides = [{"slide": 1, "segments": ["hello"]}, {"slide": 2, "segments": ["world"]}]
    m1 = synthesize_slides(slides, "voice.onnx", tmp_path, gap_ms=900, tts=tts)
    assert [e["slide"] for e in m1] == [1, 2] and tts.calls == 2
    m2 = synthesize_slides(slides, "voice.onnx", tmp_path, gap_ms=900, tts=tts)
    assert m2 == m1 and tts.calls == 2          # cache hit, no re-synthesis
    changed = [{"slide": 1, "segments": ["hello again"]}]
    m3 = synthesize_slides(changed, "voice.onnx", tmp_path, gap_ms=900, tts=tts)
    assert m3[0]["wav"] != m1[0]["wav"] and tts.calls == 3   # new content -> new file
    m4 = synthesize_slides(slides[:1], "voice.onnx", tmp_path, gap_ms=600, tts=tts)
    assert m4[0]["wav"] != m1[0]["wav"]          # gap is part of the key
```

Run: `python3 -m pytest tests/test_tts_segments.py -v` → FAIL (`ImportError: cannot import name 'synthesize_slides'`).

- [ ] **Step 2: Implement**

In `src/tts.py` add `import hashlib` to the imports. Replace the whole `PiperTTS` class with:

```python
def _write_wav(wav_path: Path, pcm: bytes) -> None:
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(TARGET_CHANNELS)
        w.setsampwidth(2)
        w.setframerate(TARGET_RATE)
        w.writeframes(pcm)


class PiperTTS:
    def __init__(self, model_path: str | Path) -> None:
        from piper import PiperVoice
        self.voice = PiperVoice.load(str(model_path))

    def _pcm16k(self, text: str) -> bytes:
        """Synthesize `text` to 16 kHz mono 16-bit PCM bytes."""
        # piper yields its native sample rate; collect, then resample to 16 kHz
        chunks: list[bytes] = []
        native_rate = None
        native_channels = None
        for chunk in self.voice.synthesize(text):
            # piper >=1.4 yields AudioChunk objects with .audio_int16_bytes
            if hasattr(chunk, "audio_int16_bytes"):
                pcm = chunk.audio_int16_bytes
                native_rate = chunk.sample_rate
                native_channels = chunk.sample_channels
            else:
                pcm = bytes(chunk)
                native_rate = native_rate or self.voice.config.sample_rate
                native_channels = native_channels or 1
            chunks.append(pcm)
        return _resample(b"".join(chunks), native_rate or TARGET_RATE, native_channels or 1)

    def synthesize(self, text: str, wav_path: Path) -> None:
        """Synthesize `text` and write a 16 kHz mono 16-bit WAV to `wav_path`."""
        _write_wav(wav_path, self._pcm16k(text))

    def synthesize_segments(self, segments: list[str], wav_path: Path, gap_ms: int) -> None:
        """One WAV of `segments`, each followed by `gap_ms` of silence.

        Piper adds no reliable pause between sentences; an explicit gap keeps
        consecutive narrated elements out of the same STT tail window."""
        gap = b"\x00\x00" * int(TARGET_RATE * gap_ms / 1000)
        _write_wav(wav_path, b"".join(self._pcm16k(s) + gap for s in segments))


def synthesize_slides(slides: list[dict], voice_path: str | Path, out_dir: Path,
                      gap_ms: int, tts: PiperTTS | None = None) -> list[dict]:
    """Per-slide WAVs from `[{"slide", "segments"}]`; returns a manifest.

    WAV names are keyed on a hash of (segments, gap, voice), so a changed
    narration rule can never silently replay stale audio."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for entry in slides:
        key = hashlib.sha1(json.dumps(
            [entry["segments"], gap_ms, str(voice_path)]).encode()).hexdigest()[:10]
        wav = out_dir / f"slide_{entry['slide']:03d}_{key}.wav"
        if not wav.exists():
            tts = tts or PiperTTS(voice_path)
            tts.synthesize_segments(entry["segments"], wav, gap_ms)
        manifest.append({"slide": entry["slide"], "text": " ".join(entry["segments"]),
                         "wav": str(wav)})
    return manifest
```

(Before replacing, confirm the original `synthesize` body matches the loop above — it is the same code moved into `_pcm16k`; if it differs, keep the original logic.)

- [ ] **Step 3: Tests pass; legacy narration byte-identical**

Run: `python3 -m pytest tests/test_tts_segments.py -v` → PASS.
Regression (refactor must not change `synthesize` output):
```bash
S=/tmp/claude-1000/-home-netvision-smartppt/45ac6273-e22d-4a7a-81cb-ce58102ee7e7/scratchpad
mkdir -p $S/tts_check && python3 -m src.tts --out $S/tts_check > /dev/null
for f in out/narration/slide_*.wav; do cmp "$f" "$S/tts_check/$(basename $f)" || echo "DIFF $f"; done; echo checked
```
Expected: only `checked` (no `DIFF`). If Piper output isn't bit-stable run-to-run, report that instead of forcing it.

- [ ] **Step 4: Full suite + commit**

`python3 -m pytest -q` → all pass.
```bash
git add src/tts.py tests/test_tts_segments.py
git commit -m "feat(tts): per-segment synthesis with silence gaps + content-hashed cache

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Narration generator

**Files:** Create `tools/narrate.py`, `tests/test_narrate.py`.

- [ ] **Step 1: Failing tests**

Create `tests/test_narrate.py`:

```python
from tools.narrate import narrate


def _el(eid, typ, text, x, y, **kw):
    return {"id": eid, "type": typ, "text": text,
            "bbox": {"x": x, "y": y, "w": 1.0, "h": 0.5}, "needs_vision": False, **kw}


def _deck(*elements, sno=1):
    return {"slides": [{"slide_number": sno, "elements": list(elements)}]}


def test_title_first_then_banded_rows_left_to_right():
    deck = _deck(
        _el("s1_e1", "body", "Right column top", 6.0, 1.50),
        _el("s1_e2", "body", "Left column top", 1.0, 1.60),   # same band (within 0.3in)
        _el("s1_e3", "body", "Left column lower", 1.0, 3.00),
        _el("s1_e0", "title", "The Title", 1.0, 0.20),
    )
    script, expected = narrate(deck)
    assert script == [{"slide": 1, "segments": [
        "The Title", "Left column top", "Right column top", "Left column lower"]}]
    assert expected == [{"slide": 1, "expected": [
        ["s1_e0"], ["s1_e2"], ["s1_e1"], ["s1_e3"]]}]


def test_allowlist_and_needs_vision_skipped():
    deck = _deck(
        _el("s1_e0", "arrow", "→", 1, 1),
        _el("s1_e1", "image", None, 1, 2, needs_vision=True),
        _el("s1_e2", "image", "Architecture diagram", 1, 3),    # alt text -> narrated
        _el("s1_e3", "chart", "PIE (5)", 1, 4),                  # container not narrated
        _el("s1_e4", "body", "Kept", 1, 5),
    )
    script, expected = narrate(deck)
    assert script[0]["segments"] == ["Architecture diagram", "Kept"]
    assert expected[0]["expected"] == [["s1_e2"], ["s1_e4"]]


def test_chart_categories_phrased_and_kept_in_index_order():
    deck = _deck(
        _el("s1_e0", "chart", "COLUMN_CLUSTERED (51)", 1, 2),
        _el("s1_e1", "chart_category", "Parse", 3.0, 2, parent_id="s1_e0", value=12.0),
        _el("s1_e2", "chart_category", "Match", 1.5, 2, parent_id="s1_e0", value=0.5),
        _el("s1_e3", "chart_category", "Q1 — Revenue", 2.0, 2, parent_id="s1_e0",
            value=40.0, series="Revenue"),
        _el("s1_e4", "chart_category", "Untitled", 2.5, 2, parent_id="s1_e0", value=None),
    )
    script, _ = narrate(deck)
    assert script[0]["segments"] == [
        "Parse is 12", "Match is 0.5", "Revenue Q1 is 40", "Untitled"]


def test_duplicates_narrated_once_with_alternates():
    deck = _deck(_el("s1_e0", "table_cell", "Live", 1, 1),
                 _el("s1_e1", "table_cell", "live ", 3, 1),
                 _el("s1_e2", "table_cell", "Offline", 5, 1))
    script, expected = narrate(deck)
    assert script[0]["segments"] == ["Live", "Offline"]
    assert expected[0]["expected"] == [["s1_e0", "s1_e1"], ["s1_e2"]]


def test_long_text_truncated_to_15_words():
    words = " ".join(f"w{i}" for i in range(30))
    script, _ = narrate(_deck(_el("s1_e0", "body", words, 1, 1)))
    assert script[0]["segments"][0].split() == [f"w{i}" for i in range(15)]


def test_slide_with_nothing_narratable_is_omitted():
    deck = {"slides": [
        {"slide_number": 1, "elements": [_el("s1_e0", "arrow", None, 1, 1)]},
        {"slide_number": 2, "elements": [_el("s2_e0", "title", "Hi", 1, 1)]},
    ]}
    script, expected = narrate(deck)
    assert [s["slide"] for s in script] == [2]
    assert [s["slide"] for s in expected] == [2]
```

Run: `python3 -m pytest tests/test_narrate.py -v` → FAIL (`ModuleNotFoundError: No module named 'tools.narrate'`).

- [ ] **Step 2: Implement `tools/narrate.py`**

```python
"""Auto-narration for the multi-deck bench: deck text -> spoken script + expected order.

Input must be the XML-only truth parse (`parse_deck(deck, ocr=False)`), never the
index under test — otherwise an OCR/VLM hallucination would be narrated and then
scored as a hit. Pure and deterministic.

    script   = [{"slide": n, "segments": ["The Title", "First bullet", ...]}]
    expected = [{"slide": n, "expected": [["s1_e0"], ["s1_e2"], ...]}]
"""
from __future__ import annotations

NARRATED_TYPES = {"title", "bullet", "body", "caption", "stat", "diagram_label",
                  "table_cell", "chart_category", "image"}
MAX_WORDS = 15
BAND_IN = 0.3  # elements whose tops differ by <= this many inches share a row


def _norm(text: str | None) -> str:
    return " ".join((text or "").split())


def _narratable(e: dict) -> bool:
    return (e.get("type") in NARRATED_TYPES and not e.get("needs_vision")
            and bool(_norm(e.get("text"))))


def _phrase(e: dict) -> str:
    if e["type"] == "chart_category":
        raw = _norm(e.get("text"))
        series = e.get("series")
        # multi-series labels are "Category — Series"; speak "Series Category"
        cat = raw.split(" — ", 1)[0] if series else raw
        who = f"{series} {cat}" if series else cat
        v = e.get("value")
        return f"{who} is {v:g}" if v is not None else who
    return " ".join(_norm(e.get("text")).split()[:MAX_WORDS])


def _reading_order(elements: list[dict]) -> list[dict]:
    """Title first; then rows (banded by top y) top-to-bottom, each left-to-right.

    A chart_category is placed at its parent chart's position; sorts are stable,
    so categories of one chart keep their index (category) order."""
    by_id = {e["id"]: e for e in elements}

    def anchor(e: dict) -> dict:
        parent = by_id.get(e.get("parent_id")) if e["type"] == "chart_category" else None
        return (parent or e)["bbox"]

    items = [e for e in elements if _narratable(e)]
    ordered = [e for e in items if e["type"] == "title"]
    rest = sorted((e for e in items if e["type"] != "title"), key=lambda e: anchor(e)["y"])
    band: list[dict] = []
    top = 0.0
    for e in rest:
        y = anchor(e)["y"]
        if band and y - top > BAND_IN:
            ordered += sorted(band, key=lambda b: anchor(b)["x"])
            band = []
        if not band:
            top = y
        band.append(e)
    ordered += sorted(band, key=lambda b: anchor(b)["x"])
    return ordered


def narrate(index: dict) -> tuple[list[dict], list[dict]]:
    """Return (script_slides, expected_slides); slides with nothing to say are omitted.

    Elements whose spoken phrase is identical are narrated once, and that step
    accepts any of their ids (three identical "Live" cells are one step)."""
    script: list[dict] = []
    expected: list[dict] = []
    for slide in index.get("slides", []):
        segments: list[str] = []
        steps: list[list[str]] = []
        seen: dict[str, int] = {}
        for e in _reading_order(slide.get("elements", [])):
            phrase = _phrase(e)
            key = phrase.lower()
            if key in seen:
                steps[seen[key]].append(e["id"])
                continue
            seen[key] = len(steps)
            segments.append(phrase)
            steps.append([e["id"]])
        if segments:
            sno = slide["slide_number"]
            script.append({"slide": sno, "segments": segments})
            expected.append({"slide": sno, "expected": steps})
    return script, expected
```

- [ ] **Step 3: Tests pass**

Run: `python3 -m pytest tests/test_narrate.py -v` → all PASS. Note: the chart test relies on chart_category anchoring to the parent (all at x=1, y=2) and stable sort, so "Parse" stays before "Match" despite x order.

- [ ] **Step 4: Commit**

```bash
git add tools/narrate.py tests/test_narrate.py
git commit -m "feat(bench): pure narration generator (reading order, chart phrasing, dup alternates)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `build_index(ocr=...)` + id invariant

**Files:** Modify `src/build_index.py` (signature ~33, call ~55). Test: `tests/test_corpus.py`.

- [ ] **Step 1: Failing test**

Append to `tests/test_corpus.py`:

```python
@pytest.mark.skipif(__import__("shutil").which("tesseract") is None,
                    reason="tesseract required for the OCR fallback")
def test_ocr_preserves_element_ids(tmp_path):
    """The bench narrates from the ocr=False truth parse and scores the ocr=True
    index — that only works if OCR fills text on existing elements, never adds,
    drops or renumbers them."""
    for d in build_corpus(tmp_path):
        ids = lambda ix: [e["id"] for s in ix["slides"] for e in s["elements"]]
        assert ids(parse_deck(d.path, ocr=False)) == ids(parse_deck(d.path, ocr=True)), d.name


def test_build_index_accepts_ocr_flag():
    import inspect
    from src.build_index import build_index
    assert inspect.signature(build_index).parameters["ocr"].default is True
```

Run: `python3 -m pytest tests/test_corpus.py -k "ocr_preserves or ocr_flag" -v` → the flag test FAILs (`KeyError: 'ocr'`); the invariant test should PASS already (if it FAILS, stop and report — the bench design depends on it).

- [ ] **Step 2: Implement**

In `src/build_index.py` add `ocr: bool = True,` after `expansions_path: str = EXPANSIONS_PATH,` in the `build_index` signature, and change `index = parse_deck(str(deck), ocr=True)` to `index = parse_deck(str(deck), ocr=ocr)`. Add to the docstring's parameter notes (or below the first line): `ocr=False skips the tesseract fallback (bench "no-ocr" variant).`

- [ ] **Step 3: Tests pass + commit**

`python3 -m pytest -q` → all pass.
```bash
git add src/build_index.py tests/test_corpus.py
git commit -m "feat(build): ocr kwarg on build_index; pin OCR id-invariant

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `tools/deck_bench.py` orchestrator + report

**Files:** Create `tools/deck_bench.py`, `tests/test_deck_bench.py`, `decks/README.md`. Modify `.gitignore`.

- [ ] **Step 1: Failing unit tests (pure parts)**

Create `tests/test_deck_bench.py`:

```python
import json
import os
import shutil

import pytest

from tools.deck_bench import inventory, totals


def test_inventory_counts():
    ix = {"slides": [
        {"slide_number": 1, "elements": [
            {"id": "a", "type": "title", "text": "T", "needs_vision": False},
            {"id": "b", "type": "image", "text": "alt", "needs_vision": False},
            {"id": "c", "type": "image", "text": None, "needs_vision": True}]},
        {"slide_number": 2, "elements": [
            {"id": "d", "type": "image", "text": None, "needs_vision": True}]},
        {"slide_number": 3, "elements": []},
    ]}
    inv = inventory(ix)
    assert inv == {"slides": 3, "elements": 4,
                   "by_type": {"image": 3, "title": 1}, "needs_vision": 2,
                   "images_with_alt": 1, "images_without_alt": 2,
                   "image_only_slides": 1}


def test_totals_sum_ok_decks_only():
    recs = [
        {"status": "OK", "hits": 3, "steps": 4, "relevant": 3, "false": 1, "flicker": 2},
        {"status": "OK", "hits": 5, "steps": 5, "relevant": 5, "false": 0, "flicker": 0},
        {"status": "FAILED: boom"},
    ]
    assert totals(recs) == {"decks_ok": 2, "decks_failed": 1, "hits": 8, "steps": 9,
                            "recall": 8 / 9, "false": 1, "precision": 8 / 9, "flicker": 2}


@pytest.mark.skipif(os.environ.get("SMARTPPT_RUN_BENCH") != "1",
                    reason="set SMARTPPT_RUN_BENCH=1 (needs soffice, piper voice, sherpa)")
def test_bench_end_to_end(tmp_path):
    from tools.deck_bench import main
    from tools.make_corpus import build_corpus
    decks = tmp_path / "decks"
    decks.mkdir()
    for d in build_corpus(tmp_path / "corpus"):
        if d.name in ("title_and_bullets", "titled_chart_acronym"):
            shutil.copy(d.path, decks)
    out = tmp_path / "bench"
    assert main(["--decks", str(decks), "--out", str(out)]) == 0
    rep = json.loads((out / "report_default.json").read_text())
    assert sorted(r["deck"] for r in rep["decks"]) == ["title_and_bullets", "titled_chart_acronym"]
    assert all(r["status"] == "OK" and r["steps"] > 0 for r in rep["decks"])
    assert (out / "report_default.md").exists()
```

Run: `python3 -m pytest tests/test_deck_bench.py -v` → FAIL (`ModuleNotFoundError: No module named 'tools.deck_bench'`).

- [ ] **Step 2: Implement `tools/deck_bench.py`**

```python
"""Multi-deck grounding benchmark.

For every decks/<name>.pptx: XML truth parse -> auto narration (tools.narrate) ->
Piper audio -> build_index for a named variant -> unpaced audit on the STT audio
clock -> out/bench/report_<variant>.{md,json}. See
docs/superpowers/specs/2026-10-01-deck-bench-design.md.

    python3 -m tools.deck_bench                       # all decks, variant "default"
    python3 -m tools.deck_bench --variant no-ocr
    python3 -m tools.deck_bench --only my_deck
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

from src.build_index import build_index
from src.encoder import load_default_encoder
from src.parser import parse_deck
from src.stt import load_default_stt_engine
from src.tts import synthesize_slides
from tools.narrate import narrate
from tools.timeline_audit import audit

ROOT = Path(__file__).resolve().parents[1]
VOICE = ROOT / "out" / "voices" / "en_US-amy-low.onnx"
GAP_MS = 900
# variant name -> build_index kwargs; later steps (PaddleOCR, VLMs) add entries
VARIANTS: dict[str, dict] = {"default": {}, "no-ocr": {"ocr": False}}


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def inventory(index: dict) -> dict:
    """What the deck contains — sizes the OCR/layout/vision steps."""
    els = [e for s in index["slides"] for e in s["elements"]]
    imgs = [e for e in els if e["type"] == "image"]
    return {
        "slides": len(index["slides"]),
        "elements": len(els),
        "by_type": dict(sorted(Counter(e["type"] for e in els).items())),
        "needs_vision": sum(bool(e.get("needs_vision")) for e in els),
        "images_with_alt": sum(bool(e.get("text")) for e in imgs),
        "images_without_alt": sum(not e.get("text") for e in imgs),
        "image_only_slides": sum(
            1 for s in index["slides"]
            if s["elements"] and all(e["type"] == "image" for e in s["elements"])),
    }


def totals(records: list[dict]) -> dict:
    ok = [r for r in records if r["status"] == "OK"]
    hits = sum(r["hits"] for r in ok)
    steps = sum(r["steps"] for r in ok)
    relevant = sum(r["relevant"] for r in ok)
    false = sum(r["false"] for r in ok)
    return {"decks_ok": len(ok), "decks_failed": len(records) - len(ok),
            "hits": hits, "steps": steps, "recall": hits / steps if steps else 0.0,
            "false": false,
            "precision": relevant / (relevant + false) if relevant + false else 1.0,
            "flicker": sum(r["flicker"] for r in ok)}


def bench_deck(deck: Path, variant: str, out_root: Path, models) -> dict:
    d = out_root / deck.stem
    truth = parse_deck(str(deck), ocr=False)          # narration source: XML only
    rec: dict = {"deck": deck.stem, "inventory": inventory(truth)}
    script, expected = narrate(truth)
    rec["skipped_slides"] = len(truth["slides"]) - len(script)
    _write(d / "narration.json", {"voice": str(VOICE), "gap_ms": GAP_MS, "slides": script})
    _write(d / "expected.json", {"slides": expected})
    _write(d / "manifest.json", synthesize_slides(script, VOICE, d / "audio", GAP_MS))

    t = time.monotonic()
    build_index(deck, d / variant, **VARIANTS[variant])
    rec["build_s"] = round(time.monotonic() - t, 1)

    t = time.monotonic()
    res = audit(d / variant / "index.scoped.json", manifest_path=d / "manifest.json",
                fixture_path=d / "expected.json", fast=True, models=models)
    rec["eval_s"] = round(time.monotonic() - t, 1)
    o = res["overall"]
    rec.update(hits=o["hits"], steps=o["steps"], relevant=o["relevant"], false=o["false"],
               precision=o["precision"], flicker=o["flicker"],
               recall=o["hits"] / o["steps"] if o["steps"] else 0.0,
               slides=res["slides"], status="OK")
    return rec


def _markdown(variant: str, records: list[dict], tot: dict) -> str:
    lines = [f"# Deck bench — variant `{variant}` (gap {GAP_MS} ms)", "",
             "| deck | status | recall | precision | false | flicker | slides (eval/skip) "
             "| needs_vision | images alt/no-alt | image-only slides | build s | eval s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        inv = r.get("inventory", {})
        if r["status"] == "OK":
            lines.append(
                f"| {r['deck']} | OK | {r['hits']}/{r['steps']} | {r['precision']:.3f} "
                f"| {r['false']} | {r['flicker']} "
                f"| {inv['slides'] - r['skipped_slides']}/{r['skipped_slides']} "
                f"| {inv['needs_vision']} | {inv['images_with_alt']}/{inv['images_without_alt']} "
                f"| {inv['image_only_slides']} | {r['build_s']} | {r['eval_s']} |")
        else:
            lines.append(f"| {r['deck']} | {r['status']} |" + " |" * 10)
    lines += ["", f"**Total:** recall {tot['hits']}/{tot['steps']} = {tot['recall']:.3f} · "
              f"precision {tot['precision']:.3f} · false {tot['false']} · flicker {tot['flicker']} · "
              f"decks OK {tot['decks_ok']}, failed {tot['decks_failed']}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score live grounding on every deck in a folder.")
    ap.add_argument("--decks", default="decks", help="folder of .pptx files (default: decks)")
    ap.add_argument("--out", default="out/bench", help="output root (default: out/bench)")
    ap.add_argument("--variant", default="default", choices=sorted(VARIANTS))
    ap.add_argument("--only", default=None, help="bench just this deck (file stem)")
    args = ap.parse_args(argv)

    decks = sorted(Path(args.decks).glob("*.pptx"))
    if args.only:
        decks = [p for p in decks if p.stem == args.only]
    if not decks:
        print(f"no .pptx decks in {args.decks}", file=sys.stderr)
        return 2

    out_root = Path(args.out)
    models = (load_default_encoder(), load_default_stt_engine(sample_rate=16000, audio_clock=True))
    records = []
    for deck in decks:
        print(f"=== {deck.stem} ===", flush=True)
        try:
            records.append(bench_deck(deck, args.variant, out_root, models))
        except Exception as e:  # one bad deck never aborts the run
            traceback.print_exc()
            records.append({"deck": deck.stem, "status": f"FAILED: {type(e).__name__}: {e}"})
    tot = totals(records)
    _write(out_root / f"report_{args.variant}.json",
           {"variant": args.variant, "gap_ms": GAP_MS, "decks": records, "totals": tot})
    md = _markdown(args.variant, records, tot)
    (out_root / f"report_{args.variant}.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Decks folder, git-ignored**

Append to `.gitignore`:

```
# user benchmark decks (may be private) — see decks/README.md
decks/*
!decks/README.md
```

Create `decks/README.md`:

```markdown
Drop `.pptx` decks here, then run `python3 -m tools.deck_bench`.
Everything in this folder except this README is git-ignored. Report: `out/bench/report_<variant>.md`.
```

- [ ] **Step 4: Unit tests pass**

Run: `python3 -m pytest tests/test_deck_bench.py -v` → 2 passed, 1 skipped.

- [ ] **Step 5: End-to-end**

Run: `SMARTPPT_RUN_BENCH=1 python3 -m pytest tests/test_deck_bench.py -v -s` → 3 passed.
Then a real-deck smoke: `mkdir -p decks && cp test_deck.pptx decks/ && python3 -m tools.deck_bench`
Expected: report table with `test_deck | OK | …`; paste the table in your report. (Auto-narration ≠ the hand script, so numbers won't equal 38/40 — just confirm it runs and recall is non-trivial.) Leave `decks/test_deck.pptx` in place (git-ignored).

- [ ] **Step 6: Full suite + commit**

`python3 -m pytest -q` → all pass.
```bash
git add tools/deck_bench.py tests/test_deck_bench.py decks/README.md .gitignore
git commit -m "feat(bench): multi-deck grounding benchmark (tools.deck_bench)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
