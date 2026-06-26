# smartppt Optimization Report — Synthesis

**Scope:** ~6,247 lines of Python (src/, tools/, scripts/). Live grounding loop: pptx parse -> bbox refine -> Vosk STT -> bge embed/match -> tkinter highlight. Plus offline TTS/narrated-video synth and build-time VLM/LLM enrichment.

**Hard constraint (overrides every "best tool" instinct):** fully **offline**, **deterministic**, **CPU-only**. No cloud/hosted LLM, no GPU (driver/library mismatch — even though this box's CUDA happens to work, the project treats it as unusable). Same input must yield the same highlight decision. Every recommendation below is installable+runnable offline on CPU, permissively licensed, and determinism-preserving.

---

## 1. Executive Summary

The audits converge on **one dominant lever and a stack of cheap, high-confidence wins**.

The dominant lever is the **embedding backend**. `BAAI/bge-large-en-v1.5` (335M params, 1024-dim, CLS pool) loaded through `torch`+`transformers` is simultaneously:
- the **per-frame live bottleneck** — `matcher.py:186` calls `encoder.encode([spoken])` every throttled frame at a measured **8.2ms median/query**, invoked up to 5x/sec, with **no dedup** of identical growing Vosk partials; and
- the **heaviest install** — the installed `torch==2.3.1+cu121` drags **~5.3GB** of CUDA/triton libraries (`torch 1.6G + nvidia/* 3.3G + triton 420M`) onto a CPU-only box, none of which can run, plus a **10.1s** model cold-start that blocks app startup.

Three stacked moves attack both axes: pin torch to the **CPU wheel** (~700MB, drops ~4.6GB dead weight), swap to **bge-small-en-v1.5** (33M, 384-dim, expected ~1-2ms/query and ~1-2s load), and add a **1-entry query LRU** to skip re-encoding identical partials. The only real cost is a **mandatory threshold re-tune** (`PolicyConfig.threshold=0.55`, `matcher.py:224`) validated against `tools/timeline_audit.py`.

Independently, a tier of **zero-new-dependency quick wins** is shippable in under a day each: debounced viewer resize, precompiled alias regex, the `audioop` 3.13 time-bomb fix, `.npz`+eviction cache, and a raw-RGB ffmpeg pipe for video synth.

The cross-cutting structural gap is the **total absence of a dependency manifest** — no `requirements.txt`/`pyproject.toml`/lockfile exists (confirmed) — which makes the project's *hard determinism guarantee unenforceable*. Fixing it with `uv` + hashed lock + vendored wheelhouse is medium-effort but foundational.

Verifiers correctly **rejected** every cloud/GPU/AGPL temptation: fastembed (GCS network fallback), PyMuPDF (AGPL), faster-whisper for the live loop (batch, not streaming), model2vec/potion (retrieval-quality regression).

---

## 2. Top Priorities (ranked by impact / effort)

| # | Recommendation | Subsystem | Impact | Effort | Key evidence |
|---|---|---|---|---|---|
| 1 | **bge-small + CPU torch wheel + query LRU** | embeddings | Transformational | Medium | 8.2ms->~1-2ms/query; 10.1s->~1-2s load; install ~5.3GB->~700MB; `encoder.py:31`, `matcher.py:186` |
| 2 | **Manifest + hashed lock + offline wheelhouse (uv)** | packaging | High | Medium | No manifest exists at all; determinism unenforceable; reconcile cu11+cu12 mess |
| 3 | **.npz + JSON sidecar + eviction cache** | embeddings | Medium | Small | 9 stale pkl / 3.8MB confirmed; pickle = ACE + hand-bumped `CACHE_VERSION` (`matcher.py:108`) |
| 4 | **Debounce viewer resize + BILINEAR-on-drag** | render+viewer | Medium | Trivial | 37ms LANCZOS/event vs 12ms BILINEAR; `viewer.py:260-272`, bound `:150` |
| 5 | **Replace audioop + `requires-python` guard** | stt+audio | Medium | Small | Top-level `import audioop` at `audio.py:15`/`tts.py:14`; removed in Py3.13 |
| 6 | **Raw-RGB ffmpeg pipe + static-frame dedup** | render+viewer | Medium | Small | ~3315 frames, ~86% identical; 9-15ms PNG/frame; `simulate_to_video.py:146-181` |
| 7 | **pypdfium2 for build-time PDF text refine** | parse+geometry | Medium | Medium | 141ms->4.8ms extract (~29x); BUILD-TIME ONLY; needs re-calibration |
| 8 | **Fix dropped LLM wiring; pin CPU/fp32/seed** | enrichment | Medium | Small | 2.9GB model output discarded by default pipeline; cross-machine fp16/fp32 divergence |
| 9 | **Precompile alias-normalizer regex** | stt+audio | Low | Trivial | per-frame `re.compile` of an unchanging map; `asr_normalize.py:141-162` |
| 10 | **conftest.py: session encoder + FakeEncoder** | packaging | Medium | Small | slow suite = ~4.6s model-load x3, ~0.3s logic; 4 independent load sites |

---

## 3. Per-Subsystem Recommendations

### 3.1 Embeddings + Matcher (the live bottleneck)

**Before:** `matcher.py:186` runs `bge-large-en-v1.5` (335M, 1024-dim) per frame at **8.2ms/query**; `encoder.py:33-43` cold-loads in **10.1s**; install carries **~5.3GB** CUDA libs that never run; identical Vosk partials are re-encoded every throttle window.

**After (stacked):**

1. **Model swap — bge-large -> bge-small-en-v1.5** (33.4M, 384-dim, same CLS-pool+L2-norm recipe).
   - Tool/license: `BAAI/bge-small-en-v1.5`, **MIT** (FlagEmbedding + weights).
   - Benefit: ~10x fewer params -> expected **~1-2ms/query**, **~1-2s load**. MTEB retrieval delta only ~2.6pts, immaterial here because the deterministic threshold/gap/hysteresis policy is the real arbiter over a 120-phrase deck.
   - Change is one line: `DEFAULT_MODEL` at `encoder.py:31`. Element caches auto-rebuild (`model_name` is in `_cache_key`, `matcher.py:112`).
   - **Mandatory caveat:** absolute cosine magnitudes differ -> **re-tune `PolicyConfig.threshold=0.55` (`matcher.py:224`) and gap=0.03**, validated on `tools/timeline_audit.py`. This is a behavior change, not a pure perf swap — *not* "trivial" despite the trivial code edit.

2. **Backend — pin torch CPU wheel** (`torch==2.3.1`, **BSD-3**, no nvidia/triton).
   - Benefit: drops **~4.6GB** of un-runnable CUDA payload (5.3GB->~700MB installed).
   - **Caveat (verified):** `torch.cuda.is_available()` returns **True** on this box and the encoder is *currently running on CUDA* (`encoder.py:38`). The CPU wheel changes the execution device; CPU vs CUDA float kernels are **not** bit-identical, so cached embeddings + thresholds must be **rebuilt/re-validated** at the boundary. Force `device='cpu'` explicitly rather than relying on availability. The claimed "numpy-1.x ABI conflict" is unfounded (interop verified working) — do this for footprint + determinism-enforceability, not an ABI fix.

3. **Query dedup — 1-entry LRU** on `(spoken, slide_no)` in `Matcher.scored`.
   - Tool/license: stdlib `functools.lru_cache` (**PSF**), no dep.
   - Benefit: `encode()` is a pure function; returning the cached vector for byte-identical partials is behavior-preserving and eliminates **idle/repeat** encodes for free. Honest bound: it does *not* help mid-utterance when the tail grows token-by-token. Key must be `(spoken, slide_no)` **together** (same tail on a different slide re-ranks different elements).

**Cache format (rank 3):** Replace pickle (`matcher.py:124-133`) with **`.npz`** (numpy, **BSD-3**) for float32 matrices + **JSON sidecar** (`sort_keys` for byte-determinism) for `ElementRef` metadata; add **LRU/age eviction** on write. Kills the arbitrary-code-execution pickle format, the silent-stale-cache footgun (the reason `CACHE_VERSION` is hand-bumped, `matcher.py:108`), and the **9-file / 3.8MB** pileup confirmed on disk. `safetensors` 0.5.3 is an equivalent already-installed alternative.

**Deferred / low-value:** A `rapidfuzz`/`rank_bm25` lexical fast-path (MIT/Apache-2.0, offline, deterministic) can short-circuit near-literal tails, but its **marginal** value shrinks once the LRU cache + bge-small land, and it adds a second ranking path that can diverge from the neural one inside a determinism-critical loop. **Flag-gated, deprioritized** behind the backend swap. (Neither lib is installed; do not auto-install.)

---

### 3.2 STT + Audio

**Keep Vosk** for the live loop (Apache-2.0, ~6.35ms/200ms-chunk true streaming partials that `RollingBuffer.tail(8)` depends on). Do **not** swap to faster-whisper (batch/whole-utterance) — see Rejected.

1. **audioop -> scipy.signal.resample_poly + numpy** (rank 5).
   - Before: top-level `import audioop` at `audio.py:15` and `tts.py:14`; `ratecv`/`tomono` used for PCM. audioop is **removed in Python 3.13** (PEP 594) — a hard ImportError that takes down MicSource/FileSource/TTS even though MicSource/FileSource never touch it.
   - After: lazy import + `scipy.signal.resample_poly` (resample) and `pcm.reshape(-1,2).mean(axis=1)` (mono). Both **BSD-3**, already installed (scipy 1.14.0, numpy 2.2.6), **verified bit-identical across runs**. Declare **`requires-python='>=3.10,<3.13'`** now as an install-time guard; **audioop-lts** (PSF-2.0) is the *conditional* `python_version>='3.13'` shim.
   - Caveat: `resample_poly` output is **not** byte-identical to audioop's linear `ratecv` (different FIR), so any golden-WAV byte tests must be re-goldened; same-in/same-out determinism is preserved.

2. **Precompile alias-normalizer regex** (rank 9).
   - `asr_normalize.py:141-162` re-sorts keys and `re.compile()`s a joined look-around pattern **every frame** for a map built once. Build a small `Normalizer` (or `lru_cache` on `id(alias_map)`) at map-build time. Stdlib `re` (**PSF**), output byte-identical. ~0.03-0.08ms/frame saved, grows with map size.

3. **Mic callback hygiene** (low).
   - `audio.py:72-78` runs `import numpy` and allocates int16 per block on the PortAudio realtime thread — the exact xrun pattern the playback path was already redesigned to avoid. Open `sd.InputStream(dtype='int16')` so PortAudio delivers int16 directly (no float truncation, deterministic), with a float32 fallback for devices that refuse int16. At minimum hoist the numpy import. sounddevice (**MIT**), no new dep.

**Offline narration accuracy (opt-in, NON-live only):** faster-whisper (CTranslate2, **MIT**, INT8) may transcribe the pre-rendered narrated video (`simulate_to_video.py`) at lower WER. **deterministic_ok=false as written** — needs `temperature=0` AND `beam_size=1` AND **pinned `cpu_threads`/`OMP_NUM_THREADS`** AND a fixed CPU build, plus the model **vendored** (it defaults to downloading from HF). Flag-gated, walled off from the live loop, behind a determinism test — or skip and stay on Vosk everywhere.

---

### 3.3 Render + Viewer

All build-time/offline except the viewer resize (live UI).

1. **Debounce viewer resize** (rank 4, **live**).
   - `viewer.py:260-272` unconditionally re-runs a full **37.2ms LANCZOS** resize + `canvas.delete('all')` + new `ImageTk.PhotoImage` on every `<Configure>` (bound `:150`). After: `after_cancel`/`after(~80ms)` debounce, early-return on unchanged `(cw,ch)`, **Image.BILINEAR (~12.5ms)** during drag + LANCZOS on settle. Pillow (**HPND**) + Tk (**PSF**), no dep, at-rest image unchanged. Verified both filters deterministic per run.

2. **Raw-RGB ffmpeg pipe + static-frame dedup** (rank 6, **offline batch**).
   - `simulate_to_video.py:146-181` `base.copy()`+`ImageDraw`+PNG-writes all **~3315 frames** to a tmpdir though only **~468 (14%)** change (39 transitions x 12 anim frames); ~86% are byte-identical statics. Pipe `frame.tobytes()` to the **already-present ffmpeg 4.3.2** via `-f rawvideo -pix_fmt rgb24 -s WxH -i -` — measured ~9.3ms->0.44ms per static frame (**~21x**), zero tmpdir churn. No new dep.
   - **Determinism caveat (verified):** x264 **multi-thread output is non-deterministic** — threads=1/2/4 gave 3 distinct hashes; current `_encode_video` sets **no** `-threads` flag (only accidentally reproducible). **Pin `-threads 1`.** The MP4 is a terminal artifact (nothing re-reads it, no golden hash test), so per-backend determinism is what matters.
   - **PyAV** (av 12.3.0, **BSD-3**, bundled ffmpeg ~6.1, in-process, byte-identical across runs at `thread_count=1`) is the lower-risk-but-version-changing alternative. Prefer the **raw-pipe to the existing 4.3.2 binary** first (keeps the current ffmpeg pin, zero new dep); treat PyAV as optional.
   - Second step: ffmpeg **concat demuxer** static-segment dedup (>85% fewer drawn frames) — needs a regression test aligning segment boundaries to `summary.json` transition timestamps.

3. **Warm LibreOffice (optional, authoring loop only).**
   - `render.py:40-64` cold-spawns soffice per pptx->pdf (**1.76s / 207MB**). The mtime PDF+PNG caches make steady-state cost **zero**; the win only appears in the tight edit/iterate loop. **unoserver** (**MIT**, persistent UNO listener) or a raw `--accept=socket` instance via bundled python-uno (MPL-2.0, `import uno` verified present). Adds a daemon to manage; pin the LO version (PDF bytes drift across releases). Medium value, not high. Also: pass the already-built slide count into `render_deck` to skip the redundant `Presentation()` in `_count_slides` (`render.py:21`, ~4-5ms).

---

### 3.4 Parse + Geometry (build-time only — never per live frame)

1. **pdfplumber -> pypdfium2** (rank 7).
   - `refine_text.py:194` `page.extract_words()` is **141.6ms of 142.7ms** refine cost (pdfminer char-grouping). pypdfium2 (**BSD-3 + Apache-2.0**, already installed 5.6.0, bundled `libpdfium.so`, no PIL/pdfminer, **deterministic SHA verified**) extracts in **4.8ms (~29x)** here.
   - **CRITICAL caveat — NOT a byte-stable drop-in:** PDFium returns rendered-glyph extents; pdfminer derives word height from font ascent/descent. Verified **0/37 bboxes byte-identical** (max 0.23-0.40in diff). The thin word-assembler must reproduce page size + `x0/x1/top/bottom` point semantics (y from page top), then **re-calibrate `SHIFT_FRAC=0.22` against the 150-dpi PNGs** (the real ground truth, *not* pdfplumber) and **re-snapshot the bbox goldens**; re-verify `test_refine_index_real_deck_tightens_text` (eps=0.05). Word reading-order differs but is harmless (assignment is by center-containment). Lower priority than the live-loop wins.

2. **Skip redundant pptx parse** (low): `render_deck` accept an `expected_count`/index and skip `_count_slides` (`render.py:21`); python-pptx (**MIT**), keep the CLI fallback, guard against a wrong count corrupting `_pngs_are_fresh`.

3. **nbsp / regex cleanup** (low, correctness-neutral): hoist the three `_norm`/`_strip_punct` patterns (`parser.py:78-87`) to module-level `re.compile`. **Verifier correction:** line 81's `re.sub(r'\s+',' ', s)` already collapses U+00A0 in Python 3, so the "nbsp leaks into trigger phrases" claim is **false** — line 82 is harmless dead code. Delete it / fix the comment; **do NOT re-snapshot goldens** (output is unchanged; any shift would be a real regression).

4. **numpy vectorization of `assign_words` (DEFER):** microseconds today (16 elements/slide). Ready in-tree target only if decks ever become dense; a vectorized version must replicate the area-then-first-encountered tie-break exactly or bbox numbers shift. Don't pre-optimize.

---

### 3.5 Enrichment (build-time VLM/LLM — heaviest deps in the project)

1. **Fix the dropped LLM wiring FIRST** (rank 8, this is a **bug**, not an optimization).
   - `enrich_llm.py` writes `index.enriched.json` (**182** phrases) but **nothing loads it**: `refine_text` defaults `--input` to `index.json`, `group_scope` produces `index.scoped.json` (120), and `matcher.py:80` loads scoped>refined>base. So the entire **+82-phrase pass and its 2.9GB model load produce dead output**. Decide intent: **chain the stages by default + add a regression test** that the matcher's loaded index contains enriched phrases, **or delete `enrich_llm.py`** and drop the 2.9GB model. Also remove the stray `temperature=1.0` (`enrich_llm.py:144`).

2. **Pin CPU + fp32 + seed for ALL build-time models** (the highest-leverage determinism fix).
   - `enrich_llm.py:121`, `enrich_vlm.py:187`, **and `encoder.py:38`** do `'cuda' if torch.cuda.is_available() else 'cpu'` with `fp16` on CUDA / `fp32` on CPU. `cuda` is True here, so the **same deck yields different trigger phrases/embeddings per machine** — a live determinism-contract violation, most critically in `encoder.py` (every match score depends on it). Pin `device='cpu'`, `dtype=float32`, `torch.manual_seed(0)`, `torch.use_deterministic_algorithms(True)` (guarded), and a **fixed thread count** (documented as part of the contract). Config-only, no dep. This is a **precondition** for every other determinism claim.

3. **Move charts off the 4.2GB Qwen2-VL to deterministic python-pptx** (**MIT**, already a dep).
   - `enrich_vlm.py` asks the VLM for chart type/title/largest-bar; the parser **already reads** `chart_type`, `categories`, `series` (`parser.py:281-287`), and `Chart.has_title`/`chart_title` is a verified plain XML read in installed 1.0.2. The VLM's marginal value is tiny (**+23 phrases deck-wide**, mostly deduped) and noisy. Read structured data instead; eliminates OCR-style nondeterminism. **Scope note:** this removes the VLM only for *native* chart objects — the 4.2GB model still backs genuine raster images / flattened-chart pictures until step 4 confirms none remain.

4. **Add a content-hash skip-if-unchanged gate** (build-time waste): both `enrich_index` reload multi-GB weights and regenerate deterministic output every run. Reuse the repo's content-hash idiom: hash(input bytes + model_name + **all prompt constants** + code-version sentinel) -> skip model load on a match. Stdlib `hashlib`. **Precondition:** outputs must be deterministic (step 2) first, and the LLM wiring resolved (step 1) — caching dead output just caches it faster.

**Optional / deferred:** a smaller LLM via **llama-cpp-python** Q4_K_M GGUF (1.12GB, MIT/Apache-2.0) only *after* the wiring is fixed and the +82 phrases prove their worth — and only with **pinned build + `--top-k 1` + fixed threads** (greedy reproducibility is build/thread-scoped). If a VLM must stay for alt-text-less raster images, prefer **python-pptx alt-text first (zero model)**, then a single revision-pinned INT8 ONNX captioner (Florence-2 MIT / moondream2 Apache-2.0) via the already-installed onnxruntime — `deterministic_ok=false` without pinned threads + pinned revision; **large effort, defer unless it earns its keep**.

---

## 4. Quick Wins (< 1 day each, zero/low new dependency)

| Win | File:line | Dep | Determinism |
|---|---|---|---|
| Debounce `<Configure>` + BILINEAR-on-drag | `viewer.py:260-272` (bound `:150`) | none (Pillow/Tk) | preserved |
| 1-entry query LRU on `(spoken, slide_no)` | `matcher.py:186` | stdlib | preserved by construction |
| Precompile alias-normalizer regex | `asr_normalize.py:141-162` | stdlib `re` | byte-identical |
| Lazy `import audioop` + `requires-python<3.13` guard | `audio.py:15`, `tts.py:14` | scipy/numpy (installed) | same-in/same-out |
| `.npz` + JSON sidecar + cache eviction | `matcher.py:124-133` | numpy (installed) | improves (no pickle) |
| Raw-RGB ffmpeg pipe, `-threads 1` | `simulate_to_video.py:146-181` | existing ffmpeg | per-backend stable |
| `sd.InputStream(dtype='int16')` mic capture | `audio.py:72-78` | sounddevice (installed) | improves (no float trunc) |
| Skip redundant `_count_slides` parse | `render.py:21,76` | python-pptx (installed) | trivially preserved |
| Pin CPU+fp32+seed in encoder/enrichers | `encoder.py:38`, `enrich_*.py` | none (config) | **fixes cross-machine drift** |
| Delete dead `parser.py:82` nbsp no-op | `parser.py:78-87` | stdlib | output unchanged |
| `conftest.py`: session encoder + FakeEncoder | `tests/` (new) | pytest (installed) | hermetic logic tests |

**Note on the matplotlib filter:** `pytest.ini`'s `ignore::UserWarning:matplotlib` is **NOT dead** — matplotlib is imported *transitively* via torch/transformers and emits a live `UserWarning` on this box. **Keep it** (the audit's "delete as dead" was a false premise); just move pytest config into `[tool.pytest.ini_options]` in the new pyproject and add the observed `audioop` DeprecationWarning filter.

---

## 5. Considered & Rejected

| Idea | Verdict reason |
|---|---|
| **PyMuPDF / fitz** (pdf text + pdf->png) | **AGPL-3.0** — network-copyleft poison pill for a redistributable tool; paid commercial escape only. pdfplumber/poppler are already permissive. |
| **fastembed** (Qdrant) as live encoder backend | **Offline violation**: `HF_HUB_OFFLINE=1` bypasses local cache and falls back to a `storage.googleapis.com` download (issue #615); firewall hangs (#218). Network-touching *default* in an air-gapped system. Hand-rolled OnnxEncoder dominates it. Also bundles INT8 bge-**small** (different+quantized model -> changes every highlight decision). |
| **model2vec / potion** static embeddings as live encoder | Retrieval quality materially lower (potion-retrieval-32M 35.06 vs bge-small 51.68 MTEB); would degrade the ambiguous decisions it is meant to pre-filter. Fails same-input->same-highlight equivalence. |
| **faster-whisper** for the LIVE loop | Batch/whole-utterance; **no streaming partials** the highlight loop + `RollingBuffer.tail(8)` depend on. (OK as opt-in, pinned, vendored backend for the *offline* narration path only.) |
| **NVENC / any GPU encode path** | GPU declared unusable + nondeterministic. |
| **unoconv** (warm LibreOffice) | Unmaintained, not installed. Use unoserver (MIT) or raw UNO socket. |
| **PyAV bundled-ffmpeg** as the *required* path | Acceptable (BSD-3, offline, deterministic at `thread_count=1`) but changes the ffmpeg version pin; prefer raw-pipe to the existing 4.3.2 first. Listed as optional, not rejected outright. |

---

## 6. Phased Roadmap

### Phase 0 — Quick Wins (days, no new deps, ship independently)
- Viewer resize debounce + BILINEAR-on-drag (`viewer.py`).
- 1-entry query LRU (`matcher.py:186`).
- Precompile alias regex (`asr_normalize.py`).
- Lazy audioop + scipy resample + `requires-python<3.13` guard.
- `.npz` + JSON sidecar + eviction (kills the 9-file pickle pileup).
- Raw-RGB ffmpeg pipe with `-threads 1` (`simulate_to_video.py`).
- **Pin CPU/fp32/seed/threads** across `encoder.py` + enrichers (fixes cross-machine determinism — precondition for everything else).
- `int16` mic capture; skip redundant `_count_slides`; delete dead nbsp line; keep matplotlib filter.

### Phase 1 — Medium (the strategic backend migration + packaging debt)
- **bge-small + CPU torch wheel** swap; **re-tune `PolicyConfig.threshold`** and rebuild caches; validate on `tools/timeline_audit.py`. (Rank 1 — biggest live-latency + install-size lever.)
- **uv pyproject + hashed `uv.lock` + offline wheelhouse**, runtime/enrich/dev extras split; reconcile the cu11+cu12 closure; validate offline on a network-cut box; pin uv itself. (Rank 2.)
- **conftest.py** session encoder + FakeEncoder; keep one real-BGE `slow` tier as a semantic-regression backstop.
- **Fix or delete the LLM-enrichment wiring** + regression test; move chart enrichment to python-pptx XML.
- Add `[tool.ruff]`/`[tool.ruff.lint]` config + mypy (dev extra) targeting the Encoder Protocol seam; offline CI on the fast tier.

### Phase 2 — Larger / deferred (only when justified by profiling or scale)
- **pypdfium2** for refine extraction — calibrated migration with SHIFT_FRAC re-derivation + golden re-snapshot (build-time only).
- ffmpeg **concat-demuxer static-frame dedup** with `summary.json` timing regression test.
- **Warm LibreOffice (unoserver)** for the authoring iterate loop (cache already blunts steady-state value).
- Optional **ONNX OnnxEncoder** (same bge weights, fp32-first, pinned opset/provider) only if the residual ~700MB torch footprint is itself a hard constraint — validate cosine ~1.0 equivalence first.
- Optional **llama-cpp-python GGUF** LLM / ONNX captioner VLM — only after enrichment proves value, with pinned-build + fixed-thread + revision-locked determinism.
- Optional **faster-whisper** opt-in backend for the offline narrated-video path (vendored model, pinned threads/decoding, determinism test).

---

**Determinism discipline that threads through everything:** pin device=CPU + fp32, pin thread counts (torch / ORT / x264 / llama.cpp all have order-sensitive parallel reductions), pin tool/model versions, and fold model_name + version sentinels into every content-hash cache key. These are the conditions under which each recommended swap stays "same input -> same highlight."

---

## 7. Critic Corrections & Additions (verified post-synthesis)

An adversarial completeness pass verified the above against the running tree and found these corrections — fold them in before acting:

1. **The INCUMBENT encoder already violates the offline constraint (biggest unflagged hole).** There is **no** `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` / `local_files_only` / pinned `revision` anywhere in `src/` (grep confirms zero). `encoder.py:40-41` and the enrichers call `from_pretrained(model_name)` bare — on a cache miss or HF etag check, `transformers` **silently hits the network** from a system that claims "no network at runtime." bge-small inherits this exact bug. **New Phase-0 quick win (config-only, zero-dep):** set `HF_HUB_OFFLINE=1`+`TRANSFORMERS_OFFLINE=1` (or `local_files_only=True`) and pin an explicit `revision='<sha>'` in every `from_pretrained`, and fold the revision SHA into the cache key. This is a *precondition* for the bge-small swap not silently re-downloading.

2. **Thread-count pinning is recommended everywhere but set NOWHERE today** — so the LIVE matcher's scores are already machine-/load-dependent via torch's order-sensitive parallel CPU reductions. This is a *present* determinism-contract violation, not a future build-time concern. **New Phase-0 quick win:** `torch.set_num_threads(1)` + `OMP_NUM_THREADS` at the live entrypoint (`live.py`/`viewer.py`), independent of the device pin; document the pinned value as part of the determinism contract.

3. **Unbundle Rank 1.** Split into **1a** (pin CPU torch wheel + force `device='cpu'` + LRU(1)) — behavior-preserving, captures the *entire* ~4.8GB footprint + idle-encode win at near-zero risk, ship in **Phase 0**; and **1b** (bge-large → bge-small) — the only piece that forces the threshold/gap re-tune + full cache rebuild + golden re-validation, gate behind `tools/timeline_audit.py`. No cheaper single lever beats bge-small on the 8.2ms/query encode itself, so #1 ranking stands.

4. **Install-size numbers are ~0.5GB high.** Measured on this box: torch 1.6G + nvidia 2.8G + triton 420M = **~4.8GB** (report says 5.3GB). The ~700MB CPU-wheel target and ~7x shrink are right against ~4.8GB.

5. **audioop is on LIVE/used paths, not just a dead import.** `PlaybackSource` uses `audioop.ratecv` (`audio.py:163`) and `tts.py:26-28` uses `tomono`+`ratecv`. The port to `scipy.signal.resample_poly` touches real resampling behavior (re-golden any WAV byte tests), so rank 5 is "small" but not cosmetic.

6. **The orphaned-enrichment waste is ~7GB, not 2.9GB.** BOTH `index.vlm.json` (4.2GB VLM, 100 phrases) AND `index.enriched.json` (2.9GB LLM, 182 phrases) are off the default live path (`index.scoped.json` wins via `best_index` scoped>refined>base). Rank 8's "chain or delete" decision governs the whole `vlm → enriched → scoped` branch (~5x model weight), and the regression test must assert the matcher's *loaded* index actually contains the enriched phrases.

7. **Lock-file fragility on the torch↔numpy axis.** Installed numpy is **2.2.6** while `torch==2.3.1` officially predates numpy-2.0 support (landed in torch **2.4.1**). It works on this pre-built box, but a hashed lock pinning `torch==2.3.1`+`numpy>=2.2` is officially-unsupported and can break on a clean network-cut reinstall (rank 2's own validation target). In rank 2, either pin `numpy<2` alongside torch 2.3.1 **or** bump the CPU wheel to `>=2.4.1` before freezing.

8. **`.npz` byte-stability needs key-order normalization.** `np.savez` writes archive members in dict-insertion order; the JSON sidecar `sort_keys=True` isn't enough — fix the `.npz` member key order too for byte-stable cache files (relevant to rank 3's determinism claim).

**Net:** the top priority (embedding backend) is justified, but the cheapest high-value slice is the Phase-0 bundle — **CPU-wheel pin + force CPU + LRU + offline-env + thread pinning** — which fixes the live determinism violations and captures the ~4.8GB footprint win *before* any model swap or re-tune.