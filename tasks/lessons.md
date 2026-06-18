
## 2026-06-18 — Adversarial verify of render+viewer tool recs
- x264 output bytes change with thread count (verified threads=1/2/4 → 3 distinct SHA-256). Current `_encode_video` sets NO `-threads`, so it inherits core-count default = accidentally machine-dependent. Any "deterministic encode" MUST pin `-threads 1` / `stream.thread_count=1`.
- Installed PyAV 12.3.0 bundles its OWN ffmpeg (libavcodec 60.31 ≈ ffmpeg 6.1), NOT system ffmpeg 4.3.2. Swapping to PyAV changes MP4 bytes vs the current pdftoppm→system-ffmpeg path even at threads=1. This is fine here ONLY because the MP4 is a terminal artifact (no golden-hash test, never re-read into a decision loop) and simulate_to_video.py is an OFFLINE batch CLI, separate from the live grounding loop.
- Lesson: "deterministic" must be scoped — per-backend reproducibility ≠ cross-backend byte-identity. For artifacts not feeding a decision loop, only per-backend matters.
