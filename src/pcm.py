"""Offline PCM helpers: resample + stereo->mono downmix.

Prefers stdlib ``audioop`` (bit-exact with the historical pipeline). ``audioop``
was removed in Python 3.13 (PEP 594), so the import is guarded here — importing
``audio``/``tts`` no longer hard-fails on 3.13. On 3.13+ this falls back to a
numpy/scipy implementation (NOT byte-identical to audioop's linear ``ratecv`` —
re-golden any WAV byte fixtures there, or install ``audioop-lts`` to keep the
stdlib behaviour). Pure functions, fully offline, CPU-only.
"""

from __future__ import annotations

from math import gcd

try:  # stdlib through 3.12; the audioop-lts shim provides it on 3.13+
    import audioop as _audioop
except ImportError:  # pragma: no cover - only on Python 3.13+ without audioop-lts
    _audioop = None


def to_mono(pcm: bytes, width: int = 2) -> bytes:
    """Downmix interleaved 16-bit stereo PCM to mono (equal 0.5/0.5 weights)."""
    if _audioop is not None:
        return _audioop.tomono(pcm, width, 0.5, 0.5)
    import numpy as np

    stereo = np.frombuffer(pcm, dtype=np.int16).reshape(-1, 2).astype(np.int32)
    mono = (stereo[:, 0] + stereo[:, 1]) // 2
    return mono.astype(np.int16).tobytes()


def resample(pcm: bytes, src_rate: int, dst_rate: int, width: int = 2,
             channels: int = 1) -> bytes:
    """Resample 16-bit PCM from ``src_rate`` to ``dst_rate``."""
    if src_rate == dst_rate:
        return pcm
    if _audioop is not None:
        out, _ = _audioop.ratecv(pcm, width, channels, src_rate, dst_rate, None)
        return out
    import numpy as np
    from scipy.signal import resample_poly

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    if channels > 1:
        samples = samples.reshape(-1, channels)
    g = gcd(src_rate, dst_rate)
    resampled = resample_poly(samples, dst_rate // g, src_rate // g, axis=0)
    return np.clip(np.round(resampled), -32768, 32767).astype(np.int16).tobytes()
