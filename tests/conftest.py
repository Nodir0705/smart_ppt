"""Shared test fixtures.

- ``real_encoder``: the default BGE encoder loaded ONCE per session and shared
  across modules (the model load dominates the slow suite; each module used to
  reload it). Skips when SKIP_SEMANTIC is set.
- ``FakeEncoder`` / ``fake_encoder``: a deterministic, model-free encoder for
  hermetic logic tests that don't need real semantics.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest


class FakeEncoder:
    """Deterministic, model-free encoder.

    Maps each text to a fixed pseudo-random unit vector via a stable hash, so the
    same text always yields the same embedding and different texts differ. Lets
    policy/index logic be tested without loading the real model.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.model_name = "fake-encoder"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int.from_bytes(hashlib.sha256((t or "").encode()).digest()[:8], "little")
            v = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
            out[i] = v / (np.linalg.norm(v) or 1.0)
        return out


@pytest.fixture(scope="session")
def real_encoder():
    """The default encoder, loaded once and shared across the whole session."""
    if os.environ.get("SKIP_SEMANTIC"):
        pytest.skip("SKIP_SEMANTIC set")
    from src.encoder import load_default_encoder
    return load_default_encoder()


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    return FakeEncoder()
