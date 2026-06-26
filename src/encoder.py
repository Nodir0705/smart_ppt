"""Local sentence-embedding wrapper.

Default backend: BAAI/bge-small-en-v1.5 via transformers (CLS pool + L2-norm).
bge-small (33M, 384-dim) replaced bge-large (335M, 1024-dim): for matching short
spoken tails against <~120 cached phrases the retrieval-quality delta is
immaterial, but the per-query CPU encode is ~10x cheaper in the live loop.
The Encoder protocol is intentionally tiny so the backend can be swapped
(set ``model_name``) without touching the matcher.

smartppt is offline + CPU-only + deterministic by design: the model is loaded
with ``local_files_only`` (no network at runtime), pinned to a single revision,
and run single-threaded on CPU so the same spoken text always yields the same
match score regardless of host machine or load.
"""

from __future__ import annotations

import os

# Determinism / offline enforcement — set BEFORE numpy/torch/transformers import
# so the OpenMP/BLAS thread caps and HF offline mode take effect. Defaults only:
# an explicit environment override (e.g. a deliberate multi-thread benchmark) wins.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from typing import Protocol

import numpy as np


class Encoder(Protocol):
    """Minimal encoder interface used by the matcher."""

    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (N, dim) float32 array of L2-normalised embeddings."""
        ...


class BGEEncoder:
    """BGE-{small,large}-en-v1.5 via Hugging Face transformers.

    Cached at ~/.cache/huggingface/hub. Runs on CPU single-threaded and offline
    by default (the project treats the GPU as unusable and forbids runtime
    network access). Pass ``device='cuda'`` explicitly to override.
    """

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
    # Pin the exact snapshot per model so an HF revision/etag check can never
    # silently fetch different weights (which would change every match score).
    _REVISIONS = {
        "BAAI/bge-small-en-v1.5": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "BAAI/bge-large-en-v1.5": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
    }

    def __init__(self, model_name: str | None = None, device: str | None = None,
                 revision: str | None = None) -> None:
        import torch  # local import; encoder is heavy
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name or self.DEFAULT_MODEL
        # Pinned revision for known models; a custom model_name may pass its own.
        self.revision = revision or self._REVISIONS.get(self.model_name)
        self._torch = torch
        # CPU-only by default for determinism (CPU vs CUDA float kernels are not
        # bit-identical). Single-threaded so parallel reduction order can't vary.
        self.device = device or "cpu"
        torch.set_num_threads(1)
        torch.manual_seed(0)
        load_kw: dict = {"local_files_only": True}
        if self.revision:
            load_kw["revision"] = self.revision
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **load_kw)
        self.model = AutoModel.from_pretrained(self.model_name, **load_kw).to(self.device).eval()
        # hidden size: 384 (bge-small) / 1024 (bge-large)
        self.dim = int(self.model.config.hidden_size)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        torch = self._torch
        # BGE-v1.5 doesn't require an instruction prefix for English retrieval.
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**enc)
        # CLS pool
        cls = out.last_hidden_state[:, 0]
        cls = torch.nn.functional.normalize(cls, p=2, dim=1)
        return cls.cpu().numpy().astype(np.float32)


def load_default_encoder() -> Encoder:
    return BGEEncoder()
