"""Local sentence-embedding wrapper.

Default backend: BAAI/bge-large-en-v1.5 via transformers (CLS pool + L2-norm).
The Encoder protocol is intentionally tiny so the backend can be swapped for
a smaller model (e.g. MiniLM) without touching the matcher.
"""

from __future__ import annotations

from typing import Protocol, Iterable

import numpy as np


class Encoder(Protocol):
    """Minimal encoder interface used by the matcher."""

    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (N, dim) float32 array of L2-normalised embeddings."""
        ...


class BGEEncoder:
    """BGE-large-en-v1.5 via Hugging Face transformers.

    Cached at ~/.cache/huggingface/hub. Runs on GPU if available, else CPU.
    """

    DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        import torch  # local import; encoder is heavy
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name or self.DEFAULT_MODEL
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        # BGE-large hidden size is 1024
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
