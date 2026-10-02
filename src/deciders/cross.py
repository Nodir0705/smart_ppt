"""Cross-encoder decider: a re-ranker reads (clause, unit description) pairs
jointly and scores relevance — one forward pass over the slide's units per call.

unit description = summary + slide text + talk + terms;
"none" = ``none_logit``, a fixed relevance bar: the re-ranker's logits are
absolute and bimodal (unrelated pairs ~ -9..-12, a named unit ~ -5..+5), so
one bar between the modes works on any slide. -7 is the centre of the band
(-8..-5.5) that passed the gate on the tune talks.

Offline: the model must already be in the HF cache (download once).
"""
from __future__ import annotations

import os

# 2 layers x 128 wide: ~24 ms a call on one CPU thread for a 6-10 unit slide;
# MiniLM-L-2 133 ms, L-4 264 ms, L-6 387 ms, bge-reranker-base ~2 s (same gate on the tune talks)
DEFAULT_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
REVISIONS = {   # pinned snapshots, so a cache refresh can't change scores
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
    "BAAI/bge-reranker-base": "2cfc18c9415c912f9d8155881c133215df768a70",
    "cross-encoder/ms-marco-MiniLM-L-4-v2": "777b2f369bc1c2f850df8bd367ed1654bda4497b",
    "cross-encoder/ms-marco-MiniLM-L-2-v2": "1b5cd67b15209f24824c50370e0397743aa9b787",
    "cross-encoder/ms-marco-TinyBERT-L-2-v2": "81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc",
}


def describe(u: dict) -> str:
    parts = [u.get("summary") or "", u.get("text") or "", "; ".join(u.get("talk") or []),
             "; ".join(u.get("terms") or [])]
    return " | ".join(p for p in parts if p)


class CrossDecider:
    def __init__(self, model: str | None = None, device: str = "cpu", none_logit: float = -7.0,
                 with_context: bool = False) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.model_name = model or DEFAULT_MODEL
        kw = {"local_files_only": True, "revision": REVISIONS.get(self.model_name)}
        torch.set_num_threads(1)   # deterministic CPU, like the encoder
        self._torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(self.model_name, **kw)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, **kw).to(device).eval()
        self.none_logit = none_logit
        self.with_context = with_context
        self._docs_key, self._docs = None, []

    def decide(self, clause: str, context: str, units: list[dict]) -> dict[str, float]:
        key = tuple(u["uid"] for u in units), tuple(u.get("summary", "") for u in units)
        if key != self._docs_key:
            self._docs_key, self._docs = key, [describe(u) for u in units]
        q = f"{context} ... {clause}" if self.with_context and context else clause
        enc = self.tok([q] * len(units), self._docs, padding=True, truncation="only_second",
                       max_length=256, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            logits = self.model(**enc).logits[:, 0].float().cpu().tolist()
        out = {u["uid"]: s for u, s in zip(units, logits)}
        out["none"] = self.none_logit
        return out
