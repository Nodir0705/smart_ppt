"""Bi-encoder decider: bge-small cosine between the clause and each unit's
descriptions (slide text, summary, talk, terms) — the baseline that isolates
what clause segmentation alone buys.

unit score = max cosine over the unit's description phrases;
"none"     = max cosine to generic presenter talk (``matcher.NULL_SPEECH``).
The commit rule (``ClauseParams.commit``) is the margin over that null score.
"""
from __future__ import annotations

import numpy as np

from src.matcher import NULL_SPEECH


def unit_phrases(u: dict) -> list[str]:
    out = [u.get("text") or "", u.get("summary") or ""] + list(u.get("talk") or []) + list(u.get("terms") or [])
    return [p for p in dict.fromkeys(" ".join(p.lower().split()) for p in out) if p]


class BiDecider:
    def __init__(self, encoder) -> None:
        self.encoder = encoder
        self._null = encoder.encode(list(NULL_SPEECH))
        self._units_key = None
        self._embs: list[np.ndarray] = []

    def _unit_embs(self, units: list[dict]) -> list[np.ndarray]:
        key = tuple((u["uid"], tuple(unit_phrases(u))) for u in units)
        if key != self._units_key:   # a new slide: embed its units once
            self._units_key = key
            self._embs = [self.encoder.encode(list(ph)) for _uid, ph in key]
        return self._embs

    def decide(self, clause: str, context: str, units: list[dict]) -> dict[str, float]:
        del context   # ponytail: clause only; context matters for pronouns, which cosine can't resolve anyway
        q = self.encoder.encode([clause])[0]
        out = {u["uid"]: float((e @ q).max()) for u, e in zip(units, self._unit_embs(units))}
        out["none"] = float((self._null @ q).max())
        return out
