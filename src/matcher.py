"""Stage 3 — Matcher (live, fast).

Pipeline:

    Index (loaded from out/index.json)
       └── per-element trigger_phrases → precomputed embeddings (E)
    spoken_text + slide_no
       └── encoder.encode(spoken) → query vector q
       └── for each element on that slide: max( q · phrase_emb ) → score
       └── return ranked candidates
    MatchPolicy wraps that with threshold + hysteresis + stickiness.

Performance: candidates per slide are < 30, so live cost is one query encode
plus a small dot-product. Element embeddings are computed once at load and
cached on disk keyed by index path + mtime.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .encoder import Encoder


# ---------- data types ----------

@dataclass
class ElementRef:
    """Light reference into the index, plus the embedding span for this element."""

    id: str
    type: str
    text: str | None
    slide_no: int
    phrases: list[str]
    emb_start: int  # inclusive index into the slide's embedding matrix
    emb_end: int    # exclusive

    @property
    def is_pointable(self) -> bool:
        return self.emb_end > self.emb_start


@dataclass
class MatchResult:
    element_id: str | None
    type: str | None
    text: str | None
    score: float
    runner_up_score: float
    runner_up_id: str | None
    slide_no: int


# ---------- index loader with embedding cache ----------

class Index:
    """Loads index.json and precomputes per-element phrase embeddings.

    Embeddings are cached at out/embeddings.pkl, keyed by a content hash so
    parser changes invalidate the cache automatically.
    """

    def __init__(self, index_path: str | Path, encoder: Encoder, cache_dir: str | Path | None = None) -> None:
        self.index_path = Path(index_path)
        self.encoder = encoder
        self.cache_dir = Path(cache_dir) if cache_dir else self.index_path.parent
        self.raw: dict = json.loads(self.index_path.read_text())
        # one EmbeddingMatrix and ElementRef list per slide_no
        self._by_slide: dict[int, tuple[np.ndarray, list[ElementRef]]] = {}
        self._build_cache_or_load()

    # ---- cache key based on index content + encoder model ----
    def _cache_key(self) -> str:
        payload = self.index_path.read_bytes()
        model_name = getattr(self.encoder, "model_name", type(self.encoder).__name__)
        h = hashlib.sha256()
        h.update(payload)
        h.update(b"|")
        h.update(model_name.encode())
        return h.hexdigest()[:16]

    def _cache_path(self) -> Path:
        return self.cache_dir / f"embeddings_{self._cache_key()}.pkl"

    def _build_cache_or_load(self) -> None:
        cache_path = self._cache_path()
        if cache_path.exists():
            with cache_path.open("rb") as f:
                self._by_slide = pickle.load(f)
            return
        self._build_from_scratch()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(self._by_slide, f)

    def _build_from_scratch(self) -> None:
        for slide in self.raw["slides"]:
            sno = slide["slide_number"]
            phrases: list[str] = []
            refs: list[ElementRef] = []
            for el in slide["elements"]:
                el_phrases = list(el.get("trigger_phrases") or [])
                start = len(phrases)
                phrases.extend(el_phrases)
                end = len(phrases)
                refs.append(ElementRef(
                    id=el["id"],
                    type=el["type"],
                    text=el.get("text"),
                    slide_no=sno,
                    phrases=el_phrases,
                    emb_start=start,
                    emb_end=end,
                ))
            embs = self.encoder.encode(phrases) if phrases else np.zeros((0, self.encoder.dim), dtype=np.float32)
            self._by_slide[sno] = (embs, refs)

    # ---- public ----
    def slide_numbers(self) -> list[int]:
        return sorted(self._by_slide.keys())

    def elements_on(self, slide_no: int) -> list[ElementRef]:
        return self._by_slide[slide_no][1]


# ---------- matcher ----------

class Matcher:
    def __init__(self, index: Index, encoder: Encoder) -> None:
        self.index = index
        self.encoder = encoder

    def match(self, spoken: str, slide_no: int) -> MatchResult:
        spoken = (spoken or "").strip()
        if not spoken or slide_no not in self.index._by_slide:
            return MatchResult(None, None, None, 0.0, 0.0, None, slide_no)
        embs, refs = self.index._by_slide[slide_no]
        if embs.shape[0] == 0:
            return MatchResult(None, None, None, 0.0, 0.0, None, slide_no)
        q = self.encoder.encode([spoken])[0]  # (dim,)
        sims = embs @ q  # (n_phrases,)
        # per-element MAX over its phrases
        per_el: list[tuple[float, ElementRef]] = []
        for ref in refs:
            if not ref.is_pointable:
                continue
            s = float(sims[ref.emb_start:ref.emb_end].max())
            per_el.append((s, ref))
        if not per_el:
            return MatchResult(None, None, None, 0.0, 0.0, None, slide_no)
        per_el.sort(key=lambda t: t[0], reverse=True)
        best_s, best = per_el[0]
        if len(per_el) > 1:
            ru_s, ru = per_el[1]
            ru_id = ru.id
        else:
            ru_s, ru_id = 0.0, None
        return MatchResult(
            element_id=best.id,
            type=best.type,
            text=best.text,
            score=best_s,
            runner_up_score=ru_s,
            runner_up_id=ru_id,
            slide_no=slide_no,
        )


# ---------- confidence + temporal policy ----------

@dataclass
class PolicyConfig:
    """Tunables for emit/hold decisions."""

    threshold: float = 0.55       # minimum cosine sim to consider any emission
    gap: float = 0.03             # best must beat runner-up by at least this
    stickiness: int = 1           # consecutive frames needed before switching active element (raise to 2–3 for streaming partials)
    clear_after_n_low: int = 4    # consecutive low-confidence frames before clearing the active element


class MatchPolicy:
    """Wraps a Matcher with hysteresis/debounce so the highlight doesn't flicker.

    State:
      active_id   — currently highlighted element id (or None)
      candidate   — element being "voted in" by recent strong matches
      cand_count  — consecutive matches for `candidate`
      low_count   — consecutive low-confidence matches (drives clear)

    Returns (emitted_id, why) where `why` is "hold" | "switch" | "clear" | "noop".
    """

    def __init__(self, matcher: Matcher, config: PolicyConfig | None = None) -> None:
        self.matcher = matcher
        self.cfg = config or PolicyConfig()
        self.active_id: str | None = None
        self._candidate: str | None = None
        self._cand_count = 0
        self._low_count = 0

    def reset(self) -> None:
        self.active_id = None
        self._candidate = None
        self._cand_count = 0
        self._low_count = 0

    def step(self, spoken: str, slide_no: int) -> tuple[MatchResult, str | None, str]:
        res = self.matcher.match(spoken, slide_no)
        confident = (
            res.element_id is not None
            and res.score >= self.cfg.threshold
            and (res.score - res.runner_up_score) >= self.cfg.gap
        )
        if not confident:
            self._low_count += 1
            self._candidate = None
            self._cand_count = 0
            if self.active_id is not None and self._low_count >= self.cfg.clear_after_n_low:
                self.active_id = None
                return res, None, "clear"
            return res, self.active_id, "hold" if self.active_id else "noop"
        # confident match
        self._low_count = 0
        if res.element_id == self.active_id:
            # already highlighting it; reset candidate
            self._candidate = None
            self._cand_count = 0
            return res, self.active_id, "hold"
        # new contender
        if res.element_id == self._candidate:
            self._cand_count += 1
        else:
            self._candidate = res.element_id
            self._cand_count = 1
        if self._cand_count >= self.cfg.stickiness:
            self.active_id = self._candidate
            self._candidate = None
            self._cand_count = 0
            return res, self.active_id, "switch"
        return res, self.active_id, "hold"
