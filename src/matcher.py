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
import sys
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
    group_id: str | None = None    # group this element belongs to (parent chart, etc.)
    group_label: str | None = None
    order: int = 0                 # position within the slide (reading order)

    @property
    def is_pointable(self) -> bool:
        return self.emb_end > self.emb_start

    @property
    def group(self) -> str:
        """Effective group key — the element's own id when it has no group."""
        return self.group_id or self.id


@dataclass
class MatchResult:
    element_id: str | None
    type: str | None
    text: str | None
    score: float
    runner_up_score: float
    runner_up_id: str | None
    slide_no: int


def best_index(preferred: str | Path | None = None, out_dir: str | Path = "out") -> Path:
    """Resolve which index file to use.

    An explicit `preferred` path wins; otherwise return the richest available
    among scoped > refined > base (so the live app gets the Phase-1 grounding
    improvements by default while still working with only the base index).
    """
    if preferred:
        return Path(preferred)
    out = Path(out_dir)
    order = ("index.scoped.json", "index.refined.json", "index.json")
    for i, name in enumerate(order):
        p = out / name
        if not p.exists():
            continue
        # Warn if a more-basic sibling is newer than the chosen file: that means
        # the deck/parse was rebuilt but the richer stage wasn't re-run, so the
        # chosen index is stale. Rebuild with `python3 -m src.build_index`.
        chosen_mtime = p.stat().st_mtime
        for upstream in order[i + 1:]:
            up = out / upstream
            if up.exists() and up.stat().st_mtime > chosen_mtime:
                print(
                    f"warning: {p} is older than {up} — it may be stale; "
                    f"rebuild with `python3 -m src.build_index <deck.pptx>`",
                    file=sys.stderr,
                )
                break
        return p
    return out / "index.json"


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
    # Bump CACHE_VERSION whenever the ElementRef schema changes so stale pickles
    # (missing new fields) are rebuilt instead of unpickled into the new dataclass.
    CACHE_VERSION = "v3-cpu-deterministic"

    def _cache_key(self) -> str:
        payload = self.index_path.read_bytes()
        model_name = getattr(self.encoder, "model_name", type(self.encoder).__name__)
        h = hashlib.sha256()
        h.update(payload)
        h.update(b"|")
        h.update(model_name.encode())
        h.update(b"|")
        h.update(self.CACHE_VERSION.encode())
        return h.hexdigest()[:16]

    # ElementRef fields persisted to the JSON sidecar (embeddings live in the .npz).
    _META_FIELDS = (
        "id", "type", "text", "slide_no", "phrases",
        "emb_start", "emb_end", "group_id", "group_label", "order",
    )
    KEEP_CACHES = 3  # most-recent cache sets to retain (age eviction)

    def _cache_stem(self) -> str:
        return f"embeddings_{self._cache_key()}"

    def _cache_path(self) -> Path:
        return self.cache_dir / f"{self._cache_stem()}.npz"

    def _meta_path(self) -> Path:
        return self.cache_dir / f"{self._cache_stem()}.meta.json"

    def _build_cache_or_load(self) -> None:
        cache_path, meta_path = self._cache_path(), self._meta_path()
        if cache_path.exists() and meta_path.exists():
            self._load_cache(cache_path, meta_path)
            return
        self._build_from_scratch()
        self._write_cache(cache_path, meta_path)
        self._evict_old_caches()

    def _load_cache(self, cache_path: Path, meta_path: Path) -> None:
        meta = json.loads(meta_path.read_text())
        with np.load(cache_path) as npz:
            for sno_str, refs_raw in meta["slides"].items():
                sno = int(sno_str)
                embs = npz[f"s{sno}"]
                refs = [ElementRef(**{k: r[k] for k in self._META_FIELDS}) for r in refs_raw]
                self._by_slide[sno] = (embs, refs)

    def _write_cache(self, cache_path: Path, meta_path: Path) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        meta: dict = {"slides": {}}
        for sno in sorted(self._by_slide):  # sorted -> deterministic layout
            embs, refs = self._by_slide[sno]
            arrays[f"s{sno}"] = np.ascontiguousarray(embs, dtype=np.float32)
            meta["slides"][str(sno)] = [
                {k: getattr(ref, k) for k in self._META_FIELDS} for ref in refs
            ]
        # Loaded arrays are bit-identical to what was saved, so match decisions
        # stay deterministic (the .npz zip's own mtimes are irrelevant — nothing
        # hashes the cache file). No pickle -> no arbitrary-code-execution format.
        np.savez(cache_path, **arrays)
        meta_path.write_text(json.dumps(meta, sort_keys=True, ensure_ascii=False))

    def _evict_old_caches(self) -> None:
        """Keep only the most-recent KEEP_CACHES sets; drop older ones + legacy pickles."""
        for legacy in self.cache_dir.glob("embeddings_*.pkl"):
            legacy.unlink(missing_ok=True)  # old format — never read again
        npzs = sorted(self.cache_dir.glob("embeddings_*.npz"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in npzs[self.KEEP_CACHES:]:
            stale.unlink(missing_ok=True)
            stale.with_name(stale.stem + ".meta.json").unlink(missing_ok=True)

    def _build_from_scratch(self) -> None:
        for slide in self.raw["slides"]:
            sno = slide["slide_number"]
            phrases: list[str] = []
            refs: list[ElementRef] = []
            for order, el in enumerate(slide["elements"]):
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
                    group_id=el.get("group_id"),
                    group_label=el.get("group_label"),
                    order=order,
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
        # 1-entry query cache: the live loop re-checks the same STT tail across
        # throttle windows whenever speech is idle/repeats. encode() is pure, so
        # returning the cached vector for an identical tail is behaviour-preserving.
        self._last_query_text: str | None = None
        self._last_query_vec: np.ndarray | None = None

    def _encode_query(self, spoken: str) -> np.ndarray:
        if spoken != self._last_query_text:
            self._last_query_vec = self.encoder.encode([spoken])[0]  # (dim,)
            self._last_query_text = spoken
        return self._last_query_vec

    def scored(self, spoken: str, slide_no: int) -> list[tuple[float, ElementRef]]:
        """Per-element raw scores (MAX over phrases), sorted high→low.

        Stable on ties so element order is preserved. Empty when nothing to match.
        """
        spoken = (spoken or "").strip()
        if not spoken or slide_no not in self.index._by_slide:
            return []
        embs, refs = self.index._by_slide[slide_no]
        if embs.shape[0] == 0:
            return []
        q = self._encode_query(spoken)  # (dim,), cached for repeat tails
        sims = embs @ q  # (n_phrases,)
        per_el: list[tuple[float, ElementRef]] = []
        for ref in refs:
            if not ref.is_pointable:
                continue
            s = float(sims[ref.emb_start:ref.emb_end].max())
            per_el.append((s, ref))
        per_el.sort(key=lambda t: t[0], reverse=True)
        return per_el

    def match(self, spoken: str, slide_no: int) -> MatchResult:
        per_el = self.scored(spoken, slide_no)
        if not per_el:
            return MatchResult(None, None, None, 0.0, 0.0, None, slide_no)
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

    threshold: float = 0.65       # minimum cosine sim to consider any emission (bge-small: its cosines run ~0.05-0.10 higher than bge-large, so the bar is raised to keep filler-rejection equivalent)
    gap: float = 0.03             # best must beat runner-up by at least this (on effective score)
    stickiness: int = 1           # consecutive frames needed before switching active element (raise to 2–3 for streaming partials)
    clear_after_n_low: int = 8    # consecutive low-confidence frames before clearing (hold through brief mis-hear gaps so group stickiness survives)
    # --- group-aware temporal bonuses (added to raw score to form the effective
    #     score the gap/ranking operate on; threshold still applies to RAW score
    #     so bonuses never promote a genuinely-irrelevant element) ---
    stay_bonus: float = 0.06          # active element resists a same-group sibling (intra-group anti-flicker)
    same_group_bonus: float = 0.04    # pull attention into the active group (e.g. keep "match" in the latency chart)


class MatchPolicy:
    """Wraps a Matcher with hysteresis/debounce so the highlight doesn't flicker.

    State:
      active_id   — currently highlighted element id (or None)
      candidate   — element being "voted in" by recent strong matches
      cand_count  — consecutive matches for `candidate`
      low_count   — consecutive low-confidence matches (drives clear)

    Returns (emitted_id, why) where `why` is "hold" | "switch" | "clear" | "noop".
    """

    def __init__(self, matcher: Matcher, config: PolicyConfig | None = None,
                 normalizer=None) -> None:
        self.matcher = matcher
        self.cfg = config or PolicyConfig()
        self.normalizer = normalizer   # optional callable: spoken_text -> spoken_text
        self.active_id: str | None = None
        self._active_group: str | None = None
        self._candidate: str | None = None
        self._cand_count = 0
        self._low_count = 0

    def reset(self) -> None:
        self.active_id = None
        self._active_group = None
        self._candidate = None
        self._cand_count = 0
        self._low_count = 0

    def _ranked(self, spoken: str, slide_no: int) -> tuple[MatchResult, list[tuple[float, ElementRef]]]:
        """Re-rank raw scores by an effective score that rewards staying on the
        current element / group. Returns (MatchResult-for-effective-best, ranked).
        `ranked` items are (raw_score, ref) sorted by EFFECTIVE score, high→low.
        """
        per_el = self.matcher.scored(spoken, slide_no)
        if not per_el:
            return MatchResult(None, None, None, 0.0, 0.0, None, slide_no), []

        # The active element only resists a SAME-GROUP sibling (intra-group
        # flicker). When the strongest rival is in a different group it's a real
        # topic change, so the active yields freely (no stay bonus) — this keeps
        # legitimate switches (e.g. between diagram nodes) working.
        top_rival = next((r for _s, r in per_el if r.id != self.active_id), None)
        challenged_same_group = (
            self.active_id is not None
            and top_rival is not None
            and self._active_group is not None
            and top_rival.group == self._active_group
        )

        def effective(raw: float, ref: ElementRef) -> float:
            if ref.id == self.active_id:
                return raw + (self.cfg.stay_bonus if challenged_same_group else 0.0)
            if self._active_group is not None and ref.group == self._active_group:
                return raw + self.cfg.same_group_bonus
            return raw

        ranked = sorted(per_el, key=lambda t: effective(*t), reverse=True)
        best_raw, best = ranked[0]
        if len(ranked) > 1:
            ru_raw, ru = ranked[1]
            ru_id, ru_eff = ru.id, effective(ru_raw, ru)
        else:
            ru_id, ru_eff = None, 0.0
        res = MatchResult(
            element_id=best.id, type=best.type, text=best.text,
            score=best_raw, runner_up_score=ru_raw if len(ranked) > 1 else 0.0,
            runner_up_id=ru_id, slide_no=slide_no,
        )
        # stash effective gap on the result via attributes the step() closure reads
        res._eff_best = effective(best_raw, best)   # type: ignore[attr-defined]
        res._eff_runner = ru_eff                    # type: ignore[attr-defined]
        res._best_group = best.group                # type: ignore[attr-defined]
        return res, ranked

    def step(self, spoken: str, slide_no: int) -> tuple[MatchResult, str | None, str]:
        if self.normalizer is not None:
            spoken = self.normalizer(spoken)
        # Group-aware rerank when the matcher exposes scored() (the real Matcher);
        # otherwise fall back to the plain match() result (legacy / scripted stubs).
        if hasattr(self.matcher, "scored"):
            res, _ranked = self._ranked(spoken, slide_no)
        else:
            res = self.matcher.match(spoken, slide_no)
        confident = (
            res.element_id is not None
            and res.score >= self.cfg.threshold
            and (getattr(res, "_eff_best", res.score) - getattr(res, "_eff_runner", res.runner_up_score)) >= self.cfg.gap
        )
        if not confident:
            self._low_count += 1
            self._candidate = None
            self._cand_count = 0
            if self.active_id is not None and self._low_count >= self.cfg.clear_after_n_low:
                self.active_id = None
                self._active_group = None
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
            self._active_group = getattr(res, "_best_group", None)
            self._candidate = None
            self._cand_count = 0
            return res, self.active_id, "switch"
        return res, self.active_id, "hold"
