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

import dataclasses
import hashlib
import json
import re
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
    floor_id: str | None = None    # unit ref: the member whose null floor applies (None = own id)

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


# ---------- slide furniture (never pointed at) ----------

# A presenter never points at the deck's furniture: the running footer/caption
# repeated on most slides, or the page number in the slide's margin. Left
# pointable they collide with speech ("handout checker", "ten", "page") and light
# instead of content. Deterministic and deck-agnostic: decided from the index's
# own repetition + geometry, no wordlist.
FURNITURE_MIN_SHARE = 0.5   # running text: on at least this share of slides (and >= 3)
FURNITURE_EDGE = 0.15       # margin: bbox centre in the top/bottom 15% of the slide (or off it)


def furniture_ids(raw: dict) -> set[str]:
    """Ids of non-title elements in a slide's top/bottom FURNITURE_EDGE margin that are

    * running text — the same text (digits ignored, so "Page 7"/"Page 8" match)
      containing a letter, on >= FURNITURE_MIN_SHARE of the slides (min 3); or
    * a page number — a bare integer equal to its slide's number.

    Titles are never furniture: "Step 1" … "Step 5" headings repeat by the same
    rule but are the slide's subject.
    """
    slides = raw.get("slides", [])
    height = (raw.get("slide_size") or {}).get("h")
    if not height:
        return set()
    key = lambda t: re.sub(r"\d+", "#", " ".join(t.lower().split()))  # noqa: E731
    on: dict[str, set[int]] = {}
    for s in slides:
        for el in s["elements"]:
            t = (el.get("text") or "").strip()
            if re.search(r"[a-z]", t, re.I):
                on.setdefault(key(t), set()).add(s["slide_number"])
    running = {k for k, v in on.items() if len(v) >= max(3, FURNITURE_MIN_SHARE * len(slides))}
    out: set[str] = set()
    for s in slides:
        for el in s["elements"]:
            t, box = (el.get("text") or "").strip(), el.get("bbox")
            if not t or not box or el.get("type") == "title":
                continue
            cy = (box["y"] + box["h"] / 2) / height
            if FURNITURE_EDGE <= cy <= 1 - FURNITURE_EDGE:
                continue
            if key(t) in running or (t.isdecimal() and int(t) == s["slide_number"]):
                out.add(el["id"])
    return out


# ---------- pointable units (cards) ----------

# A presenter points at a visual unit — a card's heading + its description, a
# big number + its caption, a captioned picture — not at each text run inside
# it. Elements forming one unit share one highlight: the policy switches
# between units and draws the union box. Decided from the index's geometry and
# text at load (deck-agnostic, deterministic, no wordlist). Lengths are shares
# of the slide size.
UNIT_GAP = 0.04          # stacked: vertical gap <= 4% of slide height ...
UNIT_GAP_LINES = 1.5     # ... and <= 1.5x the upper element's line height
UNIT_ALIGN = 0.02        # same column: left edges or centres within 2% of slide width
HEAD_MAX_WORDS = 5       # a heading is short ...
HEAD_LARGER = 1.15       # ... and has fewer words than its body, or taller glyphs
BREAK_MAX_WORDS = 3      # a <=3-word element heading its own body starts a new unit
PANEL_MAX_AREA = 0.4     # a card's shape covers <= 40% of the slide ...
PANEL_MIN_HEIGHT = 3.0   # ... and is >= 3x as tall as the label text it holds
NEVER_GROUPED = {"title", "chart", "chart_category", "table", "table_cell"}


def _n_words(el: dict) -> int:
    return len(re.findall(r"\w+", el.get("text") or ""))


def _line_h(el: dict) -> float:
    return max([b["h"] for b in el.get("line_bboxes") or []] or [el["bbox"]["h"]])


def _heads(a: dict, b: dict) -> bool:
    """`a` reads as a heading of `b`: short, not a sentence or code line, and
    either fewer words or visibly larger type."""
    t = (a.get("text") or "").rstrip()
    return (_n_words(a) <= HEAD_MAX_WORDS and "=" not in t
            and not t.endswith((".", "!", "?", ";", ",", "{"))
            and (_n_words(a) < _n_words(b) or _line_h(a) >= HEAD_LARGER * _line_h(b)))


def _centre_in(box: dict, el: dict) -> bool:
    b = el["bbox"]
    cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
    return box["x"] <= cx <= box["x"] + box["w"] and box["y"] <= cy <= box["y"] + box["h"]


def _slide_units(elements: list[dict], W: float, H: float, skip: set[str]) -> dict[str, str]:
    els = [dict(e, _order=k) for k, e in enumerate(elements)]
    cand = sorted((e for e in els if e.get("trigger_phrases") and e.get("bbox")
                   and e["id"] not in skip and e["type"] not in NEVER_GROUPED),
                  key=lambda e: (e["bbox"]["y"], e["bbox"]["x"], e["_order"]))
    rank = {e["id"]: k for k, e in enumerate(cand)}
    blockers = [e for e in els if e.get("bbox") and e["type"] in NEVER_GROUPED - {"title"}]

    def gap(a: dict, b: dict) -> float | None:
        """Vertical gap when `b` sits directly below `a` in the same column."""
        A, B = a["bbox"], b["bbox"]
        g = B["y"] - (A["y"] + A["h"])
        if not -0.01 * H <= g <= min(UNIT_GAP * H, UNIT_GAP_LINES * _line_h(a)):
            return None
        same_col = (abs(A["x"] - B["x"]) <= UNIT_ALIGN * W
                    or abs(A["x"] + A["w"] / 2 - B["x"] - B["w"] / 2) <= UNIT_ALIGN * W)
        return g if same_col else None

    below: dict[str, dict] = {}
    for a in cand:
        opts = [(g, rank[b["id"]], b) for b in cand if b is not a and (g := gap(a, b)) is not None]
        if opts:
            below[a["id"]] = min(opts, key=lambda t: t[:2])[2]
    members = {e["id"]: [e] for e in cand}      # unit key -> members
    key = {e["id"]: e["id"] for e in cand}      # element id -> unit key

    def single(e: dict) -> bool:
        return len(members[key[e["id"]]]) == 1

    def join(a: dict, b: dict) -> None:
        ka, kb = key[a["id"]], key[b["id"]]
        if ka != kb:
            for m in members[kb]:
                key[m["id"]] = ka
            members[ka] += members.pop(kb)

    # 1. stack: a heading, the body under it, and what follows in that column
    #    until the gap opens or a short element heads its own body.
    for a in cand:
        b = below.get(a["id"])
        if (b is None or not single(a) or not single(b) or "image" in (a["type"], b["type"])
                or not _heads(a, b)):
            continue
        join(a, b)
        prev = b
        while (c := below.get(prev["id"])) is not None and single(c) and c["type"] != "image":
            n = below.get(c["id"])
            if n is not None and _n_words(c) <= BREAK_MAX_WORDS and _heads(c, n):
                break
            join(a, c)
            prev = c
    # 2. a pointable picture and its caption (<= 2 lines) directly above/below
    for im in cand:
        if im["type"] != "image" or not single(im):
            continue
        opts = []
        for t in cand:
            if t["type"] != "image" and single(t) and len(t.get("line_texts") or [1]) <= 2:
                g = gap(im, t)
                g = gap(t, im) if g is None else g
                if g is not None:
                    opts.append((g, rank[t["id"]], t))
        if opts:
            join(im, min(opts, key=lambda t: t[:2])[2])
    # 3. card: a label whose own shape is a panel around other elements joins
    #    them — unless the panel holds several multi-element units (then it is a
    #    board of cards, not one card). Smallest panels first, so cards form
    #    before the boards that contain them.
    panels = [p for p in cand if (sb := p.get("shape_bbox"))
              and sb["w"] * sb["h"] <= PANEL_MAX_AREA * W * H
              and sb["h"] >= PANEL_MIN_HEIGHT * p["bbox"]["h"] and _centre_in(sb, p)]
    panels.sort(key=lambda p: (p["shape_bbox"]["w"] * p["shape_bbox"]["h"], rank[p["id"]]))
    for p in panels:
        sb = p["shape_bbox"]
        inside = [e for e in cand if e is not p and _centre_in(sb, e)]
        if not inside or any(_centre_in(sb, x) for x in blockers):
            continue
        units = sorted({key[e["id"]] for e in inside} - {key[p["id"]]}, key=lambda k: rank[k])
        if any(not all(_centre_in(sb, m) for m in members[k]) for k in units):
            continue
        if sum(len(members[k]) > 1 for k in units) > 1:
            continue
        for k in units:
            join(p, members[k][0])
    out = {}
    for ms in members.values():
        head = min(ms, key=lambda e: e["_order"])["id"]   # first in reading order
        for m in ms:
            out[m["id"]] = head
    return out


def pointable_units(raw: dict) -> dict[str, str]:
    """element id -> unit id (the unit's first member in reading order) for every
    pointable element; a singleton maps to itself. Rules, per slide:

    * stack — a short heading directly above its body in the same column, plus
      the elements that continue that column (description, bullets);
    * picture — a pointable image and its <= 2-line caption directly above/below;
    * card — a label whose shape is a panel joins what the panel holds, unless
      the panel holds several multi-element units.

    Never grouped: titles, charts and their categories, tables and cells,
    furniture, non-pointable elements (icons), anything not adjacent.
    """
    size = raw.get("slide_size") or {}
    W, H = size.get("w"), size.get("h")
    if not W or not H:
        return {}
    skip = furniture_ids(raw)
    out: dict[str, str] = {}
    for s in raw.get("slides", []):
        out.update(_slide_units(s["elements"], W, H, skip))
    return out


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
        # Furniture keeps its rows (the cache is untouched) but gets an empty
        # span, so it is never pointable.
        furniture = furniture_ids(self.raw)
        for _embs, refs in self._by_slide.values():
            for r in refs:
                if r.id in furniture:
                    r.emb_end = r.emb_start
        # Pointable units (cards): element id -> unit id, decided post-load like
        # furniture, so the embedding cache is untouched.
        self.unit_of: dict[str, str] = pointable_units(self.raw)
        self.ref_by_id: dict[str, ElementRef] = {
            r.id: r for _embs, refs in self._by_slide.values() for r in refs}
        self._bbox_by_id = {el["id"]: el["bbox"] for s in self.raw["slides"]
                            for el in s["elements"] if el.get("bbox")}
        self._unit_members: dict[str, list[str]] = {}
        for eid, u in self.unit_of.items():
            self._unit_members.setdefault(u, []).append(eid)

    def unit_bbox(self, unit_id: str | None) -> dict | None:
        """Union bbox of a multi-element unit (None for a singleton / no unit)."""
        ms = self._unit_members.get(unit_id or "", [])
        if len(ms) < 2:
            return None
        bs = [self._bbox_by_id[m] for m in ms]
        x0, y0 = min(b["x"] for b in bs), min(b["y"] for b in bs)
        x1 = max(b["x"] + b["w"] for b in bs)
        y1 = max(b["y"] + b["h"] for b in bs)
        return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0, "units": bs[0].get("units", "inches")}

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
        # CPU and CUDA kernels aren't bit-identical: a non-CPU encoder gets its own
        # cache (CPU keys unchanged, so existing CPU caches stay valid).
        device = str(getattr(self.encoder, "device", "cpu"))
        if device != "cpu":
            h.update(f"|{device}".encode())
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

# Per-phrase scoring window (multi-scale tail). A trigger phrase of n words is
# scored against only the last max(PHRASE_WINDOW_MIN, n + 1) spoken words, so a
# 2-word label isn't diluted by six older words still lingering in the tail,
# while a long phrase still sees the whole tail. Structural, not tuned per deck.
PHRASE_WINDOW_MIN = 4


def phrase_window(phrase: str) -> int:
    return max(PHRASE_WINDOW_MIN, len(phrase.split()) + 1)


# Null hypothesis: generic presenter talk that names nothing on any slide. An
# element's null floor is its best similarity to any of these lines; it may only
# light when the spoken tail beats that floor by PolicyConfig.null_margin. Short,
# generic labels ("When", "2", "Display", "Thank you!") sit close to ANY sentence
# and so get a higher bar than specific ones. Fixed text, never tuned per deck.
NULL_SPEECH = (
    "Okay, let's get started.",
    "So here is what I wanted to show you.",
    "This is a key point, so pay attention.",
    "Moving on.",
    "Does anyone have a question about that?",
    "Let's go to the next slide.",
    "As I mentioned earlier, this matters a lot.",
    "Well, um, you know, basically.",
    "Good, so that covers the basics.",
    "Now, look at this.",
)


class Matcher:
    QUERY_CACHE_SIZE = 16

    def __init__(self, index: Index, encoder: Encoder) -> None:
        self.index = index
        self.encoder = encoder
        # Small query cache: one step encodes several windows of the same tail,
        # and the live loop re-checks identical tails while speech is idle.
        # encode() is pure, so a cache hit is behaviour-preserving.
        self._query_cache: dict[str, np.ndarray] = {}
        # slide -> per-phrase window length, aligned with the embedding rows
        self._windows: dict[int, np.ndarray] = {
            sno: np.array([phrase_window(p) for r in refs for p in r.phrases], dtype=np.int64)
            for sno, (_embs, refs) in index._by_slide.items()
        }
        null = encoder.encode(list(NULL_SPEECH))   # (n_null, dim), once per load
        self.null_floor: dict[str, float] = {
            r.id: float((embs[r.emb_start:r.emb_end] @ null.T).max())
            for embs, refs in index._by_slide.values() for r in refs if r.is_pointable
        }

    def _encode_query(self, spoken: str) -> np.ndarray:
        vec = self._query_cache.get(spoken)
        if vec is None:
            if len(self._query_cache) >= self.QUERY_CACHE_SIZE:
                self._query_cache.clear()
            vec = self._query_cache[spoken] = self.encoder.encode([spoken])[0]  # (dim,)
        return vec

    def scored(self, spoken: str, slide_no: int) -> list[tuple[float, ElementRef]]:
        """Per-element raw scores (MAX over phrases), sorted high→low: the whole
        text against every phrase. Stable on ties so element order is preserved.
        Empty when nothing to match."""
        return self._scored(spoken, slide_no, windowed=False)

    def scored_tail(self, spoken: str, slide_no: int) -> list[tuple[float, ElementRef]]:
        """Like ``scored`` but for a live STT tail: each phrase is scored against
        only its own window of the most recent words (``phrase_window``), so old
        words lingering in the tail can't dilute a short mention."""
        return self._scored(spoken, slide_no, windowed=True)

    def _scored(self, spoken: str, slide_no: int, windowed: bool) -> list[tuple[float, ElementRef]]:
        spoken = (spoken or "").strip()
        if not spoken or slide_no not in self.index._by_slide:
            return []
        embs, refs = self.index._by_slide[slide_no]
        if embs.shape[0] == 0:
            return []
        sims = embs @ self._encode_query(spoken)  # (n_phrases,) — whole text
        if windowed:
            words = spoken.split()
            win = self._windows[slide_no]
            for k in np.unique(win[win < len(words)]):   # phrases wanting a shorter window
                sel = win == k
                sims[sel] = embs[sel] @ self._encode_query(" ".join(words[-int(k):]))
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
    gap: float = 0.03             # best must beat runner-up by at least this (on effective score; units compete, not card parts)
    stickiness: int = 1           # consecutive frames needed before switching active element (raise to 2–3 for streaming partials)
    clear_after_n_low: int = 8    # consecutive low-confidence frames before clearing (hold through brief mis-hear gaps; a card stays lit while its parts are talked through)
    # --- group-aware temporal bonuses (added to raw score to form the effective
    #     score the gap/ranking operate on; threshold still applies to RAW score
    #     so bonuses never promote a genuinely-irrelevant element) ---
    stay_bonus: float = 0.06          # active element resists a same-group sibling (intra-group anti-flicker)
    same_group_bonus: float = 0.04    # pull attention into the active group (e.g. keep "match" in the latency chart)
    # --- short-tail override (multi-scale tail). The full tail stays primary; the
    #     last `short_words` words alone may name a DIFFERENT element and win, but
    #     only on a stricter raw bar (plus the usual `gap`, on raw short scores).
    #     Catches 1-2 word mentions ("Stage", "8 GB") that the older words in the
    #     tail would otherwise outvote. 0 disables. ---
    short_words: int = 2
    short_threshold: float = 0.73
    # --- revert guard: after switching A -> B, A may not come back for the next
    #     `revert_guard` steps (A -> B -> A bounce is the flicker a short-scale
    #     switch would otherwise cause while A's words linger). 0 disables. ---
    revert_guard: int = 8
    # --- null-hypothesis bar: an element must also beat its own similarity to
    #     generic presenter talk (Matcher.null_floor) by this much, so filler
    #     ("as you can see here...") can't light short/generic labels. ---
    null_margin: float | None = 0.12   # None disables
    # --- short-override confirmation: a switch won by the short-tail override
    #     (1-2 words, the hop-prone path) needs the same element to be the
    #     confident winner on this many consecutive updates. Full-tail switches
    #     stay instant. 1 disables. ---
    short_confirm: int = 2
    # --- stale release: the lit element is cleared after this many consecutive
    #     updates on which its own full-tail score is below its bar (the talk has
    #     moved on), unless that update switches to something else. 0 disables.
    #     Off by default since pointable units: the released card was mostly
    #     re-lit moments later (~1 in 10 AI_Handout switches). ---
    stale_release: int = 3


def _floor(ref: ElementRef) -> str:
    """Id whose null floor gates `ref` (a unit ref carries its winning member)."""
    return getattr(ref, "floor_id", None) or ref.id


class MatchPolicy:
    """Wraps a Matcher with hysteresis/debounce so the highlight doesn't flicker.

    State:
      active_id   — currently highlighted element id (or None)
      candidate   — element being "voted in" by recent strong matches
      cand_count  — consecutive matches for `candidate`
      low_count   — consecutive low-confidence matches (drives clear)
      displaced   — element the last switch left; barred for `revert_guard` steps

    `spoken` is the full (≤ tail_words) tail; the short-tail scale is derived
    from it here, after normalization, so every caller gets identical
    multi-scale decisions.

    The policy moves between pointable UNITS (``pointable_units``): ids it
    emits are unit ids (the unit's first member; a singleton's own id), so
    talking about any part of a card keeps one steady highlight.

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
        self._displaced: str | None = None   # element the last switch moved away from
        self._guard_left = 0                 # steps it stays barred from returning
        self._recent: list[str | None] = []  # confident winner per recent step (None = none)
        self._stale = 0                      # consecutive steps the active lacked support

    def reset(self) -> None:
        self.active_id = None
        self._active_group = None
        self._candidate = None
        self._cand_count = 0
        self._low_count = 0
        self._displaced = None
        self._guard_left = 0
        self._recent = []
        self._stale = 0

    def _remember(self, winner: str | None) -> None:
        self._recent = (self._recent + [winner])[-max(1, self.cfg.short_confirm):]

    def _score(self, spoken: str, slide_no: int) -> list[tuple[float, ElementRef]]:
        """Scores per pointable UNIT, high→low: a unit (card) takes its best
        member's score and is represented by its head ref, carrying that
        member's null floor (`floor_id`). Singletons pass through unchanged."""
        # the real Matcher windows the tail per phrase; scripted stubs only expose scored()
        fn = getattr(self.matcher, "scored_tail", None) or self.matcher.scored
        per_el = fn(spoken, slide_no)
        index = getattr(self.matcher, "index", None)
        unit_of = getattr(index, "unit_of", None)
        if not unit_of:
            return per_el
        out, seen = [], set()
        for s, r in per_el:
            u = unit_of.get(r.id, r.id)
            if u in seen:
                continue
            seen.add(u)
            out.append((s, r if u == r.id else dataclasses.replace(index.ref_by_id[u], floor_id=r.id)))
        return out

    def _bar(self, element_id: str | None, base: float) -> float:
        """Raw score `element_id` needs: `base`, raised to its null floor + margin."""
        floor = getattr(self.matcher, "null_floor", {}).get(element_id)
        if floor is None or self.cfg.null_margin is None:
            return base
        return max(base, floor + self.cfg.null_margin)

    def _short_override(self, spoken: str, slide_no: int, res: MatchResult) -> MatchResult:
        """Return a result for the short-tail winner when it clears the stricter
        bar and names an element other than the full-tail best and the active one;
        otherwise `res` unchanged."""
        k = self.cfg.short_words
        words = spoken.split()
        if k <= 0 or len(words) <= k:
            return res
        per_el = self._score(" ".join(words[-k:]), slide_no)
        if not per_el:
            return res
        best_raw, best = per_el[0]
        ru_raw, ru_id = (per_el[1][0], per_el[1][1].id) if len(per_el) > 1 else (0.0, None)
        if (best_raw < self._bar(_floor(best), self.cfg.short_threshold) or best_raw - ru_raw < self.cfg.gap
                or best.id in (res.element_id, self.active_id)):
            return res
        out = MatchResult(element_id=best.id, type=best.type, text=best.text,
                          score=best_raw, runner_up_score=ru_raw,
                          runner_up_id=ru_id, slide_no=slide_no)
        out._eff_best, out._eff_runner, out._best_group = best_raw, ru_raw, best.group  # type: ignore[attr-defined]
        out._floor = _floor(best)  # type: ignore[attr-defined]
        return out

    def _ranked(self, spoken: str, slide_no: int) -> tuple[MatchResult, list[tuple[float, ElementRef]]]:
        """Re-rank raw scores by an effective score that rewards staying on the
        current element / group. Returns (MatchResult-for-effective-best, ranked).
        `ranked` items are (raw_score, ref) sorted by EFFECTIVE score, high→low.
        """
        per_el = self._score(spoken, slide_no)
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
        res._floor = _floor(best)                   # type: ignore[attr-defined]
        return res, ranked

    def _supported(self, ranked: list[tuple[float, ElementRef]]) -> bool | None:
        """Whether the active unit's full-tail score clears its own bar
        (None when it isn't scored at all)."""
        act = next(((s, r) for s, r in ranked if r.id == self.active_id), None)
        if act is None:
            return None
        return act[0] >= self._bar(_floor(act[1]), self.cfg.threshold)

    def step(self, spoken: str, slide_no: int) -> tuple[MatchResult, str | None, str]:
        if self.normalizer is not None:
            spoken = self.normalizer(spoken)
        # Group-aware rerank when the matcher exposes scored() (the real Matcher);
        # otherwise fall back to the plain match() result (legacy / scripted stubs).
        ranked: list[tuple[float, ElementRef]] = []
        if hasattr(self.matcher, "scored"):
            res, ranked = self._ranked(spoken, slide_no)
            full = res
            res = self._short_override(spoken, slide_no, res)
            is_short = res is not full
        else:
            res = self.matcher.match(spoken, slide_no)
            is_short = False
        barred = False
        if self._guard_left > 0:
            self._guard_left -= 1
            barred = res.element_id == self._displaced
        if barred and self.active_id is not None:
            # The displaced element topping the tail is the bounce the guard
            # exists to stop, not evidence the active one went stale: hold it
            # while it still clears its own bar (no step toward clearing).
            if self._supported(ranked):
                self._remember(None)
                self._candidate = None
                self._cand_count = 0
                return res, self.active_id, "hold"
        confident = (
            not barred
            and res.element_id is not None
            and res.score >= self._bar(getattr(res, "_floor", res.element_id), self.cfg.threshold)
            and (getattr(res, "_eff_best", res.score) - getattr(res, "_eff_runner", res.runner_up_score)) >= self.cfg.gap
        )
        self._remember(res.element_id if confident else None)
        # Stale release: the active's own full-tail support is gone for
        # `stale_release` updates and nothing else is taking over this update.
        if self.active_id is not None and self.cfg.stale_release > 0 and hasattr(self.matcher, "scored"):
            if not self._supported(ranked):
                self._stale += 1
            else:
                self._stale = 0
            if self._stale >= self.cfg.stale_release and not (confident and res.element_id != self.active_id):
                self.active_id = None
                self._active_group = None
                self._stale = 0
                return res, None, "clear"
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
        # a short-override win must repeat: the confident winner on each of the
        # last `short_confirm` steps
        confirmed = (not is_short or self.cfg.short_confirm <= 1
                     or (len(self._recent) >= self.cfg.short_confirm
                         and all(x == res.element_id for x in self._recent)))
        if self._cand_count >= self.cfg.stickiness and confirmed:
            if self.active_id is not None and self.cfg.revert_guard > 0:
                self._displaced, self._guard_left = self.active_id, self.cfg.revert_guard
            self.active_id = self._candidate
            self._active_group = getattr(res, "_best_group", None)
            self._candidate = None
            self._cand_count = 0
            self._stale = 0
            return res, self.active_id, "switch"
        return res, self.active_id, "hold"
