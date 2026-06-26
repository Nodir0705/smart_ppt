"""asr_normalize.py — deterministic, offline, CPU-only post-STT text normalizer.

Rewrites known mis-hearings / phonetic / spelled-out forms back to the
canonical deck term BEFORE embedding-based matching. Engine-agnostic, no
neural model, pure functions (except load_aliases which reads a file).
"""

from __future__ import annotations

import json
import re
from itertools import product
from pathlib import Path

# ---------------------------------------------------------------------------
# Module-level constant: per-letter spelled/phonetic sound(s).
# Used to synthesise spaced spelled-letter variants for ALL-CAPS acronyms.
# ---------------------------------------------------------------------------
LETTER_SOUNDS: dict[str, list[str]] = {
    "a": ["a"], "b": ["be", "bee"], "c": ["see", "sea"], "d": ["de", "dee"],
    "e": ["e"], "f": ["ef"], "g": ["gee", "g"], "h": ["aitch"], "i": ["i"],
    "j": ["jay"], "k": ["kay"], "l": ["el"], "m": ["em"], "n": ["en"],
    "o": ["oh"], "p": ["pe", "pee"], "q": ["cue"], "r": ["are", "ar"],
    "s": ["es", "ess"], "t": ["te", "tee"], "u": ["you"], "v": ["ve", "vee"],
    "w": ["double you"], "x": ["ex"], "y": ["why"], "z": ["zee"],
}

# Regex pattern that matches an ALL-CAPS acronym token (2–5 uppercase letters).
_ACRONYM_RE = re.compile(r"\b([A-Z]{2,5})\b")


def _variants_for_acronym(acronym: str) -> list[str]:
    """Generate spoken variants for an ALL-CAPS acronym.

    Produces:
    1. The literal spaced-letter form: "s t t"
    2. All combinations from LETTER_SOUNDS joined with spaces:
       "es te te", "ess te te", "es tee te", ... etc.

    Returns lowercase strings.
    """
    letters = [c.lower() for c in acronym]

    # 1. Literal spaced letters: "s t t"
    spaced_literal = " ".join(letters)

    # 2. Spelled-letter combinations from LETTER_SOUNDS
    sound_options = [LETTER_SOUNDS[l] for l in letters]
    spelled_variants = [" ".join(combo) for combo in product(*sound_options)]

    # Combine, deduplicate, return
    all_variants: list[str] = [spaced_literal]
    for v in spelled_variants:
        if v not in all_variants:
            all_variants.append(v)

    return all_variants


def build_alias_map(index: dict) -> dict[str, str]:
    """Auto-derive an alias map {spoken_variant -> canonical} from the deck.

    For every ALL-CAPS acronym token (2–5 uppercase letters) found in any
    element's ``text``, generate spoken variants that map back to the
    lowercased canonical acronym:
      - literal spaced-letter form: "s t t"
      - spelled-letter combinations via LETTER_SOUNDS: "es te te", "ess tee tee", …

    Also merges hand overrides loaded from data/aliases.json (overrides win).
    Cross-acronym collision: if a generated variant maps to two different
    canonicals, the variant is dropped (ambiguous). File overrides are not
    subject to collision-drop — they are applied last and always win.

    Returns lowercase keys -> lowercase canonical.
    """
    # Collect all acronyms from every element's text field.
    acronyms: set[str] = set()
    for slide in index.get("slides", []):
        for elem in slide.get("elements", []):
            text = elem.get("text") or ""
            for match in _ACRONYM_RE.findall(text):
                acronyms.add(match)

    # Build per-acronym variant sets; track collisions.
    # variant_map: variant_str -> canonical_str
    # collision_set: variants that map to >1 canonical (to be dropped)
    variant_map: dict[str, str] = {}
    collision_set: set[str] = set()

    for acronym in sorted(acronyms):  # sorted for determinism
        canonical = acronym.lower()
        for variant in _variants_for_acronym(acronym):
            if variant in collision_set:
                continue
            if variant in variant_map:
                if variant_map[variant] != canonical:
                    # Collision: two acronyms share this variant → drop it
                    collision_set.add(variant)
                    del variant_map[variant]
                # else: same canonical, no-op
            else:
                variant_map[variant] = canonical

    # Merge file overrides — these always win (no collision-drop applied).
    file_overrides = load_aliases("data/aliases.json")
    variant_map.update(file_overrides)

    return variant_map


def load_aliases(path: str | Path) -> dict[str, str]:
    """Load a flat {variant: canonical} JSON file if it exists.

    Returns an empty dict if the file is absent. All keys and values are
    lowercased.
    """
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as fh:
        raw: dict = json.load(fh)
    return {k.lower(): v.lower() for k, v in raw.items()}


class _CompiledNormalizer:
    """Pre-compiled alias matcher for one alias_map.

    The combined look-around regex depends only on the alias_map's keys, so it
    is built once per map instead of on every frame (the live loop normalizes
    every STT partial). Output is byte-identical to the inline version.
    """

    __slots__ = ("alias_map", "_pattern")

    def __init__(self, alias_map: dict[str, str]) -> None:
        self.alias_map = alias_map  # strong ref keeps id() stable while cached
        # Sort variants by length descending so longer matches take priority.
        sorted_variants = sorted(alias_map.keys(), key=len, reverse=True)
        # Variants may contain spaces (e.g. "s t t"), so \b is unreliable at
        # space boundaries; use look-arounds: not preceded/followed by a word char.
        self._pattern = re.compile(
            "|".join(r"(?<!\w)" + re.escape(v) + r"(?!\w)" for v in sorted_variants),
            re.IGNORECASE,
        )

    def __call__(self, text: str) -> str:
        result = self._pattern.sub(
            lambda m: self.alias_map[m.group(0).lower()], text
        )
        return _collapse_ws(result)


# Cache compiled normalizers by alias_map identity. Each entry holds a strong
# reference to its alias_map, so the id can't be reused while cached. In practice
# one entry per session (the live alias_map is built once and reused).
_NORMALIZER_CACHE: dict[int, _CompiledNormalizer] = {}


def compile_normalizer(alias_map: dict[str, str]) -> _CompiledNormalizer:
    """Return a reusable, pre-compiled normalizer for ``alias_map``."""
    key = id(alias_map)
    norm = _NORMALIZER_CACHE.get(key)
    if norm is None or norm.alias_map is not alias_map:
        norm = _CompiledNormalizer(alias_map)
        _NORMALIZER_CACHE[key] = norm
    return norm


def normalize(text: str, alias_map: dict[str, str]) -> str:
    """Return ``text`` with alias variants replaced by their canonical form.

    Rules:
    - Case-insensitive matching (text is lowercased for comparison, but we
      replace the original span).
    - Word-boundary safe: variants must appear as whole words / token sequences
      (surrounded by non-word chars or start/end of string).
    - Longest-variant-first to avoid partial replacements.
    - Collapses repeated whitespace in the result.
    - Pure function — no I/O. The compiled regex is cached per alias_map.
    """
    if not alias_map:
        return _collapse_ws(text)
    return compile_normalizer(alias_map)(text)


def _collapse_ws(text: str) -> str:
    """Collapse runs of whitespace (except newlines) to a single space."""
    return re.sub(r"[^\S\n]+", " ", text).strip()
