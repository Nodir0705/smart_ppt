"""Stage: group_scope — author-time index enrichment.

Adds group metadata and scoped/expansion trigger phrases to each element
in the index. Deterministic, offline, CPU-only — no neural model.

Usage:
    python3 -m src.group_scope [--input ...] [--out ...] [--expansions ...]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

EXPANSIONS_PATH = "data/expansions.json"

# ---------------------------------------------------------------------------
# chart_noun
# ---------------------------------------------------------------------------

_NOUN_MAP = [
    ("column", "bar"),
    ("bar", "bar"),
    ("pie", "slice"),
    ("doughnut", "slice"),
    ("line", "point"),
    ("scatter", "point"),
    ("xy", "point"),
    ("bubble", "point"),
]


def chart_noun(chart_text: str) -> str:
    """Map a chart container's text to the per-item noun.

    'bar' for COLUMN/BAR, 'slice' for PIE/DOUGHNUT, 'point' for
    LINE/SCATTER/XY/BUBBLE, else 'item'.  Case-insensitive substring match.
    """
    lower = chart_text.lower()
    for keyword, noun in _NOUN_MAP:
        if keyword in lower:
            return noun
    return "item"


# ---------------------------------------------------------------------------
# group_label_for_chart
# ---------------------------------------------------------------------------

_TITLE_KEYWORDS = [
    (("latency", "time"), "latency"),
    (("effort", "build"), "effort"),
    (("accuracy", "threshold"), "accuracy"),
]


def group_label_for_chart(chart_elem: dict, slide_title: str | None) -> str:
    """Return a short human label for a chart group, derived deterministically.

    Priority:
      1. Use chart's own text if it is a non-generic title (not a bare
         chart-type string like "COLUMN_CLUSTERED (51)").
      2. Infer from slide title keywords.
      3. Fall back to ``<noun> chart``.
    Returns a lowercase label.
    """
    # Heuristic: if the chart text doesn't look like a bare chart-type token
    # (i.e. it doesn't begin with an all-caps word followed by optional
    # underscore parts and a parenthesised number), treat it as a real title.
    text = (chart_elem.get("text") or "").strip()
    import re
    _CHART_TYPE_RE = re.compile(r'^[A-Z][A-Z0-9_]+\s*\(\d+\)\s*$')
    if text and not _CHART_TYPE_RE.match(text):
        return text.lower()

    # Infer from slide title
    if slide_title:
        lower_title = slide_title.lower()
        for keywords, label in _TITLE_KEYWORDS:
            if any(k in lower_title for k in keywords):
                return label

    # Fallback
    noun = chart_noun(text)
    return f"{noun} chart"


# ---------------------------------------------------------------------------
# load_expansions
# ---------------------------------------------------------------------------

def load_expansions(path: str = EXPANSIONS_PATH) -> dict:
    """Load {ACRONYM: expansion} JSON if present, else {}.  Keys uppercased."""
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        return {k.upper(): v for k, v in raw.items()}
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# scope_index
# ---------------------------------------------------------------------------

def _add_phrase(phrases: list, phrase: str) -> bool:
    """Append phrase if not already present (case-sensitive de-dup). Returns True if added."""
    if phrase and phrase not in phrases:
        phrases.append(phrase)
        return True
    return False


# ---------------------------------------------------------------------------
# stat spoken-form phrases  (e.g. "< 1 ms" -> "less than one millisecond")
# ---------------------------------------------------------------------------

_SYMBOL_WORDS = {
    "<": ["less than", "under", "below"], ">": ["more than", "over", "above"],
    "≤": ["at most", "up to"], "≥": ["at least"], "~": ["about", "around"],
}
_UNIT_WORDS = {
    "ms": ["millisecond", "milliseconds"], "s": ["second", "seconds"],
    "fps": ["frames per second"], "%": ["percent"], "x": ["times"],
    "kb": ["kilobyte", "kilobytes"], "mb": ["megabyte", "megabytes"],
    "gb": ["gigabyte", "gigabytes"],
}
_NUM_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
}


def stat_phrases(text: str) -> list[str]:
    """Spoken-form trigger phrases for a "[symbol] number unit" stat callout.

    "< 1 ms" -> ["one millisecond", "less than one millisecond",
                 "less than a millisecond", "sub millisecond", ...].
    Returns [] when the text isn't a recognisable stat.
    """
    import re
    m = re.match(r"^\s*([<>≤≥~]?)\s*(\d+(?:\.\d+)?)\s*([a-zA-Z%]+)\s*$", text.strip())
    if not m:
        return []
    sym, num, unit = m.group(1), m.group(2), m.group(3).lower()
    num_word = _NUM_WORDS.get(num, num)
    units = _UNIT_WORDS.get(unit, [unit])
    out: list[str] = []
    for u in units:
        out.append(f"{num_word} {u}")
        for sw in _SYMBOL_WORDS.get(sym, []):
            out.append(f"{sw} {num_word} {u}")
            out.append(f"{sw} a {u}")
    if sym == "<" and unit == "ms":
        out += ["sub millisecond", "sub-millisecond"]
    return out


def scope_index(index: dict, expansions: dict | None = None) -> dict:
    """Return index (mutated + returned) with group metadata and extra trigger phrases.

    For every element:
      - 'group_id': parent_id if the element has one (chart_category), else its own id.
      - 'group_label': for chart_category, the group_label_for_chart of its parent chart
        (looked up within the same slide); else None.

    Added trigger phrases (append, dedupe, keep existing):
      - chart_category: scoped phrases using label + noun + group_label:
          'the {label} {noun}', '{label} {group_label}',
          'the {label} {noun} in the {group_label} chart'
      - ANY element with acronym(s) from expansions in its text:
          expansion and 'expansion + remaining words' as phrases.

    Idempotent: re-running adds nothing new.
    """
    if expansions is None:
        expansions = {}

    for slide in index.get("slides", []):
        elements = slide.get("elements", [])

        # Build lookup: id -> element (for parent resolution)
        elem_by_id: dict[str, dict] = {e["id"]: e for e in elements}

        # Identify slide title text
        slide_title: str | None = None
        for elem in elements:
            if elem.get("type") == "title":
                slide_title = elem.get("text") or None
                break

        # A bullet list is one group, so the temporal tracker's same-group
        # stickiness damps backward flicker between bullets (Extract -> Parse)
        # while still allowing the sustained forward progression.
        bullet_ids = [e["id"] for e in elements if e.get("type") == "bullet"]
        bullet_group = bullet_ids[0] if len(bullet_ids) >= 2 else None

        for elem in elements:
            eid = elem["id"]
            parent_id = elem.get("parent_id")
            phrases: list = elem.setdefault("trigger_phrases", [])

            # ------ group_id / group_label ------
            if parent_id:
                elem["group_id"] = parent_id
                parent_elem = elem_by_id.get(parent_id)
                if parent_elem is not None:
                    elem["group_label"] = group_label_for_chart(parent_elem, slide_title)
                else:
                    elem["group_label"] = None
            elif bullet_group is not None and elem.get("type") == "bullet":
                elem["group_id"] = bullet_group
                elem["group_label"] = "bullets"
            else:
                elem["group_id"] = eid
                elem["group_label"] = None

            # ------ spoken-form phrases for stat callouts ------
            if elem.get("type") == "stat":
                for ph in stat_phrases(elem.get("text") or ""):
                    _add_phrase(phrases, ph)

            # ------ scoped phrases for chart_category ------
            if elem.get("type") == "chart_category":
                label = (elem.get("text") or "").strip().lower()
                parent_elem = elem_by_id.get(parent_id) if parent_id else None
                noun = chart_noun(parent_elem.get("text", "") if parent_elem else "") if parent_elem else "item"
                group_label = elem.get("group_label") or ""

                if label and noun:
                    _add_phrase(phrases, f"the {label} {noun}")
                if label and group_label:
                    _add_phrase(phrases, f"{label} {group_label}")
                if label and noun and group_label:
                    _add_phrase(phrases, f"the {label} {noun} in the {group_label} chart")

            # ------ acronym expansion ------
            text = elem.get("text") or ""
            # Split text into tokens (preserve original casing for comparison)
            import re
            tokens = re.split(r'(\s+)', text)
            word_tokens = [t for t in tokens if t.strip()]

            for i, token in enumerate(word_tokens):
                upper_token = token.upper()
                if upper_token in expansions:
                    expansion = expansions[upper_token]
                    # Add bare expansion
                    _add_phrase(phrases, expansion.lower())
                    # Add expansion + remaining words
                    remaining = " ".join(word_tokens[i + 1:]).strip().lower()
                    if remaining:
                        _add_phrase(phrases, f"{expansion.lower()} {remaining}")

    return index


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Scope index with group metadata and trigger phrases.")
    parser.add_argument("--input", default="out/index.refined.json",
                        help="Input index JSON (default: out/index.refined.json)")
    parser.add_argument("--out", default="out/index.scoped.json",
                        help="Output index JSON (default: out/index.scoped.json)")
    parser.add_argument("--expansions", default=EXPANSIONS_PATH,
                        help=f"Expansions JSON (default: {EXPANSIONS_PATH})")
    args = parser.parse_args(argv)

    with open(args.input, encoding="utf-8") as fh:
        index = json.load(fh)

    expansions = load_expansions(args.expansions)

    # Count existing phrases before
    before = sum(
        len(e.get("trigger_phrases", []))
        for s in index.get("slides", [])
        for e in s.get("elements", [])
    )

    scope_index(index, expansions=expansions)

    after = sum(
        len(e.get("trigger_phrases", []))
        for s in index.get("slides", [])
        for e in s.get("elements", [])
    )
    n_elements = sum(len(s.get("elements", [])) for s in index.get("slides", []))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)

    print(f"scoped {n_elements} elements, +{after - before} trigger phrases -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
