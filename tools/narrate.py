"""Auto-narration for the multi-deck bench: deck text -> spoken script + expected order.

Input must be the XML-only truth parse (`parse_deck(deck, ocr=False)`), never the
index under test — otherwise an OCR/VLM hallucination would be narrated and then
scored as a hit. Pure and deterministic.

    script   = [{"slide": n, "segments": ["The Title", "First bullet", ...]}]
    expected = [{"slide": n, "expected": [["s1_e0"], ["s1_e2"], ...]}]
"""
from __future__ import annotations

NARRATED_TYPES = {"title", "bullet", "body", "caption", "stat", "diagram_label",
                  "table_cell", "chart_category", "image"}
MAX_WORDS = 15
BAND_IN = 0.3  # elements whose tops differ by <= this many inches share a row


def _norm(text: str | None) -> str:
    return " ".join((text or "").split())


def _narratable(e: dict) -> bool:
    return (e.get("type") in NARRATED_TYPES and not e.get("needs_vision")
            and bool(_norm(e.get("text"))))


def _phrase(e: dict) -> str:
    if e["type"] == "chart_category":
        raw = _norm(e.get("text"))
        series = e.get("series")
        # multi-series labels are "Category — Series"; speak "Series Category"
        cat = raw.split(" — ", 1)[0] if series else raw
        who = f"{series} {cat}" if series else cat
        v = e.get("value")
        if v is None:
            return who
        # speakable digits: 12.0 -> "12", 1234567.891 -> "1234567.89" (never "1.2e+06")
        return f"{who} is {int(v) if float(v).is_integer() else round(v, 2)}"
    return " ".join(_norm(e.get("text")).split()[:MAX_WORDS])


def _reading_order(elements: list[dict]) -> list[dict]:
    """Title first; then rows (banded by top y) top-to-bottom, each left-to-right.

    A chart_category is placed at its parent chart's position; sorts are stable,
    so categories of one chart keep their index (category) order."""
    by_id = {e["id"]: e for e in elements}

    def anchor(e: dict) -> dict:
        parent = by_id.get(e.get("parent_id")) if e["type"] == "chart_category" else None
        return (parent or e)["bbox"]

    items = [e for e in elements if _narratable(e)]
    ordered = [e for e in items if e["type"] == "title"]
    rest = sorted((e for e in items if e["type"] != "title"), key=lambda e: anchor(e)["y"])
    band: list[dict] = []
    top = 0.0
    for e in rest:
        y = anchor(e)["y"]
        if band and y - top > BAND_IN:
            ordered += sorted(band, key=lambda b: anchor(b)["x"])
            band = []
        if not band:
            top = y
        band.append(e)
    ordered += sorted(band, key=lambda b: anchor(b)["x"])
    return ordered


def narrate(index: dict) -> tuple[list[dict], list[dict]]:
    """Return (script_slides, expected_slides); slides with nothing to say are omitted.

    Elements whose spoken phrase is identical are narrated once, and that step
    accepts any of their ids (three identical "Live" cells are one step)."""
    script: list[dict] = []
    expected: list[dict] = []
    for slide in index.get("slides", []):
        segments: list[str] = []
        steps: list[list[str]] = []
        seen: dict[str, int] = {}
        for e in _reading_order(slide.get("elements", [])):
            phrase = _phrase(e)
            key = phrase.lower()
            if key in seen:
                steps[seen[key]].append(e["id"])
                continue
            seen[key] = len(steps)
            segments.append(phrase)
            steps.append([e["id"]])
        if segments:
            sno = slide["slide_number"]
            script.append({"slide": sno, "segments": segments})
            expected.append({"slide": sno, "expected": steps})
    return script, expected
