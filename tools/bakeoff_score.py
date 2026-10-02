"""bakeoff_score.py — pure scorer for the OCR bake-off.

Scores every `out/bakeoff/raw/<engine>.json` against the deck truth JSONs.
No OCR engine imports here (keeps this importable/testable without
tesseract/paddleocr installed) — pure functions over plain dicts.

    python3 -m tools.bakeoff_score                 # default paths, writes report
    python3 -m tools.bakeoff_score --raw-dir DIR --truth-dir DIR --out FILE

Raw format (one file per engine): see tools/ocr_bakeoff.py docstring.
Reserved `_`-prefixed keys inside an engine file (not decks):
  _unit: "line"|"word"
  _seconds_per_slide: float
  _deterministic: bool|null   (null = not tested)
  _icons: {"<label>": [{"text":..,"conf":..}, ...]}  # icon/negative crops
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = ROOT / "out" / "bakeoff" / "raw"
DEFAULT_TRUTH_DIR = ROOT / "out" / "image_decks"
DEFAULT_OUT = ROOT / "out" / "bakeoff" / "ocr_report.md"

ICON_MAX_AREA_SQIN = 0.5
IOU_THRESHOLD = 0.3
ELEMENT_WORD_COVERAGE = 0.8

_PUNCT_EDGE = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)


def normalize_words(text: str) -> list[str]:
    """lowercase, strip edge punctuation per word, drop empties."""
    out = []
    for w in (text or "").split():
        w = _PUNCT_EDGE.sub("", w).lower()
        if w:
            out.append(w)
    return out


# ---------- geometry ----------

def _bbox_xyxy(b: dict | list) -> tuple[float, float, float, float]:
    if isinstance(b, dict):
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
    else:
        x, y, w, h = b
    return x, y, x + w, y + h


def _union_xyxy(boxes: list[tuple[float, float, float, float]]):
    xs0 = [b[0] for b in boxes]
    ys0 = [b[1] for b in boxes]
    xs1 = [b[2] for b in boxes]
    ys1 = [b[3] for b in boxes]
    return min(xs0), min(ys0), max(xs1), max(ys1)


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _centre_inside(inner, outer) -> bool:
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _intersects(a, b) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


# ---------- ground truth ----------

EXCLUDED_GT_TYPES = {"chart", "image"}


def gt_elements_for_slide(slide: dict) -> list[dict]:
    """Truth elements with non-empty text, excluding chart/image types."""
    out = []
    for el in slide.get("elements", []):
        if el.get("type") in EXCLUDED_GT_TYPES:
            continue
        text = (el.get("text") or "").strip()
        if not text:
            continue
        out.append(el)
    return out


# ---------- per-slide scoring ----------

def score_slide(gt_elements: list[dict], ocr_entries: list[dict]) -> dict:
    """Score one slide's OCR entries against its GT elements.

    Returns word-level multisets (for recall/precision aggregation) and
    element-hit count, all unaggregated so callers can pool across slides.
    """
    gt_words: list[str] = []
    for el in gt_elements:
        gt_words.extend(normalize_words(el.get("text", "")))

    ocr_words: list[str] = []
    ocr_boxes = []
    for e in ocr_entries:
        ws = normalize_words(e.get("text", ""))
        ocr_words.extend(ws)
        if e.get("bbox") is not None:  # some engines (VLM captions) have no bbox
            ocr_boxes.append((_bbox_xyxy(e["bbox"]), ws))

    # multiset overlap for recall/precision
    from collections import Counter
    gt_counter = Counter(gt_words)
    ocr_counter = Counter(ocr_words)
    overlap = sum((gt_counter & ocr_counter).values())

    # element recall
    element_hits = 0
    for el in gt_elements:
        el_box = _bbox_xyxy(el["bbox"])
        el_words = normalize_words(el.get("text", ""))
        if not el_words:
            continue
        candidates = [(box, ws) for box, ws in ocr_boxes if _intersects(box, el_box)]
        if not candidates:
            continue
        gate = any(_centre_inside(box, el_box) or _centre_inside(el_box, box)
                   for box, _ in candidates)
        if not gate:
            union_box = _union_xyxy([box for box, _ in candidates])
            gate = _iou(el_box, union_box) >= IOU_THRESHOLD
        if not gate:
            continue
        cand_words: list[str] = []
        for _, ws in candidates:
            cand_words.extend(ws)
        cand_counter = Counter(cand_words)
        covered = sum((Counter(el_words) & cand_counter).values())
        if covered >= ELEMENT_WORD_COVERAGE * len(el_words):
            element_hits += 1

    return {
        "gt_word_count": len(gt_words),
        "ocr_word_count": len(ocr_words),
        "word_overlap": overlap,
        "gt_element_count": sum(1 for el in gt_elements if normalize_words(el.get("text", ""))),
        "element_hits": element_hits,
    }


def score_deck(truth: dict, engine_deck_raw: dict) -> dict:
    """Pool per-slide scores across a whole deck."""
    totals = {"gt_word_count": 0, "ocr_word_count": 0, "word_overlap": 0,
              "gt_element_count": 0, "element_hits": 0, "slides": 0}
    for slide in truth.get("slides", []):
        n = slide.get("slide_number")
        gt_els = gt_elements_for_slide(slide)
        ocr_entries = engine_deck_raw.get(str(n), [])
        s = score_slide(gt_els, ocr_entries)
        for k in ("gt_word_count", "ocr_word_count", "word_overlap",
                  "gt_element_count", "element_hits"):
            totals[k] += s[k]
        totals["slides"] += 1
    return totals


def summarize(totals: dict) -> dict:
    gt_w = totals["gt_word_count"]
    ocr_w = totals["ocr_word_count"]
    gt_e = totals["gt_element_count"]
    return {
        "word_recall": (totals["word_overlap"] / gt_w) if gt_w else None,
        "word_precision": (totals["word_overlap"] / ocr_w) if ocr_w else None,
        "element_recall": (totals["element_hits"] / gt_e) if gt_e else None,
        "slides": totals["slides"],
    }


# ---------- icon / negative false-positive rate ----------

def icon_false_positive_rate(icons: dict) -> dict:
    """Fraction of icon/negative crops where the engine (after its own
    filtering, already applied when `_icons` was built) emitted ANY token."""
    if not icons:
        return {"n": 0, "fp": 0, "rate": None}
    n = len(icons)
    fp = sum(1 for entries in icons.values() if entries)
    return {"n": n, "fp": fp, "rate": fp / n if n else None}


# ---------- IO / report ----------

def load_truths(truth_dir: Path, deck_names: list[str]) -> dict[str, dict]:
    out = {}
    for name in deck_names:
        p = truth_dir / f"{name}__truth.json"
        out[name] = json.loads(p.read_text())
    return out


def discover_engines(raw_dir: Path) -> list[str]:
    return sorted(p.stem for p in raw_dir.glob("*.json") if not p.stem.startswith("_"))


def score_all(raw_dir: Path, truth_dir: Path) -> dict:
    truth_dir = Path(truth_dir)
    deck_names = sorted(p.stem.removesuffix("__truth") for p in truth_dir.glob("*__truth.json"))
    truths = load_truths(truth_dir, deck_names)

    report: dict = {"engines": {}}
    for engine in discover_engines(raw_dir):
        raw = json.loads((raw_dir / f"{engine}.json").read_text())
        meta = {k: v for k, v in raw.items() if k.startswith("_")}
        per_deck = {}
        grand = {"gt_word_count": 0, "ocr_word_count": 0, "word_overlap": 0,
                 "gt_element_count": 0, "element_hits": 0, "slides": 0}
        for name in deck_names:
            if name not in raw:
                continue
            totals = score_deck(truths[name], raw[name])
            per_deck[name] = {**summarize(totals), **totals}
            for k in grand:
                grand[k] += totals[k]
        report["engines"][engine] = {
            "per_deck": per_deck,
            "total": {**summarize(grand), **grand},
            "unit": meta.get("_unit"),
            "seconds_per_slide": meta.get("_seconds_per_slide"),
            "deterministic": meta.get("_deterministic"),
            "icon_fp": icon_false_positive_rate(meta.get("_icons", {})),
        }
    return report


def _fmt(x, pct=False, nd=3):
    if x is None:
        return "n/a"
    if pct:
        return f"{x * 100:.1f}%"
    return f"{x:.{nd}f}"


def render_markdown(report: dict) -> str:
    lines = ["# OCR bake-off report", ""]
    lines.append("| engine | deck | word recall | word precision | element recall | s/slide | icon FP | deterministic |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for engine, data in sorted(report["engines"].items()):
        for deck, d in sorted(data["per_deck"].items()):
            lines.append(
                f"| {engine} | {deck} | {_fmt(d['word_recall'], True)} | "
                f"{_fmt(d['word_precision'], True)} | {_fmt(d['element_recall'], True)} | "
                f"{_fmt(data['seconds_per_slide'])} | {_fmt(data['icon_fp']['rate'], True)} | "
                f"{data['deterministic']} |"
            )
        t = data["total"]
        lines.append(
            f"| **{engine}** | **TOTAL** | **{_fmt(t['word_recall'], True)}** | "
            f"**{_fmt(t['word_precision'], True)}** | **{_fmt(t['element_recall'], True)}** | "
            f"{_fmt(data['seconds_per_slide'])} | {_fmt(data['icon_fp']['rate'], True)} ({data['icon_fp']['fp']}/{data['icon_fp']['n']}) | "
            f"{data['deterministic']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--truth-dir", type=Path, default=DEFAULT_TRUTH_DIR)
    ap.add_argument("--out", type=Path, default=None, help="write report JSON here (markdown table also printed)")
    args = ap.parse_args()
    report = score_all(args.raw_dir, args.truth_dir)
    md = render_markdown(report)
    print(md)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
