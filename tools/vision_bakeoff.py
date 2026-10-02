"""Static vision-model bake-off for what smartppt can't read from the pptx XML.

Three questions, scored against the parse_deck truth exported by the image-deck
generator (``out/image_decks/<deck>__truth.json`` + rendered slide PNGs):

  A. read text from pixels (OCR competitor)       -> raw/<engine>.json
  B. describe image elements so they become pointable (the needs_vision gap)
  C. layout regions from pixels vs. the pptx element boxes

Models run in ``tools/vision_runner.py`` under ``~/.venvs/smartppt-vision`` (a
--system-site-packages venv that only adds timm), one subprocess per model so
VRAM peaks are per-model. This module is pure (no torch import): geometry,
classification, metrics, orchestration and the markdown report.

    python3 tools/vision_bakeoff.py run            # every engine, twice (determinism)
    python3 tools/vision_bakeoff.py run --only florence2-base
    python3 tools/vision_bakeoff.py tables         # markdown tables for out/bakeoff/vision_report.md
    python3 tools/vision_bakeoff.py spot           # side-by-side captions for eyeballing
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.get("SMARTPPT_ROOT", "/home/netvision/smartppt"))
TRUTH_DIR = ROOT / "out" / "image_decks"
OUT = ROOT / "out" / "bakeoff"
VENV_PY = Path.home() / ".venvs" / "smartppt-vision" / "bin" / "python"
RUNNER = Path(__file__).with_name("vision_runner.py")

NON_TEXT_TYPES = {"chart", "image"}
ICON_MAX_AREA = 0.5      # in^2
BACKGROUND_FRAC = 0.9    # w AND h >= 90% of the slide -> background
NEAR_INCH = 1.0          # caption proxy: text element within this gap of the image

STOPWORDS = set("""
a an the and or of to in on at for with by from as is are was were be been it its this that
these those there their they them he she his her we our you your i my me not no but so if
into onto over under up down out off than then also very can will just has have had do does
""".split())
# caption filler: words captioners emit about the medium, not the content
FILLER = set("""
image picture photo photograph illustration drawing graphic icon logo screenshot close closeup
view background white black showing shows displayed display depicting depicts some small large
green blue red yellow orange purple pink gray grey dark light colored colorful
""".split())


# ---------------------------------------------------------------- geometry

def px_per_inch(png_w: int, png_h: int, slide_size: dict) -> tuple[float, float]:
    """(sx, sy) pixels per inch. slide_size must be in inches."""
    assert slide_size.get("units", "inches") == "inches", slide_size
    sx, sy = png_w / slide_size["w"], png_h / slide_size["h"]
    assert abs(sx - sy) / sy < 0.01, f"non-uniform scale {sx} vs {sy}"
    return sx, sy


def inch_to_px_box(bbox: dict, scale: tuple[float, float], png_w: int, png_h: int):
    """Truth bbox (inches, x/y/w/h) -> clipped integer pixel crop (l, t, r, b); None if empty."""
    sx, sy = scale
    l = max(0, math.floor(bbox["x"] * sx))
    t = max(0, math.floor(bbox["y"] * sy))
    r = min(png_w, math.ceil((bbox["x"] + bbox["w"]) * sx))
    b = min(png_h, math.ceil((bbox["y"] + bbox["h"]) * sy))
    return (l, t, r, b) if r > l and b > t else None


def px_to_inch_box(x0: float, y0: float, x1: float, y1: float, scale) -> list[float]:
    sx, sy = scale
    return [x0 / sx, y0 / sy, (x1 - x0) / sx, (y1 - y0) / sy]


def quad_to_xyxy(quad: list[float]) -> tuple[float, float, float, float]:
    xs, ys = quad[0::2], quad[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def _xyxy(b) -> tuple[float, float, float, float]:
    if isinstance(b, dict):
        return b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    return b[0], b[1], b[0] + b[2], b[1] + b[3]


def center(b) -> tuple[float, float]:
    x0, y0, x1, y1 = _xyxy(b)
    return (x0 + x1) / 2, (y0 + y1) / 2


def rect_gap(a, b) -> float:
    """Euclidean gap between two rectangles (0 when they touch/overlap)."""
    ax0, ay0, ax1, ay1 = _xyxy(a)
    bx0, by0, bx1, by1 = _xyxy(b)
    dx = max(0.0, bx0 - ax1, ax0 - bx1)
    dy = max(0.0, by0 - ay1, ay0 - by1)
    return math.hypot(dx, dy)


def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = _xyxy(a)
    bx0, by0, bx1, by1 = _xyxy(b)
    iw, ih = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter)


def contains_point(b, pt) -> bool:
    x0, y0, x1, y1 = _xyxy(b)
    return x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1


# ---------------------------------------------------------- classification

def is_full_slide(bbox: dict, slide_w: float, slide_h: float) -> bool:
    return bbox["w"] >= BACKGROUND_FRAC * slide_w and bbox["h"] >= BACKGROUND_FRAC * slide_h


def classify_image(bbox: dict, slide_w: float, slide_h: float) -> str:
    """'background' (near-full-slide), 'icon' (< 0.5 in^2) or 'content'. Uses the raw bbox."""
    if is_full_slide(bbox, slide_w, slide_h):
        return "background"
    return "icon" if bbox["w"] * bbox["h"] < ICON_MAX_AREA else "content"


def gt_text_elements(slide: dict) -> list[dict]:
    """OCR ground truth: elements carrying text, excluding chart/image types."""
    return [e for e in slide["elements"] if e.get("text") and e["type"] not in NON_TEXT_TYPES]


def anchor_text_elements(slide: dict, slide_w: float, slide_h: float) -> list[dict]:
    """Text elements usable as spatial anchors / layout GT: drops full-slide text boxes."""
    return [e for e in gt_text_elements(slide) if not is_full_slide(e["bbox"], slide_w, slide_h)]


# ------------------------------------------------------------------- text

_EDGE = re.compile(r"^[\W_]+|[\W_]+$")


def words(text: str) -> list[str]:
    """Lowercase, whitespace split, edge punctuation stripped, empties dropped."""
    out = []
    for tok in (text or "").lower().split():
        tok = _EDGE.sub("", tok)
        if tok:
            out.append(tok)
    return out


def content_words(text: str) -> set[str]:
    return {w for w in words(text)
            if len(w) >= 3 and w not in STOPWORDS and w not in FILLER and not w.isdigit()}


def word_overlap(pred: list[str], gt: list[str]) -> int:
    """Multiset intersection size."""
    return sum((Counter(pred) & Counter(gt)).values())


# --------------------------------------------------------------- metrics

def nearest_text_id(img_bbox: dict, texts: list[dict]) -> str | None:
    """Text element whose bbox centre is closest to the image's centre."""
    if not texts:
        return None
    cx, cy = center(img_bbox)
    return min(texts, key=lambda e: (math.hypot(center(e["bbox"])[0] - cx,
                                                center(e["bbox"])[1] - cy), e["id"]))["id"]


def rank_of(target: str, scores: dict[str, float]) -> int:
    """1-based rank of target under descending score (ties broken by id, deterministic)."""
    order = sorted(scores, key=lambda k: (-scores[k], k))
    return order.index(target) + 1


def caption_hit(caption: str, img_bbox: dict, texts: list[dict], near: float = NEAR_INCH):
    """(has_neighbour, hit): any text element within `near` inches; caption shares a content word."""
    neigh = [e for e in texts if rect_gap(img_bbox, e["bbox"]) <= near]
    cw = content_words(caption)
    return bool(neigh), any(cw & content_words(e["text"]) for e in neigh)


def has_overlaid_text(img_bbox: dict, texts: list[dict]) -> bool:
    """A truth text element is centred inside the image: the crop shows text the XML already has."""
    return any(contains_point(img_bbox, center(e["bbox"])) for e in texts)


def icon_risk(caption: str, slide_texts: list[dict]) -> bool:
    """Caption would mint a new pointable: >=1 content word absent from all slide text."""
    slide_words = set().union(*(content_words(e["text"]) for e in slide_texts)) if slide_texts else set()
    return bool(content_words(caption) - slide_words)


def layout_scores(gt_boxes: list, all_truth_boxes: list, pred_boxes: list, thr: float):
    """(gt matched by any pred with IoU>=thr, preds matching no truth element at IoU>=thr)."""
    matched = sum(any(iou(g, p) >= thr for p in pred_boxes) for g in gt_boxes)
    orphan = sum(not any(iou(p, t) >= thr for t in all_truth_boxes) for p in pred_boxes)
    return matched, orphan


# ------------------------------------------------------------- deck I/O

def load_decks(truth_dir: Path = TRUTH_DIR) -> dict[str, dict]:
    """{deck_stem: truth} for every <deck>__truth.json."""
    return {p.name[: -len("__truth.json")]: json.loads(p.read_text())
            for p in sorted(truth_dir.glob("*__truth.json"))}


def slide_png(truth: dict, slide_no) -> Path:
    return ROOT / truth["slide_png"][str(slide_no)]


def image_jobs(decks: dict[str, dict]) -> list[dict]:
    """Every non-background image element with its pixel crop and set (content|icon)."""
    from PIL import Image  # local: keep module import light
    jobs = []
    for deck, t in decks.items():
        W, H = t["slide_size"]["w"], t["slide_size"]["h"]
        for s in t["slides"]:
            png = slide_png(t, s["slide_number"])
            pw, ph = Image.open(png).size
            scale = px_per_inch(pw, ph, t["slide_size"])
            for e in s["elements"]:
                if e["type"] != "image":
                    continue
                kind = classify_image(e["bbox"], W, H)
                crop = inch_to_px_box(e["bbox"], scale, pw, ph)
                if kind == "background" or crop is None:
                    continue
                jobs.append({"deck": deck, "slide": str(s["slide_number"]), "id": e["id"],
                             "set": kind, "png": str(png), "crop": crop,
                             "candidates": [(c["id"], c["text"]) for c in
                                            anchor_text_elements(s, W, H)] if kind == "content" else []})
    return jobs


# ----------------------------------------------------------- orchestration

ENGINES = {
    # name: (runner model key, tasks)
    "florence2-base": ("florence2-base", ["ocr", "caption", "layout"]),
    "florence2-large": ("florence2-large", ["ocr", "caption", "layout"]),
    "smolvlm-256m": ("smolvlm-256m", ["ocr", "caption"]),
    "smolvlm-500m": ("smolvlm-500m", ["ocr", "caption"]),
    "smolvlm2-2.2b": ("smolvlm2-2.2b", ["ocr", "caption"]),
    # prompt-sensitivity check, not in the spec: does a stricter prompt stop it summarising?
    "smolvlm2-2.2b-transcribe": ("smolvlm2-2.2b", ["ocr"], ["--suffix=-transcribe", "--prompt",
                                 "Transcribe every piece of text visible in this slide, line by line, "
                                 "exactly as written. Output only the text."]),
    "vit-base": ("vit-base", ["caption"]),
    "clip-b32": ("clip-b32", ["clip"]),
    "clip-b32-laion": ("clip-b32-laion", ["clip"]),
    "yolov8-doclaynet": ("yolov8-doclaynet", ["layout"]),
}


def run(only: list[str] | None, passes: int, limit: int | None) -> None:
    for name, (key, tasks, *extra) in ENGINES.items():
        if only and name not in only:
            continue
        for task in tasks:
            for p in range(1, passes + 1):
                cmd = [str(VENV_PY), str(RUNNER), "--model", key, "--task", task, "--pass", str(p),
                       *(extra[0] if extra else [])]
                if limit and p > 1:
                    cmd += ["--limit", str(limit)]
                print("+", " ".join(cmd), flush=True)
                r = subprocess.run(cmd)
                if r.returncode:
                    print(f"!! {name}/{task}/pass{p} failed rc={r.returncode}", flush=True)


# ------------------------------------------------------------- scoring

def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _strip_meta(d):
    return {k: v for k, v in d.items() if not k.startswith("_")} if isinstance(d, dict) else d


def determinism(p1: Path, p2: Path) -> str:
    """Compare pass-2 outputs (possibly a subset) with the same keys from pass 1."""
    a, b = _load(p1), _load(p2)
    if a is None or b is None:
        return "not run"
    a, b = _strip_meta(a), _strip_meta(b)
    keys = same = 0
    for deck, items in b.items():
        for k, v in items.items():
            keys += 1
            same += a.get(deck, {}).get(k) == v
    return "yes" if keys and same == keys else f"no ({same}/{keys} identical)"


def score_ocr(raw: dict, decks: dict) -> dict:
    gt_n = pred_n = hit = 0
    pred_n_x = hit_x = 0   # excluding predicted lines centred inside a non-background image
    for deck, t in decks.items():
        W, H = t["slide_size"]["w"], t["slide_size"]["h"]
        for s in t["slides"]:
            gt = [w for e in gt_text_elements(s) for w in words(e["text"])]
            lines = raw.get(deck, {}).get(str(s["slide_number"]), [])
            pred = [w for ln in lines for w in words(ln["text"])]
            gt_n += len(gt)
            pred_n += len(pred)
            hit += word_overlap(pred, gt)
            imgs = [e["bbox"] for e in s["elements"]
                    if e["type"] == "image" and not is_full_slide(e["bbox"], W, H)]
            kept = [w for ln in lines
                    if not (ln.get("bbox") and any(contains_point(i, center(ln["bbox"])) for i in imgs))
                    for w in words(ln["text"])]
            pred_n_x += len(kept)
            hit_x += word_overlap(kept, gt)
    return {"recall": hit / gt_n if gt_n else 0.0, "precision": hit / pred_n if pred_n else 0.0,
            "precision_ex_img": hit_x / pred_n_x if pred_n_x else 0.0,
            "gt_words": gt_n, "pred_words": pred_n}


def score_captions(raw: dict, decks: dict) -> dict:
    content = icons = neigh = hits = risk = conf_icons = pics = pic_hits = 0
    lens = []
    for deck, t in decks.items():
        W, H = t["slide_size"]["w"], t["slide_size"]["h"]
        by_slide = {str(s["slide_number"]): s for s in t["slides"]}
        for eid, rec in raw.get(deck, {}).items():
            s = by_slide[rec["slide"]]
            el = next(e for e in s["elements"] if e["id"] == eid)
            texts = anchor_text_elements(s, W, H)
            lens.append(len(words(rec["caption"])))
            if rec["set"] == "content":
                content += 1
                n, h = caption_hit(rec["caption"], el["bbox"], texts)
                neigh += n
                hits += h
                if not has_overlaid_text(el["bbox"], texts):
                    pics += 1
                    pic_hits += h
            else:
                icons += 1
                risk += icon_risk(rec["caption"], texts)
                conf_icons += (rec.get("conf") or 0) >= 0.5
    return {"content": content, "with_neighbour": neigh, "hits": hits,
            "hit_rate": hits / content if content else 0.0,
            "hit_rate_reachable": hits / neigh if neigh else 0.0,
            "pictures": pics, "picture_hits": pic_hits,
            "icons": icons, "icon_risk": risk, "icon_risk_rate": risk / icons if icons else 0.0,
            "icon_conf50": conf_icons, "mean_words": sum(lens) / len(lens) if lens else 0.0}


def score_clip(raw: dict, decks: dict) -> dict:
    n = top1 = top3 = pics = pic1 = pic3 = 0
    chance1 = chance3 = 0.0
    for deck, t in decks.items():
        W, H = t["slide_size"]["w"], t["slide_size"]["h"]
        by_slide = {str(s["slide_number"]): s for s in t["slides"]}
        for eid, rec in raw.get(deck, {}).items():
            s = by_slide[rec["slide"]]
            el = next(e for e in s["elements"] if e["id"] == eid)
            texts = anchor_text_elements(s, W, H)
            target = nearest_text_id(el["bbox"], texts)
            if target is None or len(texts) < 2:
                continue
            r = rank_of(target, rec["scores"])
            n += 1
            top1 += r == 1
            top3 += r <= 3
            if not has_overlaid_text(el["bbox"], texts):
                pics += 1
                pic1 += r == 1
                pic3 += r <= 3
            chance1 += 1 / len(texts)
            chance3 += min(3, len(texts)) / len(texts)
    return {"images": n, "top1": top1 / n if n else 0.0, "top3": top3 / n if n else 0.0,
            "chance1": chance1 / n if n else 0.0, "chance3": chance3 / n if n else 0.0,
            "pictures": pics, "pic_top1": pic1, "pic_top3": pic3}


def score_layout(raw: dict, decks: dict) -> dict:
    gt_n = pred_n = m5 = m3 = orph5 = 0
    for deck, t in decks.items():
        W, H = t["slide_size"]["w"], t["slide_size"]["h"]
        for s in t["slides"]:
            gt = [e["bbox"] for e in anchor_text_elements(s, W, H)]
            truth = [e["bbox"] for e in s["elements"] if not is_full_slide(e["bbox"], W, H)]
            pred = [r["bbox"] for r in raw.get(deck, {}).get(str(s["slide_number"]), [])]
            a, o = layout_scores(gt, truth, pred, 0.5)
            b, _ = layout_scores(gt, truth, pred, 0.3)
            gt_n += len(gt)
            pred_n += len(pred)
            m5 += a
            m3 += b
            orph5 += o
    return {"gt": gt_n, "pred": pred_n, "recall50": m5 / gt_n if gt_n else 0.0,
            "recall30": m3 / gt_n if gt_n else 0.0, "orphans50": orph5}


def collect(decks: dict) -> dict:
    """Score everything present under OUT/raw; returns rows per section."""
    raw_dir = OUT / "raw"
    rows = {"A": [], "B": [], "B_clip": [], "C": []}
    for name in ENGINES:
        meta_dir = raw_dir / "meta"
        for task, fname in (("ocr", f"{name}.json"), ("caption", f"captions_{name}.json"),
                            ("clip", f"clip_{name}.json"), ("layout", f"layout_{name}.json")):
            for variant in ([""] if task != "layout" or not name.startswith("florence") else ["od", "rp"]):
                fn = fname if not variant else f"layout_{name}-{variant}.json"
                d = _load(raw_dir / fn)
                if d is None:
                    continue
                meta = _load(meta_dir / fn) or {}
                det = determinism(raw_dir / fn, raw_dir / "pass2" / fn)
                base = {"engine": name + (f" <{ {'od': 'OD', 'rp': 'REGION_PROPOSAL'}[variant]}>" if variant else ""),
                        "vram_mb": meta.get("vram_peak_mb"), "sec": meta.get("seconds_per_item"),
                        "det": det, "meta": meta}
                if task == "ocr":
                    rows["A"].append({**base, **score_ocr(d, decks)})
                elif task == "caption":
                    rows["B"].append({**base, **score_captions(d, decks)})
                elif task == "clip":
                    rows["B_clip"].append({**base, **score_clip(d, decks)})
                else:
                    rows["C"].append({**base, **score_layout(d, decks)})
    return rows


def _f(x, pct=False, nd=2):
    if x is None:
        return "–"
    return f"{100 * x:.1f}%" if pct else f"{x:.{nd}f}"


def report(decks: dict) -> str:
    rows = collect(decks)
    L = []
    A = rows["A"]
    L.append("## A. Read text from pixels\n")
    L.append("| engine | unit | word recall | word precision | precision excl. text inside images | "
             "pred words / GT words | VRAM peak MB | s/slide | deterministic | hit token cap |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in A:
        m = r["meta"]
        L.append(f"| {r['engine']} | {m.get('unit', '?')} | {_f(r['recall'], 1)} | {_f(r['precision'], 1)} | "
                 f"{_f(r['precision_ex_img'], 1) if m.get('has_boxes') else 'n/a (no boxes)'} | "
                 f"{r['pred_words']} / {r['gt_words']} | {_f(r['vram_mb'], nd=0)} | {_f(r['sec'])} | {r['det']} | "
                 f"{m.get('hit_token_cap', '–')} |")
    L.append("\n## B. Describe images (needs_vision)\n")
    L.append("| engine | content imgs | caption shares word w/ text ≤1in | (of imgs with text ≤1in) | "
             "same, pictures w/o overlaid text only | icons | "
             "icon captions minting a new word (risk) | mean caption words | VRAM peak MB | s/image | deterministic |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows["B"]:
        extra = f" (top-1 p≥0.5: {r['icon_conf50']})" if r["engine"].startswith("vit") else ""
        L.append(f"| {r['engine']} | {r['content']} | {r['hits']} = {_f(r['hit_rate'], 1)} | "
                 f"{r['hits']}/{r['with_neighbour']} = {_f(r['hit_rate_reachable'], 1)} | "
                 f"{r['picture_hits']}/{r['pictures']} | {r['icons']} | "
                 f"{r['icon_risk']} = {_f(r['icon_risk_rate'], 1)}{extra} | {_f(r['mean_words'], nd=1)} | "
                 f"{_f(r['vram_mb'], nd=0)} | {_f(r['sec'], nd=3)} | {r['det']} |")
    L.append("\nCLIP as a scorer (rank slide's own text elements by image–text similarity; target = "
             "spatially nearest text element by centre distance):\n")
    L.append("| engine | content imgs scored | nearest text top-1 | top-3 | chance top-1 | chance top-3 | "
             "pictures w/o overlaid text: top-1 / top-3 | VRAM peak MB | s/image | deterministic |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows["B_clip"]:
        L.append(f"| {r['engine']} | {r['images']} | {_f(r['top1'], 1)} | {_f(r['top3'], 1)} | "
                 f"{_f(r['chance1'], 1)} | {_f(r['chance3'], 1)} | "
                 f"{r['pic_top1']}/{r['pictures']} / {r['pic_top3']}/{r['pictures']} | {_f(r['vram_mb'], nd=0)} | "
                 f"{_f(r['sec'], nd=3)} | {r['det']} |")
    L.append("\n## C. Layout from pixels\n")
    L.append("| engine | GT text boxes | recall IoU≥0.5 | recall IoU≥0.3 | predicted regions | "
             "predicted w/o truth match (IoU≥0.5) | VRAM peak MB | s/slide | deterministic |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows["C"]:
        L.append(f"| {r['engine']} | {r['gt']} | {_f(r['recall50'], 1)} | {_f(r['recall30'], 1)} | {r['pred']} | "
                 f"{r['orphans50']} | {_f(r['vram_mb'], nd=0)} | {_f(r['sec'])} | {r['det']} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--only", nargs="*")
    r.add_argument("--passes", type=int, default=2)
    r.add_argument("--limit", type=int, default=None,
                   help="pass 2 only re-runs the first N jobs (slides or images) - for slow VLMs")
    sub.add_parser("tables")
    sub.add_parser("spot", help="every captioner's output on picture-type content images + 8 icons")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.only, a.passes, a.limit)
    elif a.cmd == "spot":
        print(spot(load_decks()))
    else:
        print(report(load_decks()))
    return 0


def spot(decks: dict) -> str:
    """Side-by-side captions for eyeballing (pictures w/o overlaid text, then every 30th icon)."""
    caps = {n: _load(OUT / "raw" / f"captions_{n}.json") for n in ENGINES}
    caps = {n: c for n, c in caps.items() if c}
    out = []
    for want in ("picture", "icon"):
        k = 0
        for deck, t in decks.items():
            W, H = t["slide_size"]["w"], t["slide_size"]["h"]
            for s in t["slides"]:
                texts = anchor_text_elements(s, W, H)
                for e in s["elements"]:
                    if e["type"] != "image":
                        continue
                    kind = classify_image(e["bbox"], W, H)
                    if want == "picture" and (kind != "content" or has_overlaid_text(e["bbox"], texts)):
                        continue
                    if want == "icon":
                        if kind != "icon":
                            continue
                        k += 1
                        if k % 30:
                            continue
                    near = nearest_text_id(e["bbox"], texts)
                    near_t = next((x["text"] for x in texts if x["id"] == near), "")
                    out.append(f"\n[{want}] {deck} slide {s['slide_number']} {e['id']} "
                               f"bbox={[round(e['bbox'][q], 2) for q in 'xywh']} nearest text: {near_t[:70]!r}")
                    for n, c in caps.items():
                        rec = c.get(deck, {}).get(e["id"])
                        if rec:
                            out.append(f"   {n:16s} {rec['caption'][:110]!r}")
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
