"""Index-build stage: link text-less pictures to the slide text they illustrate.

A picture with no alt text / OCR text (``type == "image"``, ``needs_vision``) is
unpointable. Here CLIP (open_clip ViT-B-32 LAION-2B, via transformers) scores the
picture's crop from the rendered slide PNG against each of the slide's text
elements; when the best text clears an absolute bar AND beats the runner-up by a
margin, the picture is linked to it and gets *visual* trigger phrases ("in the
photo", ...) — never the text's own words, which would compete with the text
element. So "...the Snapdragon 845" lights the text and "...that's the package
in the photo" lights the picture. Abstaining keeps panels/diagrams unpointable.

Build-time only (CPU fp32 for bit-stable scores); the live loop never loads CLIP.
Missing torch / model -> log and skip; pictures stay needs_vision.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

from .geom import inches_to_image_px

_log = logging.getLogger(__name__)

MODEL_ID = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
MIN_AREA = 0.5           # in^2: below this it's an icon/bullet glyph
BACKGROUND_FRAC = 0.9    # w AND h >= 90% of the slide -> background
FOOTER_FRAC = 0.9        # text starting below 90% of slide height is footer
TEXT_TYPES_EXCLUDED = {"chart", "chart_category", "image", "picture", "arrow"}
# Link decision on CLIP cosine similarity (image crop vs raw element text). Fit on
# the 6 candidate pictures of the 3 bench decks (2026-10-02): the 3 right links
# score 0.299-0.455 with margin >= 0.026; the rest (a chip-in-hand photo on a
# driver slide, a diagram strip, a placeholder glyph) <= 0.271. Abstain-biased.
MIN_SCORE = 0.28
MIN_MARGIN = 0.02


# ------------------------------------------------------------- geometry

def _xyxy(b: dict) -> tuple[float, float, float, float]:
    return b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]


def _contains_center(outer: dict, inner: dict) -> bool:
    x0, y0, x1, y1 = _xyxy(outer)
    cx, cy = inner["x"] + inner["w"] / 2, inner["y"] + inner["h"] / 2
    return x0 <= cx <= x1 and y0 <= cy <= y1


def crop_box(bbox: dict, png_w: int, png_h: int, slide_w: float, slide_h: float):
    """Inch bbox -> clipped integer pixel box (l, t, r, b) on the slide PNG; None if empty."""
    x, y, w, h = inches_to_image_px((bbox["x"], bbox["y"], bbox["w"], bbox["h"]),
                                    slide_w, slide_h, png_w, png_h)
    l, t = max(0, math.floor(x)), max(0, math.floor(y))
    r, b = min(png_w, math.ceil(x + w)), min(png_h, math.ceil(y + h))
    return (l, t, r, b) if r > l and b > t else None


# ------------------------------------------------------------ candidates

def text_candidates(slide: dict, slide_w: float, slide_h: float) -> list[dict]:
    """Text elements a picture may illustrate: has text, not chart/image, not footer/full-slide."""
    return [e for e in slide["elements"]
            if (e.get("text") or "").strip() and e["type"] not in TEXT_TYPES_EXCLUDED
            and e["bbox"]["y"] < FOOTER_FRAC * slide_h
            and not (e["bbox"]["w"] >= BACKGROUND_FRAC * slide_w
                     and e["bbox"]["h"] >= BACKGROUND_FRAC * slide_h)]


def picture_candidates(slide: dict, slide_w: float, slide_h: float) -> list[dict]:
    """Text-less pictures worth linking.

    Drops icons (< MIN_AREA), backgrounds (near full-slide), panels/cards (they
    own text: the words are already pointable), and of two nested pictures
    (frame + photo) keeps only the inner one, so the pair never ties on
    identical phrases. A text element is owned by the SMALLEST image containing
    its centre, so a caption band laid over a photo owns the caption, not the photo.
    """
    area = lambda e: e["bbox"]["w"] * e["bbox"]["h"]  # noqa: E731
    images = [e for e in slide["elements"] if e["type"] == "image"
              and not (e["bbox"]["w"] >= BACKGROUND_FRAC * slide_w
                       and e["bbox"]["h"] >= BACKGROUND_FRAC * slide_h)]
    owners = set()
    for t in text_candidates(slide, slide_w, slide_h):
        holders = [i for i in images if _contains_center(i["bbox"], t["bbox"])]
        if holders:
            owners.add(min(holders, key=lambda i: (area(i), i["id"]))["id"])
    pics = [e for e in images
            if e.get("needs_vision") and not (e.get("text") or "").strip()
            and area(e) >= MIN_AREA and e["id"] not in owners]
    return [p for p in pics
            if not any(q is not p and area(q) < area(p) and _contains_center(p["bbox"], q["bbox"])
                       for q in pics)]


# --------------------------------------------------------------- decision

def decide(scores: dict[str, float], min_score: float = MIN_SCORE,
           min_margin: float = MIN_MARGIN) -> tuple[str, float, float] | None:
    """(best_id, score, margin) if the best clears the bar and the runner-up margin; else None.
    Ties break by id so the result is deterministic."""
    if not scores:
        return None
    order = sorted(scores, key=lambda k: (-scores[k], k))
    best = scores[order[0]]
    margin = best - scores[order[1]] if len(order) > 1 else best
    if best < min_score or margin < min_margin:
        return None
    return order[0], best, margin


# ---------------------------------------------------------------- phrases

# What a presenter says when pointing at a picture ("...that's the package in the
# photo"). Deliberately free of the linked text: on the Open-Q bench, "the photo
# of X" / "X photo" phrasings let the picture steal X's own mentions (verbatim
# narration -3 hits, +3 false highlights), while these lit all 3 annotated
# pictures and never fired on verbatim or filler speech.
# ponytail: two linked pictures on one slide tie on these and neither lights
# (abstain); add "the photo of X" forms for that case if a deck needs it.
VISUAL_PHRASES = ("in the photo", "shown in the photo", "in the picture", "the photo shows")


# ---------------------------------------------------------------- scorer

def load_clip_scorer():
    """CPU fp32 CLIP scorer: (PIL crop, [texts]) -> [cosine]. None when unavailable."""
    try:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from transformers import CLIPModel, CLIPProcessor
        proc = CLIPProcessor.from_pretrained(MODEL_ID, use_fast=False)  # pin: fast differs numerically
        model = CLIPModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()
    except Exception as e:  # torch/transformers/model missing -> skip the stage
        _log.warning("vision_link: CLIP unavailable (%s: %s); pictures stay needs_vision",
                     type(e).__name__, e)
        return None

    @torch.inference_mode()
    def score(image, texts: list[str]) -> list[float]:
        inp = proc(text=texts, images=image, return_tensors="pt", padding=True,
                   truncation=True, max_length=77)
        out = model(**inp)
        img = torch.nn.functional.normalize(out.image_embeds, dim=-1)
        txt = torch.nn.functional.normalize(out.text_embeds, dim=-1)
        return (txt @ img[0]).tolist()

    return score


# ------------------------------------------------------------------ stage

def link_pictures(index: dict, slides_dir: str | Path, scorer=None,
                  min_score: float = MIN_SCORE, min_margin: float = MIN_MARGIN) -> list[dict]:
    """Link pictures in `index` (mutated) to text; returns one record per scored picture
    (linked or not) for the build's link table. `scorer` defaults to CLIP."""
    from PIL import Image
    slides_dir = Path(slides_dir)
    W, H = index["slide_size"]["w"], index["slide_size"]["h"]
    jobs = []
    for slide in index["slides"]:
        png = slides_dir / f"slide_{slide['slide_number']:03d}.png"
        texts = text_candidates(slide, W, H)
        pics = picture_candidates(slide, W, H)
        if png.exists() and texts and pics:
            jobs.append((slide, png, texts, pics))
    if not jobs:
        return []
    scorer = scorer or load_clip_scorer()
    if scorer is None:
        return []
    records = []
    for slide, png, texts, pics in jobs:
        im = Image.open(png).convert("RGB")
        for pic in pics:
            box = crop_box(pic["bbox"], im.width, im.height, W, H)
            if box is None:
                continue
            sims = scorer(im.crop(box), [t["text"] for t in texts])
            scores = {t["id"]: round(float(s), 4) for t, s in zip(texts, sims)}
            d = decide(scores, min_score, min_margin)
            rec = {"slide": slide["slide_number"], "id": pic["id"], "scores": scores,
                   "linked_to": None}
            if d:
                tid, s, m = d
                text = next(t["text"] for t in texts if t["id"] == tid)
                pic.update(linked_to=tid, link_score=s, needs_vision=False,
                           trigger_phrases=list(VISUAL_PHRASES))
                rec.update(linked_to=tid, score=s, margin=round(m, 4), text=text)
            records.append(rec)
    return records


def format_links(records: list[dict]) -> str:
    lines = [f"vision_link: {sum(r['linked_to'] is not None for r in records)}"
             f"/{len(records)} pictures linked"]
    for r in records:
        if r["linked_to"]:
            lines.append(f"  slide {r['slide']:>3} {r['id']:<8} -> {r['linked_to']:<8} "
                         f"score {r['score']:.3f} margin {r['margin']:.3f}  {r['text'][:50]!r}")
        else:
            best = sorted(r["scores"].items(), key=lambda kv: (-kv[1], kv[0]))[:2]
            lines.append(f"  slide {r['slide']:>3} {r['id']:<8} -- abstain  top {best}")
    return "\n".join(lines)
