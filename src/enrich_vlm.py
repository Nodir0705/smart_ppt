"""VLM enrichment via focused single-answer questions.

A small (2B) VLM produces unreliable prose when asked open-ended "describe
what a presenter might say" prompts — it hallucinates context across charts
and confuses one chart's type for another's. But the same model is
*accurate* on focused single-answer questions ("what type of chart?" /
"what's the title?" / "what's the largest bar?").

So this enricher asks 2–3 narrow questions per visual element, then
assembles trigger phrases programmatically from the answers. No prose
parsing, no hallucinated cross-references.

Output:
    out/index.vlm.json            — input index with enriched trigger phrases
    out/index.vlm.vlm_debug.json  — raw question/answer log per element
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from PIL import Image


DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"


# ---------- prompts ----------

_CHART_Q_TYPE = (
    "What kind of chart is this? Answer with exactly one of these words: "
    "bar, column, pie, line, scatter."
)
_CHART_Q_TITLE = (
    "What is the title of this chart? Answer with just the title text, "
    "nothing else. If there is no title, answer 'none'."
)
_CHART_Q_MAX = (
    "What is the label of the largest bar or slice in this chart? "
    "Answer with just the category name (e.g., 'STT' or 'Parsing'). "
    "If you can't tell, answer 'none'."
)

_IMAGE_Q_KIND = (
    "What is this visual? Answer with one short phrase like "
    "'a diagram', 'a photo', 'an icon', 'an illustration', 'a logo'. "
    "Just the phrase, nothing else."
)
_IMAGE_Q_DESC = (
    "Describe what is visible in this image in one short phrase, "
    "under 10 words. Just the phrase, no preamble."
)


# ---------- answer sanitation ----------

_NONE_ANSWERS = {"none", "n/a", "no title", "unknown", "unsure", "i can't tell"}

_TYPE_NORMALIZE = {
    "bar": "bar", "column": "bar",  # treat both as "bar" — same vocabulary
    "pie": "pie", "donut": "pie",
    "line": "line", "scatter": "scatter", "area": "line",
}


def _sanitize(ans: str) -> str:
    """Strip quotes, periods, leading/trailing whitespace; collapse whitespace."""
    s = ans.strip().strip('"\'`')
    s = s.strip().rstrip(".").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _is_none(s: str) -> bool:
    return s.lower() in _NONE_ANSWERS or not s


def _extract_chart_type(raw: str) -> Optional[str]:
    low = raw.lower()
    for keyword, normalized in _TYPE_NORMALIZE.items():
        if keyword in low:
            return normalized
    return None


# ---------- crop helper ----------

def _crop_for_bbox(slide_img: Image.Image, bbox_in: tuple[float, float, float, float],
                   slide_w_in: float, slide_h_in: float, pad_px: int = 8) -> Image.Image:
    iw, ih = slide_img.size
    x, y, w, h = bbox_in
    sx = iw / slide_w_in
    sy = ih / slide_h_in
    l = max(0, int(x * sx) - pad_px)
    t = max(0, int(y * sy) - pad_px)
    r = min(iw, int((x + w) * sx) + pad_px)
    b = min(ih, int((y + h) * sy) + pad_px)
    return slide_img.crop((l, t, r, b))


# ---------- trigger phrase assembly ----------

_SUPERLATIVE_BY_TYPE = {
    "bar":     {"noun": "bar",   "max_adj": "tallest"},
    "pie":     {"noun": "slice", "max_adj": "biggest"},
    "line":    {"noun": "point", "max_adj": "highest"},
    "scatter": {"noun": "point", "max_adj": "highest"},
}


def _phrases_for_chart(chart_type: Optional[str], title: Optional[str],
                       max_label: Optional[str]) -> list[str]:
    out: list[str] = []
    type_word = chart_type or "chart"
    style = _SUPERLATIVE_BY_TYPE.get(chart_type or "", {"noun": "bar", "max_adj": "tallest"})
    noun = style["noun"]
    max_adj = style["max_adj"]

    # type-based phrases
    out.append(f"the {type_word} chart")
    out.append(f"this {type_word} chart")
    if chart_type:
        out.append(f"the {chart_type}")

    # title-based phrases
    if title:
        t_low = title.lower().rstrip(".:")
        out.append(t_low)
        out.append(f"the {t_low} chart")

    # Generic superlative phrases on the WHOLE chart — these don't identify
    # a specific category, so "the largest bar" matches the chart-level
    # element. Label-specific phrases like "the STT bar" live on the
    # chart_category element (handled by the parser).
    if chart_type:
        out.append(f"the largest {noun}")
        out.append(f"the biggest {noun}")
        out.append(f"the {max_adj} {noun}")

    # de-dup while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def _strip_leading_article(s: str) -> str:
    parts = s.split(maxsplit=1)
    if len(parts) == 2 and parts[0].lower() in {"a", "an", "the"}:
        return parts[1]
    return s


def _phrases_for_image(kind: Optional[str], description: Optional[str]) -> list[str]:
    out: list[str] = ["the image", "the picture", "the figure", "the visual"]
    if kind:
        # strip a leading article so we don't end up with "the an icon"
        k = _strip_leading_article(kind.lower())
        out.append(f"the {k}")
        out.append(f"the {k} on the slide")
    if description:
        out.append(description.lower())
    seen: set[str] = set()
    deduped: list[str] = []
    for p in out:
        if p and p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


# ---------- VLM wrapper ----------

class VLMEnricher:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"loading {model_name} on {self.device} ({dtype})…")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype, device_map=self.device,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self._torch = torch

    def ask(self, image: Image.Image, prompt: str, max_new_tokens: int = 40) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen_ids = out[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]


# ---------- per-element enrichers ----------

def _enrich_chart(enricher: VLMEnricher, crop: Image.Image) -> tuple[list[str], dict]:
    a_type = _sanitize(enricher.ask(crop, _CHART_Q_TYPE))
    a_title = _sanitize(enricher.ask(crop, _CHART_Q_TITLE))
    a_max = _sanitize(enricher.ask(crop, _CHART_Q_MAX))
    chart_type = _extract_chart_type(a_type)
    title = None if _is_none(a_title) else a_title
    max_label = None if _is_none(a_max) else a_max
    qa = {"type": a_type, "title": a_title, "max": a_max,
          "normalized": {"chart_type": chart_type, "title": title, "max_label": max_label}}
    return _phrases_for_chart(chart_type, title, max_label), qa


def _enrich_image(enricher: VLMEnricher, crop: Image.Image) -> tuple[list[str], dict]:
    a_kind = _sanitize(enricher.ask(crop, _IMAGE_Q_KIND))
    a_desc = _sanitize(enricher.ask(crop, _IMAGE_Q_DESC, max_new_tokens=30))
    kind = None if _is_none(a_kind) else a_kind
    desc = None if _is_none(a_desc) else a_desc
    qa = {"kind": a_kind, "description": a_desc,
          "normalized": {"kind": kind, "description": desc}}
    return _phrases_for_image(kind, desc), qa


# ---------- index-level pass ----------

def enrich_index(index_path: Path, slides_dir: Path, out_path: Path,
                 model_name: str = DEFAULT_MODEL) -> dict:
    raw = json.loads(index_path.read_text())
    slide_w_in = raw["slide_size"]["w"]
    slide_h_in = raw["slide_size"]["h"]

    enricher = VLMEnricher(model_name=model_name)
    raw_responses: dict[str, dict] = {}
    enriched_count = 0

    for slide in raw["slides"]:
        sno = slide["slide_number"]
        png = slides_dir / f"slide_{sno:03d}.png"
        if not png.exists():
            print(f"  slide {sno}: missing image, skipping")
            continue
        slide_img = Image.open(png).convert("RGB")
        for el in slide["elements"]:
            if el["type"] not in ("image", "chart"):
                continue
            bb = el["bbox"]
            crop = _crop_for_bbox(slide_img, (bb["x"], bb["y"], bb["w"], bb["h"]),
                                  slide_w_in, slide_h_in)
            print(f"  slide {sno} {el['id']} ({el['type']}, {crop.size[0]}x{crop.size[1]} crop) → asking VLM…")
            if el["type"] == "chart":
                phrases, qa = _enrich_chart(enricher, crop)
            else:
                phrases, qa = _enrich_image(enricher, crop)
            raw_responses[el["id"]] = qa
            existing = {p.lower() for p in (el.get("trigger_phrases") or [])}
            fresh = [p for p in phrases if p not in existing]
            el["trigger_phrases"] = (el.get("trigger_phrases") or []) + fresh
            if el["type"] == "image" and fresh:
                el["needs_vision"] = False
            enriched_count += 1
            print(f"    normalized: {qa['normalized']}")
            print(f"    + {len(fresh)} phrases: {fresh}")

    # Cross-element dedup per slide
    duplicate_drop_count = 0
    for slide in raw["slides"]:
        counts: Counter[str] = Counter()
        for el in slide["elements"]:
            for p in el.get("trigger_phrases") or []:
                counts[p.lower()] += 1
        dups = {p for p, c in counts.items() if c > 1}
        for el in slide["elements"]:
            phrases = el.get("trigger_phrases") or []
            kept = [p for p in phrases if p.lower() not in dups]
            if len(kept) != len(phrases):
                duplicate_drop_count += len(phrases) - len(kept)
                el["trigger_phrases"] = kept

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False))
    debug_path = out_path.with_suffix(".vlm_debug.json")
    debug_path.write_text(json.dumps(raw_responses, indent=2, ensure_ascii=False))
    print(f"\nenriched {enriched_count} elements "
          f"({duplicate_drop_count} cross-element dups dropped) → {out_path}")
    print(f"debug Q/A log → {debug_path}")
    return raw


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="VLM enrichment via focused questions.")
    ap.add_argument("--index", default="out/index.json")
    ap.add_argument("--slides", default="out/slides")
    ap.add_argument("--out", default="out/index.vlm.json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args(argv)
    enrich_index(
        index_path=Path(args.index),
        slides_dir=Path(args.slides),
        out_path=Path(args.out),
        model_name=args.model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
