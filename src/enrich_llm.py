"""LLM trigger-phrase enrichment.

For each text-bearing element in the index, ask a small instruction-tuned
LLM (default: Qwen2.5-1.5B-Instruct) for a few short phrases a presenter
might say to highlight this element. Append the phrases to
`trigger_phrases` so the matcher's semantic embedding has more surface
forms to anchor to.

Output:
    out/index.enriched.json   — input index with expanded trigger_phrases

The embedding cache (out/embeddings_*.pkl) is keyed by index-content hash,
so swapping in this file automatically forces a fresh embedding pass on
first load.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# elements with a short text label benefit most; skip arrows/images
_ENRICHABLE_TYPES = {
    "title", "body", "caption", "bullet", "stat",
    "table_cell", "chart_category", "diagram_label", "vlm_region",
}

_SYS = (
    "You produce paraphrases. Given a snippet of text from a presentation slide, "
    "output 3-5 short alternative phrasings of that exact snippet — different "
    "words a speaker might use to refer to the SAME thing. "
    "Rules: Each phrase must be a natural-sounding fragment a presenter "
    "could say (e.g., 'the parser stage', 'when we parse the deck'). "
    "Do NOT output imperative instructions ('show me', 'highlight as', 'start with'). "
    "Do NOT output meta-statements about presenting. "
    "Do NOT use quotes or numbering. "
    "Output one phrase per line, 3-8 words each."
)


_EXAMPLES = (
    "Examples of GOOD paraphrases:\n"
    'Snippet: "Parse the deck offline and index every pointable element"\n'
    "  parsing the deck\n"
    "  the offline indexing step\n"
    "  building the index ahead of time\n"
    "  when we parse and index\n"
    "Snippet: \"Speech to text\"\n"
    "  the speech recognition step\n"
    "  transcribing the audio\n"
    "  the STT engine\n"
    "  converting audio to words\n"
)


def _build_prompt(slide_title: Optional[str], el_type: str, el_text: str, n: int = 4) -> str:
    title_clause = f' (from a slide titled "{slide_title}")' if slide_title else ""
    return (
        f"{_EXAMPLES}\n"
        f'Now do the same. Snippet: "{el_text}"{title_clause}\n'
        f"Give {n} paraphrases, one per line, no numbering, no quotes."
    )


_BAD_OPENERS = (
    # imperative / meta starters the small LLM tends to emit
    "show me", "show the", "highlight as", "highlight it",
    "start with", "begin with", "specify it", "note it", "note that",
    "introduce", "describe the", "describe a",
    "this is the", "this is a", "that is the", "that's where",
    "we should", "you should", "let me", "let's",
    # leftover schema noise
    "snippet:", "paraphrases:", "example", "examples",
)

# tokens that strongly indicate a meta-instruction even mid-phrase
_BAD_SUBSTRINGS = ('"', "''", "highlight as ", "specify it's", "note it's")


def _clean_phrases(text: str, original_text: str, max_phrases: int = 6) -> list[str]:
    out: list[str] = []
    original = original_text.lower().strip()
    seen = {original}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        line = re.sub(r"^[\-\*\d\.\)\(:•]+\s*", "", line)
        line = line.strip(' "\'`,.')
        if not line:
            continue
        words = line.split()
        if len(words) > 12 or len(words) < 2:
            continue
        low = line.lower()
        if low in seen:
            continue
        if any(low.startswith(p) for p in _BAD_OPENERS):
            continue
        if any(s in low for s in _BAD_SUBSTRINGS):
            continue
        # reject phrases that are too similar to the original (just a synonym swap)
        if original and original in low and len(words) <= len(original.split()) + 1:
            continue
        seen.add(low)
        out.append(low)
        if len(out) >= max_phrases:
            break
    return out


class LLMEnricher:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"loading {model_name} on {self.device} ({dtype})…")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=self.device,
        ).eval()
        self._torch = torch

    def generate_phrases(self, slide_title: Optional[str], el_type: str, el_text: str, n: int = 4) -> list[str]:
        if not el_text or not el_text.strip():
            return []
        messages = [
            {"role": "system", "content": _SYS},
            {"role": "user", "content": _build_prompt(slide_title, el_type, el_text, n=n)},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0, inputs.input_ids.shape[1]:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True)
        return _clean_phrases(text, el_text)


def enrich_index(index_path: Path, out_path: Path,
                 model_name: str = DEFAULT_MODEL,
                 phrases_per_element: int = 4) -> dict:
    raw = json.loads(index_path.read_text())
    enricher = LLMEnricher(model_name=model_name)
    added_total = 0
    for slide in raw["slides"]:
        sno = slide["slide_number"]
        # pick a slide title for context (first title/body element)
        slide_title = None
        for el in slide["elements"]:
            if el["type"] == "title":
                slide_title = el.get("text")
                break
        if not slide_title:
            for el in slide["elements"]:
                if el["type"] in ("body", "caption") and el.get("text"):
                    slide_title = el["text"]
                    break
        added_slide = 0
        for el in slide["elements"]:
            if el["type"] not in _ENRICHABLE_TYPES:
                continue
            text = el.get("text")
            if not text:
                continue
            # Skip very short labels — a 1.5B model invents irrelevant
            # context when the text is one or two tokens.
            if len(text.split()) < 3 and el["type"] in ("diagram_label", "chart_category", "table_cell", "stat"):
                continue
            existing = {p.lower() for p in (el.get("trigger_phrases") or [])}
            new_phrases = enricher.generate_phrases(slide_title, el["type"], text,
                                                    n=phrases_per_element)
            fresh = [p for p in new_phrases if p not in existing]
            if fresh:
                el["trigger_phrases"] = (el.get("trigger_phrases") or []) + fresh
                added_slide += len(fresh)
        print(f"  slide {sno}: +{added_slide} new phrases")
        added_total += added_slide

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_path}  (+{added_total} trigger phrases total)")
    return raw


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LLM-driven trigger-phrase enrichment.")
    ap.add_argument("--input", default="out/index.vlm.json",
                    help="index to enrich (defaults to the VLM-enriched one if present)")
    ap.add_argument("--out", default="out/index.enriched.json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("-n", "--phrases", type=int, default=4)
    args = ap.parse_args(argv)
    in_path = Path(args.input)
    if not in_path.exists():
        # fall back to the raw parser output if no VLM-enriched index exists yet
        in_path = Path("out/index.json")
    enrich_index(
        index_path=in_path,
        out_path=Path(args.out),
        model_name=args.model,
        phrases_per_element=args.phrases,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
