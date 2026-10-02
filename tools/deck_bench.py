"""Multi-deck grounding benchmark.

For every decks/<name>.pptx: XML truth parse -> auto narration (tools.narrate) ->
Piper audio -> build_index for a named variant -> unpaced audit on the STT audio
clock -> out/bench/report_<variant>.{md,json}. See
docs/superpowers/specs/2026-10-01-deck-bench-design.md.

    python3 -m tools.deck_bench                       # all decks, variant "default"
    python3 -m tools.deck_bench --variant no-ocr
    python3 -m tools.deck_bench --only my_deck
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from src.build_index import build_index
from src.encoder import eval_device, load_default_encoder
from src.parser import parse_deck
from src.stt import load_default_stt_engine
from src.tts import synthesize_slides
from tools.narrate import narrate
from tools.timeline_audit import audit

ROOT = Path(__file__).resolve().parents[1]
VOICE = ROOT / "out" / "voices" / "en_US-amy-low.onnx"
GAP_MS = 900
# variant name -> build_index kwargs; later steps (PaddleOCR, VLMs) add entries
VARIANTS: dict[str, dict] = {"default": {}, "no-ocr": {"ocr": False},
                             "no-clip": {"vision_link": False}}

# Negative pass: presenter filler that names NO element, so every highlight is
# false. TUNE is what matcher thresholds may be fitted on; HOLDOUT (reported)
# shares no content word with TUNE or with any bench/hand deck text.
FILLER_TUNE = [
    "Okay, so let's take a look at this one.",
    "As you can see here, this is really important.",
    "Now I want to move on to the next part.",
    "And this is where things get interesting.",
    "Let me just point out a couple of things before we continue.",
    "So, any questions so far? Great, let's keep going.",
]
FILLER_HOLDOUT = [
    "Alright, bear with me for just a moment.",
    "You might be wondering why I bring this up.",
    "Honestly, I think people tend to underestimate this.",
    "Think about that, because we will revisit it later.",
    "I hope that makes sense to everybody in the room.",
    "So, with that said, let me carry on.",
]


def filler_script(slides: list[int], sentences: list[str]) -> list[dict]:
    """Four consecutive filler sentences per slide, rotated by slide number."""
    return [{"slide": n, "segments": sentences[n % 3: n % 3 + 4]} for n in slides]


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def inventory(index: dict) -> dict:
    """What the deck contains — sizes the OCR/layout/vision steps."""
    els = [e for s in index["slides"] for e in s["elements"]]
    imgs = [e for e in els if e["type"] == "image"]
    return {
        "slides": len(index["slides"]),
        "elements": len(els),
        "by_type": dict(sorted(Counter(e["type"] for e in els).items())),
        "needs_vision": sum(bool(e.get("needs_vision")) for e in els),
        "images_with_alt": sum(bool(e.get("text")) for e in imgs),
        "images_without_alt": sum(not e.get("text") for e in imgs),
        "image_only_slides": sum(
            1 for s in index["slides"]
            if s["elements"] and all(e["type"] == "image" for e in s["elements"])),
    }


def totals(records: list[dict]) -> dict:
    ok = [r for r in records if r["status"] == "OK"]
    hits = sum(r["hits"] for r in ok)
    steps = sum(r["steps"] for r in ok)
    relevant = sum(r["relevant"] for r in ok)
    false = sum(r["false"] for r in ok)
    return {"decks_ok": len(ok), "decks_failed": len(records) - len(ok),
            "hits": hits, "steps": steps, "recall": hits / steps if steps else 0.0,
            "false": false,
            "precision": relevant / (relevant + false) if relevant + false else 1.0,
            "flicker": sum(r["flicker"] for r in ok),
            "filler_false": sum(r.get("filler_false", 0) for r in ok)}


def bench_deck(deck: Path, variant: str, out_root: Path, models, rec: dict,
               reuse_index: bool = False) -> dict:
    """Fill `rec` in place, so a deck that fails mid-way keeps its inventory.
    `reuse_index` scores an existing <variant>/index.scoped.json instead of rebuilding it."""
    d = out_root / deck.stem
    truth = parse_deck(str(deck), ocr=False)          # narration source: XML only
    rec["inventory"] = inventory(truth)
    script, expected = narrate(truth)
    rec["skipped_slides"] = len(truth["slides"]) - len(script)
    _write(d / "narration.json", {"voice": str(VOICE), "gap_ms": GAP_MS, "slides": script})
    _write(d / "expected.json", {"slides": expected})
    _write(d / "manifest.json", synthesize_slides(script, VOICE, d / "audio", GAP_MS))

    t = time.monotonic()
    if not (reuse_index and (d / variant / "index.scoped.json").exists()):
        build_index(deck, d / variant, **VARIANTS[variant])
    rec["build_s"] = round(time.monotonic() - t, 1)

    t = time.monotonic()
    res = audit(d / variant / "index.scoped.json", manifest_path=d / "manifest.json",
                fixture_path=d / "expected.json", fast=True, models=models)
    rec["eval_s"] = round(time.monotonic() - t, 1)
    o = res["overall"]
    rec.update(hits=o["hits"], steps=o["steps"], relevant=o["relevant"], false=o["false"],
               precision=o["precision"], flicker=o["flicker"],
               recall=o["hits"] / o["steps"] if o["steps"] else 0.0,
               slides=res["slides"])

    # filler pass: same slides, narration that names nothing -> expected []
    slides = [s["slide"] for s in script]
    _write(d / "filler_expected.json", {"slides": [{"slide": n, "expected": []} for n in slides]})
    _write(d / "filler_manifest.json", synthesize_slides(
        filler_script(slides, FILLER_HOLDOUT), VOICE, d / "filler_audio", GAP_MS))
    fr = audit(d / variant / "index.scoped.json", manifest_path=d / "filler_manifest.json",
               fixture_path=d / "filler_expected.json", fast=True, models=models)
    rec.update(filler_false=fr["overall"]["false"],
               filler_lit={s: v["trajectory"] for s, v in fr["slides"].items() if v["trajectory"]},
               status="OK")
    return rec


def load_models(device: str = "cpu"):
    """(text encoder on `device`, CPU sherpa STT on the audio clock) for one process."""
    return load_default_encoder(device), load_default_stt_engine(sample_rate=16000, audio_clock=True)


def per_deck_processes(job, arg_tuples: list[tuple]):
    """Run `job(*args)` for each deck in its own spawned process (each loads its
    own models), yielding results in input order so reports stay deterministic.
    `job` must be a module-level function returning (result, captured_stdout)."""
    ctx = multiprocessing.get_context("spawn")   # CUDA can't be forked
    with ProcessPoolExecutor(max_workers=max(1, len(arg_tuples)), mp_context=ctx) as ex:
        futs = [ex.submit(job, *a) for a in arg_tuples]
        for f in futs:
            yield f.result()


def _deck_job(deck: Path, variant: str, out_root: Path, device: str,
              reuse_index: bool) -> tuple[dict, str]:
    rec, buf = {"deck": deck.stem}, io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            bench_deck(deck, variant, out_root, load_models(device), rec, reuse_index)
        except Exception as e:  # one bad deck never aborts the run
            traceback.print_exc(file=buf)
            rec["status"] = f"FAILED: {type(e).__name__}: {e}"
    return rec, buf.getvalue()


def _markdown(variant: str, records: list[dict], tot: dict) -> str:
    lines = [f"# Deck bench — variant `{variant}` (gap {GAP_MS} ms)", "",
             "| deck | status | recall | precision | false | flicker | slides (eval/skip) "
             "| needs_vision | images alt/no-alt | image-only slides | filler false | build s | eval s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        inv = r.get("inventory", {})
        if r["status"] == "OK":
            lines.append(
                f"| {r['deck']} | OK | {r['hits']}/{r['steps']} | {r['precision']:.3f} "
                f"| {r['false']} | {r['flicker']} "
                f"| {inv['slides'] - r['skipped_slides']}/{r['skipped_slides']} "
                f"| {inv['needs_vision']} | {inv['images_with_alt']}/{inv['images_without_alt']} "
                f"| {inv['image_only_slides']} | {r['filler_false']} | {r['build_s']} | {r['eval_s']} |")
        else:
            lines.append(f"| {r['deck']} | {r['status']} |" + " |" * 11)
    lines += ["", f"**Total:** recall {tot['hits']}/{tot['steps']} = {tot['recall']:.3f} · "
              f"precision {tot['precision']:.3f} · false {tot['false']} · flicker {tot['flicker']} · "
              f"filler false (holdout, expected none) {tot['filler_false']} · "
              f"decks OK {tot['decks_ok']}, failed {tot['decks_failed']}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score live grounding on every deck in a folder.")
    ap.add_argument("--decks", default="decks", help="folder of .pptx files (default: decks)")
    ap.add_argument("--out", default="out/bench", help="output root (default: out/bench)")
    ap.add_argument("--variant", default="default", choices=sorted(VARIANTS))
    ap.add_argument("--only", default=None, help="bench just this deck (file stem)")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                    help="text-encoder device (auto = cuda if available); STT stays CPU")
    ap.add_argument("--reuse-index", action="store_true",
                    help="score an existing <variant>/index.scoped.json instead of rebuilding it")
    args = ap.parse_args(argv)

    decks = sorted(Path(args.decks).glob("*.pptx"))
    if args.only:
        decks = [p for p in decks if p.stem == args.only]
    if not decks:
        print(f"no .pptx decks in {args.decks}", file=sys.stderr)
        return 2

    out_root = Path(args.out)
    device = eval_device(args.device)
    records = []
    for deck, (rec, log) in zip(decks, per_deck_processes(
            _deck_job, [(d, args.variant, out_root, device, args.reuse_index) for d in decks])):
        print(f"=== {deck.stem} ===\n{log}", end="", flush=True)
        records.append(rec)
    tot = totals(records)
    # --only writes its own report so a debug re-run never clobbers the full one
    name = f"report_{args.variant}" + (f"_{args.only}" if args.only else "")
    _write(out_root / f"{name}.json",
           {"variant": args.variant, "gap_ms": GAP_MS, "decks": records, "totals": tot})
    md = _markdown(args.variant, records, tot)
    (out_root / f"{name}.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
