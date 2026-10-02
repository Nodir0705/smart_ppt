"""Headless grounding-accuracy audit.

Runs the live pipeline (TTS wav -> STT -> matcher -> policy, muted, real-time
paced) for every slide in the narration, records the highlight trajectory, and
scores it against tests/fixtures/expected_highlights.json.

Metric — ordered recall: for each slide, walk the expected ordered steps while
advancing a pointer through the actual (consecutive-deduped) trajectory; a step
"hits" if any acceptable id for that step appears at/after the pointer. Reports
per-slide hits/steps and an overall accuracy, plus a flicker count (switches that
revert within 2 updates) and the list of MISSED expected elements (the failures).

Usage:
    python3 -m tools.timeline_audit                       # uses out/index.refined.json
    python3 -m tools.timeline_audit --index out/index.scoped.json
    python3 -m tools.timeline_audit --only 4              # one slide
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.audio import FileSource, PlaybackSource
from src.encoder import eval_device, load_default_encoder
from src.live import LiveTranscriber
from src.live_controller import LiveController
from src.matcher import Index, Matcher
from src.stt import load_default_stt_engine, hotwords_from_index

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "expected_highlights.json"


def record_trajectory(sno, wav, raw_index, engine, matcher, normalize=None,
                      tail_words=8, fast=False):
    """Return (events, flicker) where events is a list of (t, active_id, tail);
    t is audio seconds when `fast`, else wall seconds since the replay started.

    Decisions come from the same ``LiveController`` the viewer and simulators
    use, so the audit measures exactly the live tail + policy logic.
    """
    ctl = LiveController(matcher, raw_index, normalizer=normalize or (lambda t: t),
                         tail_words=tail_words, slide_no=sno)
    buf = ctl.buffer
    if fast and not getattr(engine, "audio_clock", False):
        raise ValueError("fast replay needs an audio_clock STT engine")
    # fast: unpaced file replay; same 100 ms chunking as PlaybackSource
    src = FileSource(str(wav), chunk_ms=100) if fast else PlaybackSource(str(wav), mute=True)
    tr = LiveTranscriber(src, engine, buf, hotwords=hotwords_from_index(raw_index))
    events = []
    last = "___init___"
    t0 = time.monotonic()

    def on_update(text, partial):
        nonlocal last
        ev = ctl.feed()
        if ev is None:
            return
        if ev.active_id != last:
            # fast replay is unpaced: wall time is meaningless, the partial's
            # audio-clock stamp is when it was said
            t = partial.timestamp if fast else time.monotonic() - t0
            events.append((t, ev.active_id, ev.tail))
            last = ev.active_id

    tr.run(on_update=on_update)
    # flicker: A -> B -> A within 2 transitions
    ids = [e[1] for e in events]
    flicker = sum(
        1 for i in range(2, len(ids))
        if ids[i] is not None and ids[i] == ids[i - 2] and ids[i] != ids[i - 1]
    )
    return events, flicker


def unit_groups(expected_steps, unit_of):
    """Expected steps at unit level: each step's ids mapped to their pointable
    units; consecutive steps naming the same unit(s) are ONE point (talking
    through a card's heading then its description keeps one box). Returns
    [(unit ids, original steps it covers)]."""
    groups: list[tuple[list[str], list[list[str]]]] = []
    for step in expected_steps:
        us = sorted({unit_of.get(i, i) for i in step})
        if groups and groups[-1][0] == us:
            groups[-1][1].append(step)
        else:
            groups.append((us, [step]))
    return groups


def score_slide(expected_steps, trajectory_ids, unit_of=None):
    """Ordered recall with alternates, plus precision.

    With `unit_of` (element id -> pointable unit id) both sides are compared at
    unit level: a highlight counts for a step when it is the unit of one of the
    step's ids, and a unit group that hits credits every original step it
    covers, so totals stay on the original-step ruler.

    Returns (hits, total, missed_steps, relevant, false):
      - hits/total: ordered-recall (did each expected step get highlighted, in
        order, allowing alternates).
      - relevant/false: precision side — distinct highlighted ids that ARE vs
        are NOT in the slide's acceptable set (anything the narration refers to).
        `false` counts highlights on elements the narration never mentioned, the
        failure ordered-recall is blind to.
    """
    # dedup consecutive
    actual = []
    for x in trajectory_ids:
        if x is None:
            continue
        if not actual or actual[-1] != x:
            actual.append(x)
    groups = (unit_groups(expected_steps, unit_of) if unit_of
              else [(step, [step]) for step in expected_steps])
    ptr = 0
    hits = 0
    missed = []
    for ids, covered in groups:
        found_at = None
        for j in range(ptr, len(actual)):
            if actual[j] in ids:
                found_at = j
                break
        if found_at is not None:
            hits += len(covered)
            ptr = found_at + 1
        else:
            missed.extend(covered)
    acceptable = set().union(*(ids for ids, _ in groups)) if groups else set()
    distinct = set(actual)
    relevant = sum(1 for x in distinct if x in acceptable)
    false = len(distinct) - relevant
    return hits, len(expected_steps), missed, relevant, false


def _eval_items(manifest, gt, only):
    """Manifest slides to evaluate, and those skipped for having no expected steps."""
    items, skipped = [], []
    for item in manifest:
        sno = item["slide"]
        if only is not None and sno != only:
            continue
        if sno in gt:
            items.append(item)
        else:
            skipped.append(sno)
    return items, skipped


def run(index_path, only=None, normalize=None, manifest_path=None, fixture_path=None,
        fast=False, models=None, device="cpu"):
    raw = json.loads(Path(index_path).read_text())
    manifest_path = Path(manifest_path or ROOT / "out" / "narration" / "manifest.json")
    manifest = json.loads(manifest_path.read_text())
    gt = {s["slide"]: s["expected"]
          for s in json.loads(Path(fixture_path or FIXTURE).read_text())["slides"]}

    if models is None:
        print(f"loading models (BGE + STT)…  index={index_path}", flush=True)
        models = (load_default_encoder(device), load_default_stt_engine(sample_rate=16000, audio_clock=fast))
    enc, engine = models
    idx = Index(Path(index_path), enc)
    matcher = Matcher(idx, enc)

    tot_hits = tot_steps = tot_flicker = 0
    tot_relevant = tot_false = 0
    per_slide: dict[int, dict] = {}
    events_by_slide: dict[int, list] = {}
    items, skipped = _eval_items(manifest, gt, only)
    for item in items:
        sno = item["slide"]
        events, flicker = record_trajectory(sno, item["wav"], raw, engine, matcher,
                                             normalize=normalize, fast=fast)
        events_by_slide[sno] = events
        traj = [e[1] for e in events]
        hits, steps, missed, relevant, false = score_slide(gt[sno], traj, idx.unit_of)
        tot_hits += hits
        tot_steps += steps
        tot_flicker += flicker
        tot_relevant += relevant
        tot_false += false
        per_slide[sno] = {"hits": hits, "steps": steps, "missed": missed,
                          "flicker": flicker, "relevant": relevant, "false": false,
                          "trajectory": [x for x in traj if x]}
        flag = "" if hits == steps else f"   MISSED {missed}"
        fflag = f"   FALSE={false}" if false else ""
        print(f"slide {sno}: {hits}/{steps}  flicker={flicker}{fflag}{flag}")
        print(f"          trajectory: {[x for x in traj]}")
    acc = tot_hits / tot_steps if tot_steps else 0.0
    total_hl = tot_relevant + tot_false
    precision = tot_relevant / total_hl if total_hl else 1.0
    if skipped:
        print(f"skipped (no expected steps): {skipped}")
    print(f"\nOVERALL: recall {tot_hits}/{tot_steps} = {acc:.3f}   "
          f"precision {tot_relevant}/{total_hl} = {precision:.3f}   "
          f"false_highlights={tot_false}   total_flicker={tot_flicker}")
    return {"slides": per_slide, "events": events_by_slide,
            "overall": {"hits": tot_hits, "steps": tot_steps, "flicker": tot_flicker,
                        "relevant": tot_relevant, "false": tot_false,
                        "precision": precision, "skipped": skipped}}


def audit(index_path, normalize_index=True, **kw):
    """Programmatic entry: returns the structured result dict (see run())."""
    normalize = None
    if normalize_index:
        from src.asr_normalize import build_alias_map, normalize as _norm
        raw = json.loads(Path(index_path).read_text())
        amap = build_alias_map(raw)
        normalize = (lambda t: _norm(t, amap)) if amap else None
    return run(index_path, normalize=normalize, **kw)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Grounding-accuracy audit.")
    ap.add_argument("--index", default="out/index.refined.json")
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--normalize", action="store_true",
                    help="apply src.asr_normalize to the STT tail before matching")
    ap.add_argument("--manifest", default=None, help="narration manifest JSON")
    ap.add_argument("--fixture", default=None, help="expected highlights JSON")
    ap.add_argument("--fast", action="store_true",
                    help="unpaced replay on the STT audio clock (bench mode)")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                    help="text-encoder device for this eval (auto = cuda if available); STT stays CPU")
    args = ap.parse_args(argv)
    normalize = None
    if args.normalize:
        from src.asr_normalize import build_alias_map, normalize as _norm
        raw = json.loads(Path(args.index).read_text())
        amap = build_alias_map(raw)
        normalize = lambda t: _norm(t, amap)
    run(args.index, only=args.only, normalize=normalize, manifest_path=args.manifest,
        fixture_path=args.fixture, fast=args.fast, device=eval_device(args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
