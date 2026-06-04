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

from src.audio import PlaybackSource
from src.encoder import load_default_encoder
from src.live import LiveTranscriber
from src.matcher import Index, Matcher, MatchPolicy, PolicyConfig
from src.stt import RollingBuffer, VoskEngine, hotwords_from_index

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "expected_highlights.json"


def record_trajectory(sno, wav, raw_index, engine, matcher, normalize=None,
                      tail_words=8):
    """Return (events, flicker) where events is a list of (t, active_id, tail)."""
    policy = MatchPolicy(matcher, PolicyConfig())
    buf = RollingBuffer(window_seconds=4.0)
    src = PlaybackSource(str(wav), mute=True)
    tr = LiveTranscriber(src, engine, buf, hotwords=hotwords_from_index(raw_index))
    events = []
    last = "___init___"
    t0 = time.monotonic()

    def on_update(text, partial):
        nonlocal last
        tail = buf.tail(tail_words)
        if normalize is not None:
            tail = normalize(tail)
        if not tail:
            return
        _res, active, _why = policy.step(tail, sno)
        if active != last:
            events.append((time.monotonic() - t0, active, tail))
            last = active

    tr.run(on_update=on_update)
    # flicker: A -> B -> A within 2 transitions
    ids = [e[1] for e in events]
    flicker = sum(
        1 for i in range(2, len(ids))
        if ids[i] is not None and ids[i] == ids[i - 2] and ids[i] != ids[i - 1]
    )
    return events, flicker


def score_slide(expected_steps, trajectory_ids):
    """Ordered recall with alternates. Returns (hits, total, missed_steps)."""
    # dedup consecutive
    actual = []
    for x in trajectory_ids:
        if x is None:
            continue
        if not actual or actual[-1] != x:
            actual.append(x)
    ptr = 0
    hits = 0
    missed = []
    for step in expected_steps:
        found_at = None
        for j in range(ptr, len(actual)):
            if actual[j] in step:
                found_at = j
                break
        if found_at is not None:
            hits += 1
            ptr = found_at + 1
        else:
            missed.append(step)
    return hits, len(expected_steps), missed


def run(index_path, only=None, normalize=None):
    raw = json.loads(Path(index_path).read_text())
    manifest = json.loads((ROOT / "out" / "narration" / "manifest.json").read_text())
    gt = {s["slide"]: s["expected"] for s in json.loads(FIXTURE.read_text())["slides"]}

    print(f"loading models (BGE + Vosk)…  index={index_path}", flush=True)
    enc = load_default_encoder()
    idx = Index(Path(index_path), enc)
    matcher = Matcher(idx, enc)
    engine = VoskEngine(sample_rate=16000)

    tot_hits = tot_steps = tot_flicker = 0
    per_slide: dict[int, dict] = {}
    for item in manifest:
        sno = item["slide"]
        if only is not None and sno != only:
            continue
        events, flicker = record_trajectory(sno, item["wav"], raw, engine, matcher,
                                             normalize=normalize)
        traj = [e[1] for e in events]
        hits, steps, missed = score_slide(gt[sno], traj)
        tot_hits += hits
        tot_steps += steps
        tot_flicker += flicker
        per_slide[sno] = {"hits": hits, "steps": steps, "missed": missed,
                          "flicker": flicker, "trajectory": [x for x in traj if x]}
        flag = "" if hits == steps else f"   MISSED {missed}"
        print(f"slide {sno}: {hits}/{steps}  flicker={flicker}{flag}")
        print(f"          trajectory: {[x for x in traj]}")
    acc = tot_hits / tot_steps if tot_steps else 0.0
    print(f"\nOVERALL: {tot_hits}/{tot_steps} = {acc:.3f}   total_flicker={tot_flicker}")
    return {"slides": per_slide,
            "overall": {"hits": tot_hits, "steps": tot_steps, "flicker": tot_flicker}}


def audit(index_path, normalize_index=True):
    """Programmatic entry: returns the structured result dict (see run())."""
    normalize = None
    if normalize_index:
        from src.asr_normalize import build_alias_map, normalize as _norm
        raw = json.loads(Path(index_path).read_text())
        amap = build_alias_map(raw)
        normalize = (lambda t: _norm(t, amap)) if amap else None
    return run(index_path, normalize=normalize)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Grounding-accuracy audit.")
    ap.add_argument("--index", default="out/index.refined.json")
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--normalize", action="store_true",
                    help="apply src.asr_normalize to the STT tail before matching")
    args = ap.parse_args(argv)
    normalize = None
    if args.normalize:
        from src.asr_normalize import build_alias_map, normalize as _norm
        raw = json.loads(Path(args.index).read_text())
        amap = build_alias_map(raw)
        normalize = lambda t: _norm(t, amap)
    run(args.index, only=args.only, normalize=normalize)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
