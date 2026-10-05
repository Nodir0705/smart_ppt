"""Replay recorded STT streams through the live decision path and score them with
the presenter_bench ruler — seconds per run instead of a full bench.

STT output does not depend on the matcher, so the partials of every presenter
script are recorded once (``--record``: out/bench/<deck>/asr_streams.json) and
replayed here through the real ``LiveController``: each 100 ms WAV chunk goes to
the buffer's pause detector, then the partials STT emitted on that chunk are
pushed and fed — the order ``LiveTranscriber`` uses live.

    python3 -m tools.clause_replay --record                       # once (CPU STT)
    python3 -m tools.clause_replay --pointer tail                 # today's map system
    python3 -m tools.clause_replay                                # clause pointer, shipped decider + params
    python3 -m tools.clause_replay --decider bi --commit 0.1      # another decider / params
    python3 -m tools.clause_replay --oracle oracle.json           # score clause decisions
Tune scripts by default; ``--holdout`` adds the meaning script.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.audio import FileSource
from src.clause_pointer import ClauseParams
from src.live_controller import LiveController
from src.matcher import Index, Matcher, pointable_units
from src.stt import Partial
from tools.presenter_bench import out_dir_name, score_motion
from tools.timeline_audit import score_slide

SCRIPTS = {"original": "presenter_script.json", "jumpy": "presenter_script_jumpy.json",
           "meaning": "presenter_script_meaning.json",
           "casual": "presenter_script_casual.json"}   # sealed: scored only via --styles casual
TUNE, HOLDOUT = ("original", "jumpy"), ("meaning",)
CHUNK_S = 0.1


def record(deck_dir: Path, out: Path) -> None:
    """Every script's STT partials on the audio clock -> out (style -> slide -> {wav, partials})."""
    from src.stt import load_default_stt_engine
    eng = load_default_stt_engine(16000, audio_clock=True)
    data = {}
    for style in TUNE + HOLDOUT:
        script = SCRIPTS[style]
        man = json.loads((deck_dir / out_dir_name(script) / "manifest.json").read_text())
        data[style] = {str(m["slide"]): {"wav": m["wav"], "partials": [
            {"t": round(p.timestamp, 2), "final": p.is_final, "text": p.text}
            for p in eng.transcribe_stream(FileSource(m["wav"], chunk_ms=100).chunks())]} for m in man}
        print(style, "recorded", flush=True)
    out.write_text(json.dumps(data))


def replay_slide(ctl: LiveController, wav: str, partials: list[dict]) -> list[tuple]:
    """Events [(t, active_id, tail)] on change, as timeline_audit records them."""
    by_chunk: dict[int, list[dict]] = {}
    for p in partials:
        by_chunk.setdefault(round(p["t"] / CHUNK_S), []).append(p)
    src = FileSource(wav, chunk_ms=100)
    events, last = [], "___init___"
    for i, chunk in enumerate(src.chunks(), 1):
        ctl.buffer.audio(chunk, src.sample_rate)
        for p in by_chunk.get(i, []):
            ctl.buffer.push(Partial(p["text"], p["final"], p["t"]))
            ev = ctl.feed()
            if ev is not None and ev.active_id != last:
                events.append((p["t"], ev.active_id, ev.tail))
                last = ev.active_id
    return events


def score(deck_dir: Path, variant: str, style: str, events: dict, unit_of: dict) -> dict:
    script = SCRIPTS[style]
    expected = {s["slide"]: s["expected"] for s in json.loads(
        (deck_dir / out_dir_name(script) / "expected.json").read_text())["slides"]}
    hits = steps = false = 0
    for n, ev in events.items():
        h, tot, _m, _r, f = score_slide(expected.get(n, []), [a for _t, a, *_ in ev], unit_of)
        hits, steps, false = hits + h, steps + tot, false + f
    m, _trace = score_motion(deck_dir, events, variant, script)
    return {"style": style, "lit": hits, "steps": steps, "false": false, "sw": m["switches_per_step"],
            "dwell": m["median_dwell"], "false_s": m["false_s"], "late": m["late_median"],
            "late_p90": m["late_p90"]}


def row(name: str, r: dict) -> str:
    return (f"| {name} | {r['style']} | {r['lit']}/{r['steps']} | {r['false']} | {r['sw']:.2f} "
            f"| {r['dwell']:.2f} | {r['false_s']:.1f} | {r['late']:.2f} | {r['late_p90']:.2f} |")


HEADER = ("| run | script | lit | false ids | sw/step | median dwell s | false s | late med s | late p90 s |\n"
          "|---|---|---|---|---|---|---|---|---|")


def oracle_events(oracle: dict, asr: dict, style: str, raw: dict, unit_of: dict) -> dict:
    """Clause decisions (first_word, unit, commit_word) -> events at the commit word's emission time."""
    maps = {s["slide_number"]: s["map"]["units"] for s in raw["slides"] if s.get("map")}
    out = {}
    for sl in oracle["slides"]:
        n = sl["slide"]
        partials = asr[style][str(n)]["partials"]
        cl = next((s["clauses"] for s in sl["scripts"] if s["style"] == style), [])
        ev, cur = [], None
        for c in sorted(cl, key=lambda c: c["first_word"]):
            if not c["unit"] or c["commit_word"] is None:
                continue
            u = unit_of.get(maps[n][c["unit"]][0])
            if u and u != cur:
                # first partial with more than commit_word words
                t = next((p["t"] for p in partials if len(p["text"].split()) > c["commit_word"]),
                         partials[-1]["t"] if partials else 0.0)
                ev.append((t, u, ""))
                cur = u
        out[n] = ev
    return out


def make_decider(name: str, matcher: Matcher, device: str):
    if name == "bi":
        from src.deciders.bi import BiDecider
        return BiDecider(matcher.encoder)
    if name.startswith("cross"):
        from src.deciders.cross import CrossDecider
        model = name.split(":", 1)[1] if ":" in name else None
        return CrossDecider(model, device=device)
    if name.startswith("llm"):   # llm[:<gguf path>]; --device cuda offloads every layer
        from src.deciders.llm import LLMDecider    # (set SMARTPPT_LLM_PYTHON to the cuda venv)
        model = name.split(":", 1)[1] if ":" in name else None
        return LLMDecider(model, gpu_layers=99 if device == "cuda" else 0)
    raise ValueError(name)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bench", default="out/bench")
    ap.add_argument("--deck", default="AI_Handout_Checker")
    ap.add_argument("--variant", default="map")
    ap.add_argument("--asr", default=None, help="recorded streams (default out/bench/<deck>/asr_streams.json)")
    ap.add_argument("--record", action="store_true", help="record the STT streams first")
    ap.add_argument("--holdout", action="store_true", help="also score the held-out meaning script")
    ap.add_argument("--styles", default=None, help="comma list overriding the script set")
    ap.add_argument("--pointer", default="clause", choices=("clause", "tail"))
    ap.add_argument("--decider", default="cross", help="bi | cross[:<hf model>] | llm[:<gguf>]")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--oracle", default=None, help="score clause decisions from this oracle.json instead")
    ap.add_argument("--name", default=None, help="row label")
    for f in ClauseParams.__dataclass_fields__.values():
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    args = ap.parse_args(argv)

    deck_dir = Path(args.bench) / args.deck
    asr_path = Path(args.asr) if args.asr else deck_dir / "asr_streams.json"
    if args.record:
        record(deck_dir, asr_path)
    asr = json.loads(asr_path.read_text())
    styles = args.styles.split(",") if args.styles else list(TUNE + (HOLDOUT if args.holdout else ()))
    index_path = deck_dir / args.variant / "index.scoped.json"
    raw = json.loads(index_path.read_text())
    unit_of = pointable_units(raw)
    print(HEADER)
    if args.oracle:
        oracle = json.loads(Path(args.oracle).read_text())
        for style in styles:
            r = score(deck_dir, args.variant, style, oracle_events(oracle, asr, style, raw, unit_of), unit_of)
            print(row(args.name or "oracle", r))
        return 0

    from src.encoder import load_default_encoder
    enc = load_default_encoder(args.device)
    idx = Index(index_path, enc)
    matcher = Matcher(idx, enc)
    params = ClauseParams(**{k: getattr(args, k) for k in ClauseParams.__dataclass_fields__})
    decider = make_decider(args.decider, matcher, args.device) if args.pointer == "clause" else None
    name = args.name or (args.pointer if args.pointer == "tail" else args.decider)
    calls = []
    if decider is not None:   # per-call wall time, for the latency report
        inner = decider.decide

        def timed(*a):
            t0 = time.perf_counter()
            out = inner(*a)
            calls.append(time.perf_counter() - t0)
            return out
        decider.decide = timed
    for style in styles:
        events = {}
        for n in sorted(int(k) for k in asr[style]):
            ctl = LiveController(matcher, raw, slide_no=n, pointer=args.pointer, decider=decider,
                                 clause_params=params)
            x = asr[style][str(n)]
            events[n] = replay_slide(ctl, x["wav"], x["partials"])
        print(row(name, score(deck_dir, args.variant, style, events, unit_of)), flush=True)
    if calls:
        calls.sort()
        print(f"decider calls {len(calls)}  median {1000 * calls[len(calls) // 2]:.1f} ms  "
              f"p90 {1000 * calls[int(0.9 * len(calls))]:.1f} ms  ({args.device})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
