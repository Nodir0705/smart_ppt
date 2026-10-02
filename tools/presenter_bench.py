"""Score live grounding on a hand-written presenter talk track — the realistic test.

`out/bench/<deck>/presenter_script.json` is natural presenter speech (paraphrase,
explanation, transitions) with each sentence annotated by the element ids a human
would point at while saying it (`refers_to`, `[]` for non-pointing speech). Unlike
the verbatim auto-narration (tools/narrate.py), only referenced elements are
expected, so the score measures "lights the right thing when you talk about it,
stays quiet otherwise".

    python3 -m tools.presenter_bench                 # every deck that has a script
    python3 -m tools.presenter_bench --only test_deck
Writes out/bench/<deck>/presenter/{narration,expected,manifest,trace}.json and
out/bench/report_presenter.md.

Beyond lit/false/flicker it measures how the highlight MOVES, on the audio clock:
switches per expected step (1.0 = one jump per thing talked about), median dwell
between switches, false seconds (time spent on an element nobody referred to) and
lateness (first lit minus the start of the sentence that introduced it).

When `presenter/adjudication.json` exists (a human verdict per false switch:
tolerated / wrong / wrong_timing), false ids and false seconds are ALSO reported
adjudicated: per slide, acceptable = every `refers_to` id on the slide plus every
id adjudicated `tolerated` on that slide. Strict numbers stay the primary ruler.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import statistics
import wave
from pathlib import Path

from src.encoder import eval_device
from src.tts import synthesize_slides
from tools.deck_bench import VOICE, _write, load_models, per_deck_processes, totals
from tools.timeline_audit import audit

PAUSE_MS = 350  # sentence gap: a natural pause, not the 900 ms element gap of the ruler


def step_sentences(segments: list[dict]) -> list[tuple[list[str], int]]:
    """Ordered expected highlights with the index of the sentence introducing
    each: every referencing sentence is one step; consecutive sentences about
    the same element(s) collapse into one."""
    steps: list[tuple[list[str], int]] = []
    for k, seg in enumerate(segments):
        ids = list(seg.get("refers_to") or [])
        if ids and (not steps or set(steps[-1][0]) != set(ids)):
            steps.append((ids, k))
    return steps


def expected_steps(segments: list[dict]) -> list[list[str]]:
    return [ids for ids, _ in step_sentences(segments)]


def false_seconds(events: list, ok: set, end: float) -> float:
    """Seconds an id outside `ok` is lit (None is free); events as in slide_motion."""
    return sum(max(0.0, (events[k + 1][0] if k + 1 < len(events) else end) - t)
               for k, (t, a, *_) in enumerate(events) if a is not None and a not in ok)


def false_ids(events: list, ok: set) -> int:
    """Distinct lit ids outside `ok`."""
    return len({a for _t, a, *_ in events if a is not None and a not in ok})


def tolerated_ids(deck_dir: Path) -> dict[int, set[str]] | None:
    """slide -> ids a human adjudicated `tolerated`; None without adjudication.json."""
    p = deck_dir / "presenter" / "adjudication.json"
    if not p.exists():
        return None
    out: dict[int, set[str]] = {}
    for r in json.loads(p.read_text())["rows"]:
        if r["verdict"] == "tolerated":
            out.setdefault(r["slide"], set()).add(r["active_id"])
    return out


def slide_motion(events: list, steps: list[list[str]], step_starts: list[float],
                 end: float) -> dict:
    """Motion metrics for one slide from `events` [(t, active_id, tail)] (audio s,
    one per change of active id; None = nothing lit) over a slide `end` s long.

    switches: changes to a non-None id. dwells: seconds between consecutive such
    switches. false_s: seconds the active id is outside every step (None is
    free). lateness: per hit step (ordered, as timeline_audit.score_slide walks
    them), first time lit minus `step_starts` of its sentence.
    """
    ok = set().union(*steps) if steps else set()
    on = [(t, a) for t, a, *_ in events if a is not None]
    false_s = false_seconds(events, ok, end)
    runs = [(t, a) for k, (t, a) in enumerate(on) if k == 0 or on[k - 1][1] != a]
    late, ptr = [], 0
    for ids, start in zip(steps, step_starts):
        j = next((j for j in range(ptr, len(runs)) if runs[j][1] in ids), None)
        if j is not None:
            late.append(runs[j][0] - start)
            ptr = j + 1
    return {"switches": len(on), "dwells": [b[0] - a[0] for a, b in zip(on, on[1:])],
            "false_s": false_s, "lateness": late}


def p90(xs: list[float]) -> float:
    """Nearest-rank 90th percentile."""
    return sorted(xs)[math.ceil(0.9 * len(xs)) - 1] if xs else float("nan")


def motion_summary(per_slide: list[dict], n_steps: int) -> dict:
    dwells = [d for m in per_slide for d in m["dwells"]]
    late = [x for m in per_slide for x in m["lateness"]]
    adj = all("false_s_adj" in m for m in per_slide)
    return {"switches_per_step": sum(m["switches"] for m in per_slide) / n_steps if n_steps else 0.0,
            "median_dwell": statistics.median(dwells) if dwells else float("nan"),
            "false_s": sum(m["false_s"] for m in per_slide),
            "false_s_adj": sum(m["false_s_adj"] for m in per_slide) if adj else None,
            "false_adj": sum(m["false_adj"] for m in per_slide) if adj else None,
            "late_median": statistics.median(late) if late else float("nan"),
            "late_p90": p90(late)}


def sentence_at(starts: list[float], t: float) -> int:
    """Index of the sentence being spoken at audio time t."""
    return max([k for k, s in enumerate(starts) if s <= t] or [0])


def _wav_seconds(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


def prepare(deck_dir: Path) -> Path:
    """Script -> narration/expected/audio under deck_dir/presenter; returns that dir."""
    script = json.loads((deck_dir / "presenter_script.json").read_text())
    d = deck_dir / "presenter"
    slides = [{"slide": s["slide"], "segments": [seg["text"] for seg in s["segments"]]}
              for s in script["slides"]]
    _write(d / "narration.json", {"voice": str(VOICE), "gap_ms": PAUSE_MS, "slides": slides})
    _write(d / "expected.json", {"slides": [
        {"slide": s["slide"], "expected": expected_steps(s["segments"])} for s in script["slides"]]})
    _write(d / "manifest.json", synthesize_slides(slides, VOICE, d / "audio", PAUSE_MS,
                                                  need_starts=True))
    return d


def score_motion(deck_dir: Path, events: dict) -> tuple[dict, list[dict]]:
    """Deck motion summary and the per-switch trace (for adjudication)."""
    tolerated = tolerated_ids(deck_dir)
    script = {s["slide"]: s["segments"]
              for s in json.loads((deck_dir / "presenter_script.json").read_text())["slides"]}
    manifest = {m["slide"]: m for m in
                json.loads((deck_dir / "presenter" / "manifest.json").read_text())}
    index = json.loads((deck_dir / "default" / "index.scoped.json").read_text())
    text = {e["id"]: e.get("text") for s in index["slides"] for e in s["elements"]}
    per_slide, trace, n_steps = [], [], 0
    for sno, evs in sorted(events.items()):
        segs, starts = script[sno], manifest[sno]["starts"]
        steps = step_sentences(segs)
        ok = set().union(*(ids for ids, _ in steps)) if steps else set()
        n_steps += len(steps)
        end = _wav_seconds(manifest[sno]["wav"])
        m = slide_motion(evs, [ids for ids, _ in steps], [starts[k] for _, k in steps], end)
        if tolerated is not None:
            adj_ok = {i for seg in segs for i in seg.get("refers_to") or []} | tolerated.get(sno, set())
            m.update(false_s_adj=false_seconds(evs, adj_ok, end), false_adj=false_ids(evs, adj_ok))
        per_slide.append(m)
        for t, a, tail in evs:
            k = sentence_at(starts, t)
            trace.append({"slide": sno, "t": round(t, 2), "active_id": a,
                          "active_text": text.get(a) if a else None, "tail": tail,
                          "ok": (a in ok) if a else None, "sentence": segs[k]["text"],
                          "sentence_refers_to": segs[k].get("refers_to") or []})
    return motion_summary(per_slide, n_steps), trace


def bench_deck(deck: Path, device: str) -> tuple[dict, str]:
    """One deck, own process + models: prepare, replay, score; writes trace.json."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        d = prepare(deck)
        res = audit(deck / "default" / "index.scoped.json", manifest_path=d / "manifest.json",
                    fixture_path=d / "expected.json", fast=True, models=load_models(device))
        o = res["overall"]
        motion, trace = score_motion(deck, res["events"])
        _write(d / "trace.json", trace)
    return {"deck": deck.name, "status": "OK", **motion, **{k: o[k] for k in
            ("hits", "steps", "relevant", "false", "flicker", "precision")}}, buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Grounding score on annotated presenter scripts.")
    ap.add_argument("--bench", default="out/bench")
    ap.add_argument("--only", default=None, help="deck folder name")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                    help="text-encoder device (auto = cuda if available); STT stays CPU")
    args = ap.parse_args(argv)

    decks = sorted(p.parent for p in Path(args.bench).glob("*/presenter_script.json"))
    if args.only:
        decks = [d for d in decks if d.name == args.only]
    device = eval_device(args.device)
    records = []
    for deck, (rec, log) in zip(decks, per_deck_processes(bench_deck, [(d, device) for d in decks])):
        print(f"=== {deck.name} ===\n{log}", end="", flush=True)
        records.append(rec)
    tot = totals(records)
    adj = lambda v, f: "–" if v is None else format(v, f)  # noqa: E731
    lines = ["# Presenter-script bench (natural speech, annotated pointing)", "",
             "Adjudicated (adj) = `refers_to` ids plus ids a human judged *tolerated* on that slide.", "",
             "| deck | referenced elements lit (in order) | false highlights | false adj | precision "
             "| flicker | switches/step | median dwell s | false s | false s adj | late median s "
             "| late p90 s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['deck']} | {r['hits']}/{r['steps']} | {r['false']} | {adj(r['false_adj'], 'd')} "
              f"| {r['precision']:.3f} | {r['flicker']} | {r['switches_per_step']:.2f} "
              f"| {r['median_dwell']:.2f} | {r['false_s']:.1f} | {adj(r['false_s_adj'], '.1f')} "
              f"| {r['late_median']:.2f} | {r['late_p90']:.2f} |"
              for r in records]
    lines += ["", f"**Total:** {tot['hits']}/{tot['steps']} lit · false {tot['false']} · "
              f"precision {tot['precision']:.3f} · flicker {tot['flicker']} · "
              f"false {sum(r['false_s'] for r in records):.1f} s", ""]
    md = "\n".join(lines)
    (Path(args.bench) / "report_presenter.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
