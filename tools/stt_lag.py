"""Speech-recognition lag of the live pointer, per spoken word: how long after a
word is spoken does it first appear in the streaming STT's partials.

    python3 -m tools.stt_lag --deck AI_Handout_Checker

Inputs: out/bench/<deck>/asr_streams.json (sherpa partials on the audio clock,
``tools.clause_replay --record``), each style's presenter_script*.json and its
``<outdir>/manifest.json`` (sentence starts, ``tools.presenter_bench``).

Ground truth: each sentence of a slide wav (cut at the manifest starts) goes to
the Whistle server with ``"words": true``; ``timings.match`` keeps the script
words Whistle matched (no interpolation). Whistle stamps are contiguous (a
word's end = the next word's start), so a word before a pause ends inside the
pause: each end is trimmed back to the last voiced 10 ms frame in [start, end]
(threshold as ``clause_pointer.PauseDetector``: max(200, 5 % of the slide's p95
frame RMS)). Raw Whistle words are cached in out/bench/<deck>/word_truth.json,
keyed by wav (wav names carry a content hash).

Emission. A partial is cumulative within an utterance and restarts after a
final; an utterance's completed text is its final, else its last partial
(``SherpaEngine`` does not flush at end of stream).
1. the completed text of every utterance, in order, is aligned once to the
   slide's script tokens (difflib on ``norm`` sub-tokens);
2. each partial is aligned to its own utterance's completed text; a completed
   token is reached at the first partial that matches it at about its own
   position (|partial index - completed index| <= ``SLACK``, room for the few
   "to day" -> "today" re-splits), with full token equality: a fragment like
   "everyon" does not count;
3. a script word is emitted when the last of its matched sub-tokens is reached.
The position bound matters: with 350 ms sentence gaps sherpa's endpoint (1.2 s
trailing silence) rarely fires, so one utterance is often a whole slide, and an
unbounded match lets the fragment "work" (of "worksheet") at 3.7 s confirm the
"student work" said at 27 s.
Lag = emission - trimmed spoken end, for words with both.

Named late: steps as ``presenter_bench`` (``step_sentences`` over unit
segments), start = the introducing sentence's manifest start. Naming word = the
first word of that sentence whose crude stem occurs in the referred units'
element text and in no other unit's text on the slide (stopwords and tokens
under 3 letters out). Per step: said = its spoken start - sentence start;
heard = its emission - sentence start, the earliest a word-level decider could
fire on this STT. A sentence with no such word is "unnamed".

Writes out/bench/<deck>/stt_lag.json (word and step rows).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import wave
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from src.matcher import pointable_units
from src.stt import _GENERIC_STOPWORDS
from tools.clause_replay import SCRIPTS
from tools.presenter_bench import out_dir_name, step_sentences, unit_segments
from tools.video.script import norm, words
from tools.video.timings import match

FRAME_S = 0.01
SLACK = 3      # tokens a partial's word may sit from its place in the completed text


def toks(text: str) -> list[str]:
    return norm(text).split()


def subtokens(ws: list[str]) -> tuple[list[str], list[int]]:
    """Words -> their ``norm`` sub-tokens and each sub-token's word index."""
    tok, owner = [], []
    for k, w in enumerate(ws):
        for s in toks(w):
            tok.append(s)
            owner.append(k)
    return tok, owner


def utterances(partials: list[dict]) -> list[list[tuple[float, list[str]]]]:
    """Partials -> utterances [(t, tokens)]; a final closes one."""
    out, cur = [], []
    for p in partials:
        cur.append((p["t"], toks(p["text"])))
        if p["final"]:
            out.append(cur)
            cur = []
    return out + [cur] if cur else out


def _blocks(a: list[str], b: list[str]):
    for m in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        yield from ((m.a + i, m.b + i) for i in range(m.size))


def emission(script_words: list[str], partials: list[dict]) -> list[float | None]:
    """Per script word: audio time of the first partial that shows it (None = never)."""
    hyp, reached = [], []
    for u in utterances(partials):
        done = u[-1][1]
        first: list[float | None] = [None] * len(done)
        for t, tk in u:
            for i, j in _blocks(tk, done):
                if abs(i - j) <= SLACK and first[j] is None:
                    first[j] = t
        hyp += done
        reached += first
    s_tok, owner = subtokens(script_words)
    out: list[float | None] = [None] * len(script_words)
    for i, j in _blocks(s_tok, hyp):
        k = owner[i]
        out[k] = reached[j] if out[k] is None else max(out[k], reached[j])
    return out


def voiced(pcm: bytes) -> np.ndarray:
    """Per 10 ms frame of 16 kHz PCM16: above the PauseDetector-style threshold."""
    x = np.frombuffer(pcm, np.int16).astype(np.float32)
    hop = int(16000 * FRAME_S)
    n = len(x) // hop
    rms = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1))
    return rms >= max(200.0, 0.05 * float(np.percentile(rms, 95)))


def trim(start: float, end: float, v: np.ndarray) -> float:
    """End moved back to the last voiced frame in [start, end] (unchanged if none)."""
    a, b = int(start / FRAME_S + 1e-6), min(int(round(end / FRAME_S)), len(v))
    on = np.flatnonzero(v[a:b])
    return round((a + on[-1] + 1) * FRAME_S, 3) if len(on) else end


def stem(t: str) -> str:
    """Crude suffix strip: grade/graded/grading/gradable -> grad, page/pages -> pag."""
    if t.endswith("ss"):
        return t
    for suf in ("able", "ing", "ed", "es", "e", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            return t[:-len(suf)]
    return t


def content_stems(text: str) -> set[str]:
    return {stem(t) for t in toks(text) if len(t) >= 3 and t not in _GENERIC_STOPWORDS}


def naming_word(sentence: list[str], refs: list[str], unit_text: dict[str, str]) -> int | None:
    """Index of the first word whose stem is in the referred units' text and no other's."""
    own = set().union(*(content_stems(unit_text.get(u, "")) for u in refs))
    other = set().union(*(content_stems(t) for u, t in unit_text.items() if u not in refs))
    keep = own - other
    return next((j for j, w in enumerate(sentence) if content_stems(w) & keep), None)


def pct(xs: list[float], q: float) -> float:
    """Nearest-rank percentile (presenter_bench.p90's rule)."""
    return sorted(xs)[math.ceil(q * len(xs)) - 1] if xs else float("nan")


def whistle_truth(deck_dir: Path, jobs: list[tuple[str, list[float]]]) -> dict:
    """wav -> per sentence the raw Whistle words, cached in word_truth.json."""
    path = deck_dir / "word_truth.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo = [(w, s) for w, s in dict(jobs).items() if w not in cache]
    if todo:
        from src.stt import WhistleEngine
        from tools.video.voice import whistle_words
        eng = WhistleEngine()
        try:
            for k, (wav, starts) in enumerate(todo, 1):
                with wave.open(wav, "rb") as w:
                    assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), wav
                    pcm = w.readframes(w.getnframes())
                cut = [round(s * 16000) * 2 for s in starts] + [len(pcm)]
                cache[wav] = [whistle_words(eng, pcm[a:b]) for a, b in zip(cut, cut[1:])]
                path.write_text(json.dumps(cache))
                print(f"whistle {k}/{len(todo)} {Path(wav).name}", flush=True)
        finally:
            eng.close()
    return cache


def slide_rows(style: str, n: int, segs: list[dict], starts: list[float], wav: str,
               heard: list[list[dict]], partials: list[dict], unit_of: dict,
               unit_text: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Word rows, step rows and per-sentence energy checks for one slide."""
    with wave.open(wav, "rb") as w:
        v = voiced(w.readframes(w.getnframes()))
    flat, rows, checks = [], [], []
    for k, seg in enumerate(segs):
        ws = words(seg["text"])
        got = match(ws, heard[k])
        for j, (w, g) in enumerate(zip(ws, got)):
            flat.append(w)
            r = {"style": style, "slide": n, "sentence": k, "word_idx": j, "word": w,
                 "content": bool(norm(w)), "start": None, "end": None, "end_raw": None}
            if g:
                st, en = round(g[0] + starts[k], 3), round(g[1] + starts[k], 3)
                r.update(start=st, end_raw=en, end=trim(st, en, v))
            rows.append(r)
        # energy check: sentence onset / offset vs its first / last matched word
        a = int(starts[k] / FRAME_S + 1e-6)
        b = int(starts[k + 1] / FRAME_S + 1e-6) if k + 1 < len(starts) else len(v)
        on = np.flatnonzero(v[a:b])
        mine = [r for r in rows if r["sentence"] == k and r["content"]]
        if len(on) and mine:
            first, last = mine[0]["start"], mine[-1]["end_raw"]
            checks.append({"onset": None if first is None else first - (a + on[0]) * FRAME_S,
                           "offset_raw": None if last is None else last - (a + on[-1] + 1) * FRAME_S})
    for r, t in zip(rows, emission(flat, partials)):
        r["emit"] = t
        r["lag"] = round(t - r["end"], 3) if t is not None and r["end"] is not None else None
    steps = []
    for refs, k in step_sentences(unit_segments(segs, unit_of)):
        ws = [r for r in rows if r["sentence"] == k]
        j = naming_word([r["word"] for r in ws], refs, unit_text)
        s = {"style": style, "slide": n, "sentence": k, "refers_to": refs, "text": segs[k]["text"],
             "name": None, "said": None, "heard": None,
             "sentence_s": round((starts[k + 1] if k + 1 < len(starts) else len(v) * FRAME_S) - starts[k], 3)}
        if j is not None:
            r = ws[j]
            s.update(name=r["word"],
                     said=round(r["start"] - starts[k], 3) if r["start"] is not None else None,
                     heard=round(r["emit"] - starts[k], 3) if r["emit"] is not None else None)
        steps.append(s)
    return rows, steps, checks


def unit_texts(slide: dict, unit_of: dict) -> dict[str, str]:
    """unit id (as unit_segments maps refers_to) -> its elements' text."""
    out: dict[str, str] = {}
    for e in slide["elements"]:
        u = unit_of.get(e["id"], e["id"])
        out[u] = f"{out.get(u, '')} {e.get('text') or ''}"
    return out


def stats(xs: list[float]) -> str:
    if not xs:
        return "– | – | – |"
    return f"{statistics.median(xs):.2f} | {pct(xs, 0.75):.2f} | {pct(xs, 0.9):.2f} |"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bench", default="out/bench")
    ap.add_argument("--deck", default="AI_Handout_Checker")
    ap.add_argument("--variant", default="map", help="index dir for unit texts (named-late)")
    args = ap.parse_args(argv)

    deck_dir = Path(args.bench) / args.deck
    asr = json.loads((deck_dir / "asr_streams.json").read_text())
    raw = json.loads((deck_dir / args.variant / "index.scoped.json").read_text())
    unit_of = pointable_units(raw)
    texts = {s["slide_number"]: unit_texts(s, unit_of) for s in raw["slides"]}
    plan = {}
    for style in asr:
        script = SCRIPTS[style]
        segs = {s["slide"]: s["segments"] for s in json.loads((deck_dir / script).read_text())["slides"]}
        man = {m["slide"]: m
               for m in json.loads((deck_dir / out_dir_name(script) / "manifest.json").read_text())}
        plan[style] = (segs, man)
    truth = whistle_truth(deck_dir, [(plan[st][1][int(n)]["wav"], plan[st][1][int(n)]["starts"])
                                     for st in asr for n in asr[st]])
    rows, steps, checks = [], [], []
    for style, (segs, man) in plan.items():
        for n_str, x in asr[style].items():
            n, m = int(n_str), man[int(n_str)]
            assert x["wav"] == m["wav"], f"asr_streams.json is stale for {style} slide {n}"
            assert len(m["starts"]) == len(segs[n]), f"{style} slide {n}: starts != sentences"
            r, s, c = slide_rows(style, n, segs[n], m["starts"], m["wav"], truth[m["wav"]],
                                 x["partials"], unit_of, texts.get(n, {}))
            rows, steps, checks = rows + r, steps + s, checks + c
    (deck_dir / "stt_lag.json").write_text(json.dumps({"words": rows, "steps": steps}, indent=1))

    styles = list(plan) + ["all"]
    pick = lambda xs, st: [x for x in xs if st == "all" or x["style"] == st]  # noqa: E731
    print(f"## STT lag — {args.deck} (emission − spoken end, audio clock, s)\n")
    print("| script | words | with truth | emitted | both | lag med | p75 | p90 | mean | < −0.1 s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for st in styles:
        ws = [r for r in pick(rows, st) if r["content"]]
        lag = [r["lag"] for r in ws if r["lag"] is not None]
        print(f"| {st} | {len(ws)} | {sum(r['end'] is not None for r in ws)} "
              f"| {sum(r['emit'] is not None for r in ws)} | {len(lag)} | {stats(lag)} "
              f"{statistics.mean(lag):.2f} | {sum(x < -0.1 for x in lag)} |")
    on = [c["onset"] for c in checks if c["onset"] is not None]
    off = [c["offset_raw"] for c in checks if c["offset_raw"] is not None]
    trims = [r["end_raw"] - r["end"] for r in rows if r["end"] is not None]
    print(f"\nGround truth: first word start − energy onset med {statistics.median(on):+.3f} s "
          f"(|·| p90 {pct([abs(x) for x in on], 0.9):.3f}, n={len(on)}); raw last word end − energy "
          f"offset med {statistics.median(off):+.3f} s (n={len(off)}); end trim med "
          f"{statistics.median(trims):.3f} s, mean {statistics.mean(trims):.3f} s, "
          f"{sum(x > 0.05 for x in trims)}/{len(trims)} trimmed > 50 ms\n")
    print("## Named late — expected steps (s from the introducing sentence's start)\n")
    print("| script | steps | named | no truth/emit | said med | p75 | p90 | heard med | p75 | p90 "
          "| sentence med s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for st in styles:
        ss = pick(steps, st)
        named = [s for s in ss if s["name"]]
        said = [s["said"] for s in named if s["said"] is not None]
        heard = [s["heard"] for s in named if s["heard"] is not None]
        print(f"| {st} | {len(ss)} | {len(named)} | {len(named) - len(heard)} | {stats(said)} "
              f"{stats(heard)} {statistics.median(s['sentence_s'] for s in ss):.2f} |")
    print(f"\nrows -> {deck_dir / 'stt_lag.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
