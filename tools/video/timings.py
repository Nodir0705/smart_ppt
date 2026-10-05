"""Script words <- ASR word timestamps (one sentence at a time, seconds).

``match`` runs difflib over the ``norm``-ed sub-tokens of both sides (an ASR word
that normalises to n tokens splits its span evenly), so ASR "8" matches "eight"
and ASR "100%" matches "one hundred percent". A script word is matched when any
of its sub-tokens is; Task 4's coverage = matched / non-punctuation words.

``align`` fills the rest. Choices where the plan was open:
1. a run of n unmatched words between matched neighbours p and q shares the gap
   [min(p.end, q.start), q.start] evenly, each end = next start. Whistle stamps
   are contiguous, so a substituted word ("one" heard as "won") gets exactly the
   slot of the ASR word that replaced it; a dropped word gets a 0-length slot;
2. leading unmatched words: start = end = first matched start; trailing:
   start = end = last matched end;
3. pure-punctuation words (norm == "") copy the previous word's (start, end);
   at index 0 they follow rule 2;
4. no matched word at all -> every word at (0.0, 0.0) (coverage flags it);
5. a final running max keeps starts non-decreasing (and end >= start) even for
   overlapping or unsorted ASR input. Times are rounded to the millisecond."""
from __future__ import annotations

from difflib import SequenceMatcher

from .script import norm


def match(script_words: list[str], asr_words: list[dict]) -> list[tuple[float, float] | None]:
    a_tok, a_t = [], []
    for w in asr_words:
        subs = norm(w["word"]).split()
        step = (w["end"] - w["start"]) / max(len(subs), 1)
        for i, s in enumerate(subs):
            a_tok.append(s)
            a_t.append((round(w["start"] + i * step, 3), round(w["start"] + (i + 1) * step, 3)))
    s_tok, owner = [], []
    for k, w in enumerate(script_words):
        for s in norm(w).split():
            s_tok.append(s)
            owner.append(k)
    got: list[tuple[float, float] | None] = [None] * len(script_words)
    for b in SequenceMatcher(None, s_tok, a_tok, autojunk=False).get_matching_blocks():
        for i in range(b.size):
            k, (st, en) = owner[b.a + i], a_t[b.b + i]
            got[k] = (got[k][0], en) if got[k] else (st, en)
    return got


def align(script_words: list[str], asr_words: list[dict]) -> list[dict]:
    got = match(script_words, asr_words)
    hit = [k for k, t in enumerate(got) if t]
    if not hit:
        return [{"word": w, "start": 0.0, "end": 0.0} for w in script_words]
    first, last = got[hit[0]][0], got[hit[-1]][1]
    times = list(got)
    content = [k for k, w in enumerate(script_words) if norm(w)]
    i = 0
    while i < len(content):                        # runs of unmatched content words
        if times[content[i]]:
            i += 1
            continue
        j = i
        while j < len(content) and not times[content[j]]:
            j += 1
        run = content[i:j]
        if i == 0:
            fill = [(first, first)] * len(run)
        elif j == len(content):
            fill = [(last, last)] * len(run)
        else:
            hi = times[content[j]][0]
            lo = min(times[content[i - 1]][1], hi)
            st = [round(lo + n * (hi - lo) / len(run), 3) for n in range(len(run))] + [hi]
            fill = list(zip(st, st[1:]))
        for k, t in zip(run, fill):
            times[k] = t
        i = j
    out, prev = [], 0.0
    for k, w in enumerate(script_words):
        if times[k] is None:                       # pure punctuation
            times[k] = times[k - 1] if k else (first, first)
        prev = max(prev, times[k][0])
        out.append({"word": w, "start": prev, "end": max(times[k][1], prev)})
    return out
