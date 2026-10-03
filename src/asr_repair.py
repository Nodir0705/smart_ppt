"""Sound-based ASR repair against one slide's own vocabulary.

STT spells a slide term the way it sounds: "colonel" for kernel, "knowed" for
node, "h l" for HAL, "elly dees" for LEDs, "esa memu" for SMMU, "kor" for core.
Edit distance on letters can't see those; pronunciations can. Per slide, every
vocabulary word of the map (unit text, terms incl. spelled forms, talk, summary)
gets its phonemes (CMU dict, letter names for acronyms and spelled letters, a
rule-based fallback for words the dict lacks). A span of 1-3 clause tokens is
rewritten to a vocabulary word when

* the span is all dictionary words: its phonemes EQUAL the word's (a homophone,
  ``colonel`` -> kernel), so an ordinary correctly heard word is never pulled to a
  merely similar slide word ("for" stays, "core" needs "kor");
* the span holds a non-dictionary token or a spelled letter: the weighted
  phoneme edit distance (vowel/vowel and voiced/unvoiced swaps cost 0.5),
  divided by the longer length, is <= ``threshold``; a span with a spelled
  letter only matches words spelled out as letters (acronyms: "sea p u" -> cpu,
  never "u sees" -> uses).

Tokens that are already slide words, function words, position words, numbers
and contractions never enter a span; a tie between two different words leaves
the span alone. Deterministic, CPU, no model; spans are memoized per slide, so
an update costs only its new tokens.
"""
from __future__ import annotations

import functools
import gzip
import itertools
import re
from pathlib import Path

DICT_PATH = Path(__file__).resolve().parent.parent / "data" / "cmudict.txt.gz"

LETTERS = {
    "a": "EY", "b": "B IY", "c": "S IY", "d": "D IY", "e": "IY", "f": "EH F", "g": "JH IY",
    "h": "EY CH", "i": "AY", "j": "JH EY", "k": "K EY", "l": "EH L", "m": "EH M", "n": "EH N",
    "o": "OW", "p": "P IY", "q": "K Y UW", "r": "AA R", "s": "EH S", "t": "T IY", "u": "Y UW",
    "v": "V IY", "w": "D AH B AH L Y UW", "x": "EH K S", "y": "W AY", "z": "Z IY",
}
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
VOICING = {frozenset(p) for p in (("P", "B"), ("T", "D"), ("K", "G"), ("F", "V"), ("S", "Z"),
                                  ("CH", "JH"), ("TH", "DH"), ("SH", "ZH"))}
STOP = set("""a about above after again against all also am an and any are as at be because been before
being below between both but by can could did do does doing down during each few for from further had has
have having he her here hers him his how i if in into is it its itself just me more most my no nor not now
of off on once only or other our ours out over own same she should so some such than that the their them
then there these they this those through to too under until up very was we were what when where which while
who whom why will with would you your yours yeah okay ok oh uh um let lets get got going gonna one two
three four five six seven eight nine ten hundred thousand first second next last like well really thing
things way see look right left top bottom middle side""".split())
POSITION = {"left", "right", "top", "bottom", "middle", "side", "upper", "lower", "centre", "center"}

# letter-to-sound fallback, longest grapheme first; (grapheme, phonemes) — "" is silent
_G2P = [
    ("tion", "SH AH N"), ("sion", "ZH AH N"), ("ough", "AO"), ("eigh", "EY"), ("tch", "CH"), ("igh", "AY"),
    ("ch", "CH"), ("sh", "SH"), ("th", "TH"), ("ph", "F"), ("ck", "K"), ("ng", "NG"), ("qu", "K W"),
    ("wh", "W"), ("gh", "G"), ("ee", "IY"), ("ea", "IY"), ("oo", "UW"), ("ou", "AW"), ("ow", "OW"),
    ("oi", "OY"), ("oy", "OY"), ("ai", "EY"), ("ay", "EY"), ("au", "AO"), ("aw", "AO"), ("ie", "IY"),
    ("ei", "EY"), ("ue", "UW"), ("ew", "UW"), ("oa", "OW"), ("ar", "AA R"), ("or", "AO R"), ("er", "ER"),
    ("ir", "ER"), ("ur", "ER"), ("a", "AE"), ("e", "EH"), ("i", "IH"), ("o", "AA"), ("u", "AH"),
    ("b", "B"), ("d", "D"), ("f", "F"), ("h", "HH"), ("j", "JH"), ("k", "K"), ("l", "L"), ("m", "M"),
    ("n", "N"), ("p", "P"), ("q", "K"), ("r", "R"), ("s", "S"), ("t", "T"), ("v", "V"), ("w", "W"),
    ("x", "K S"), ("z", "Z"),
]
_ONE = {g: ph for g, ph in _G2P if len(g) == 1}


@functools.lru_cache(maxsize=1)
def cmudict() -> dict[str, tuple[str, ...]]:
    """word -> its pronunciations (space-joined phonemes, no stress)."""
    out = {}
    for line in gzip.decompress(DICT_PATH.read_bytes()).decode().splitlines():
        if line and not line.startswith("#"):
            w, prons = line.split(" ", 1)
            out[w] = tuple(prons.split("|"))
    return out


def g2p(word: str) -> str:
    """Deterministic rule-based phonemes for a word the dictionary lacks."""
    w = re.sub(r"(.)\1", r"\1", word.lower())             # doubled letters sound once
    if w.startswith(("kn", "wr")):
        w = w[1:]
    out, i = [], 0
    while i < len(w):
        ch, nxt, last = w[i], w[i + 1:i + 2], i == len(w) - 1
        g = next((g for g, _ in _G2P if len(g) > 1 and w.startswith(g, i)), None)
        if g:
            out.extend(dict(_G2P)[g].split()); i += len(g); continue
        if ch == "e" and last and len(w) > 2 and out and out[-1] not in VOWELS:
            pass                                             # silent final e
        elif ch == "c":
            out.append("S" if nxt in ("e", "i", "y") else "K")
        elif ch == "g":
            out.append("JH" if i and nxt in ("e", "i", "y") else "G")
        elif ch == "y":
            out.append("Y" if i == 0 else "IY" if last else "IH")
        elif ch == "u" and last:
            out.append("UW")
        elif ch in _ONE:
            out.extend(_ONE[ch].split())
        i += 1
    return " ".join(out)


def pron_options(token: str) -> tuple[tuple[str, ...], bool]:
    """(pronunciations of one clause token, in the dictionary as a word?)"""
    if len(token) == 1:
        return (LETTERS.get(token, ""),), False             # a spelled letter
    d = cmudict().get(token)
    return (d, True) if d else ((g2p(token),), False)


def distance(a: list[str], b: list[str]) -> float:
    """Weighted phoneme edit distance / longer length."""
    if not a or not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            sub = 0.0 if x == y else 0.5 if ({x, y} <= VOWELS or frozenset((x, y)) in VOICING) else 1.0
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def _acronym_prons(caps: str, plural: bool) -> list[str]:
    letters = " ".join(LETTERS[c] for c in caps.lower()) + (" Z" if plural else "")
    word = [p + (" Z" if plural else "") for p in cmudict().get(caps.lower(), ())]
    return [letters, *word]


def slide_vocab(units: list[dict]) -> dict[str, list[str]]:
    """spelling -> pronunciations for the slide's map words (>= 3 phonemes)."""
    vocab: dict[str, list[str]] = {}

    def add(spelling: str, prons) -> None:
        for p in prons:
            if p and len(p.split()) >= 3 and p not in vocab.setdefault(spelling, []):
                vocab[spelling].append(p)

    for u in units:
        texts = [u.get("text") or "", u.get("summary") or "", *u.get("talk", []), *u.get("terms", [])]
        for t in texts:
            # spelled letters in a term ("s m m u", "h a l") -> the joined acronym
            for run in re.findall(r"\b(?:[a-z] ){1,5}[a-z]\b", t):
                add(run.replace(" ", ""), _acronym_prons(run.replace(" ", "").upper(), False))
            for tok in re.findall(r"[A-Za-z][A-Za-z']*", t):
                low = tok.lower()
                if low in STOP or low in POSITION or "'" in tok:
                    continue
                if m := re.fullmatch(r"([A-Z]{2,6})(s?)", tok):
                    add(low, _acronym_prons(m.group(1), bool(m.group(2))))
                elif len(low) >= 3:
                    add(low, cmudict().get(low) or (g2p(low),))
    return {w: p for w, p in vocab.items() if p and w not in STOP}


class SoundRepair:
    """Rewrite misheard spans of a clause to this slide's vocabulary words."""

    def __init__(self, units: list[dict], threshold: float = 0.25) -> None:
        self.threshold = threshold
        self.vocab = slide_vocab(units)
        self.exact: dict[str, set[str]] = {}
        self.entries: list[tuple[str, list[str], bool]] = []   # (word, phonemes, spelled as letters?)
        for w, prons in sorted(self.vocab.items()):
            spelled = " ".join(LETTERS[c] for c in w)
            for p in prons:
                self.exact.setdefault(p, set()).add(w)
                self.entries.append((w, p.split(), p in (spelled, " ".join(LETTERS[c] for c in w[:-1]) + " Z")))
        self._memo: dict[tuple[str, ...], tuple[str, float] | None] = {}

    def _eligible(self, tok: str) -> bool:
        return (tok.isalpha() and tok.isascii() and tok not in self.vocab and tok not in STOP
                and tok not in POSITION and (len(tok) >= 2 or tok not in ("a", "i")))

    def _match(self, span: tuple[str, ...]) -> tuple[str, float] | None:
        if span in self._memo:
            return self._memo[span]
        opts, fuzzy = [], False
        for tok in span:
            prons, in_dict = pron_options(tok)
            opts.append(prons[:2])
            fuzzy |= not in_dict
        letters = any(len(t) == 1 for t in span)   # spelled letters match only spelled vocab (acronyms)
        cands = [" ".join(c) for c in itertools.product(*opts)]
        best: tuple[str, float] | None = None
        if not fuzzy:
            hits = set().union(*(self.exact.get(c, set()) for c in cands))
            best = (hits.pop(), 0.0) if len(hits) == 1 else None
        else:
            scored: dict[str, float] = {}
            for c in cands:
                a = c.split()
                for w, b, spelled in self.entries:
                    if (spelled or not letters) and abs(len(a) - len(b)) <= self.threshold * max(len(a), len(b)):
                        d = distance(a, b)
                        if d <= self.threshold and d < scored.get(w, 9.0):
                            scored[w] = d
            if scored:
                lo = min(scored.values())
                tied = [w for w, d in scored.items() if d == lo]
                best = (tied[0], lo) if len(tied) == 1 else None
        self._memo[span] = best
        return best

    def __call__(self, text: str) -> str:
        toks = text.split()
        out, i = [], 0
        while i < len(toks):
            pick = None
            for n in (1, 2, 3):
                span = tuple(toks[i:i + n])
                if len(span) < n or not all(self._eligible(t) for t in span):
                    break
                m = self._match(span)
                if m and (pick is None or m[1] < pick[1]):
                    pick = (m[0], m[1], n)
            if pick:
                out.append(pick[0]); i += pick[2]
            else:
                out.append(toks[i]); i += 1
        return " ".join(out)

