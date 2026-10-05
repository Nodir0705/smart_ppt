"""Clause pointer: pause detection, clause segmentation + commit rule, and the
LiveController switch (mapped slides only). Scripted deciders, no models."""
from __future__ import annotations

import numpy as np

from src.clause_pointer import ClauseParams, ClausePointer, PauseDetector, slide_units
from src.live import LiveTranscriber
from src.live_controller import LiveController
from src.stt import Partial, RollingBuffer

UNITS = [{"uid": u, "text": t, "summary": "", "talk": [], "terms": []}
         for u, t in (("t", "title"), ("a", "alpha"), ("b", "beta"))]


class KeywordDecider:
    """Scores a unit 1 per occurrence of its keyword in the clause; none = 0.5."""

    def __init__(self, kw: dict[str, str]) -> None:
        self.kw, self.calls = kw, []

    def decide(self, clause, context, units):
        self.calls.append((clause, context))
        out = {u["uid"]: float(clause.split().count(self.kw[u["uid"]])) for u in units}
        out["none"] = 0.5
        return out


def _ptr(**kw):
    d = KeywordDecider({"t": "title", "a": "alpha", "b": "beta"})
    p = ClauseParams(**{"min_words": 2, "commit": 0.0, "gap": 0.5, "jump_gap": 0.5, "switch": 1.0,
                        "title_extra": 2.0, **kw})
    return ClausePointer(d, UNITS, p, low_prior={"t"}), d


def test_commits_mid_clause_and_holds_through_non_referential_clauses():
    cp, _ = _ptr()
    w = "so here alpha is".split()
    assert cp.update(w[:1]).active is None              # below min_words: no decision
    assert cp.update(w[:2]).why == "noop"
    assert cp.update(w[:3]).active == "a"               # early commit, mid-clause
    w += "and then some filler talk".split()
    d = cp.update(w, [4])                               # new clause names nothing
    assert (d.active, d.why) == ("a", "hold")


def test_hysteresis_inside_a_clause_and_free_choice_after_a_boundary():
    w = "look alpha then beta beta".split()
    cp2, _ = _ptr()
    cp2.update(w[:2])
    assert cp2.update(w[:4]).active == "a"              # beta 1 vs alpha 1: not +1 better, hold
    assert cp2.update(w).active == "b"                  # beta 2 vs alpha 1: switch
    cp3, _ = _ptr()
    cp3.update("look alpha".split())
    d = cp3.update("look alpha now beta".split(), [2])  # boundary: clause "now beta" decides afresh
    assert d.active == "b"


def test_title_units_need_the_extra_margin():
    cp, _ = _ptr(title_extra=1.0)
    assert cp.update("the title here".split()).active is None
    assert cp.update("the title title".split()).active == "t"


def test_a_new_clause_needs_jump_gap_to_move_a_lit_box():
    cp, _ = _ptr(jump_gap=1.5)
    cp.update("look alpha".split())
    assert cp.update("look alpha now beta".split(), [2]).active == "a"        # beta 1 vs alpha 0: < 1.5
    assert cp.update("look alpha now beta beta".split()).active == "b"        # 2 vs 0
    fresh, _ = _ptr(jump_gap=1.5)
    assert fresh.update("now beta".split()).active == "b"                     # first light: plain gap


def test_min_content_waits_for_a_content_word_before_a_clause_first_decision():
    cp, d = _ptr(min_content=1)
    assert cp.update("and on the left is".split()).active is None   # filler + position words only
    assert d.calls == []                                            # no decider call either
    assert cp.update("and on the left is alpha".split()).active == "a"
    cp.update("and on the left is alpha so then".split(), [6])      # new clause "so then": gated
    assert d.calls[-1][0] == "and on the left is alpha"


def test_titles_light_only_on_their_own_words():
    class Loud(KeywordDecider):                     # scores the title high whatever is said
        def decide(self, clause, context, units):
            return {**super().decide(clause, context, units), "t": 9.0}
    p = ClauseParams(min_words=2, commit=0.0, gap=0.5, title_extra=1.0)
    cp = ClausePointer(Loud({"t": "title", "a": "alpha", "b": "beta"}), UNITS, p, low_prior={"t"})
    assert cp.title_words == {"t": {"title"}}
    assert cp.update("so how alpha".split()).active == "a"          # title gated out: alpha wins
    assert cp.update("so how alpha the title".split()).active == "t"


def test_max_words_starts_a_new_clause_and_keeps_context():
    cp, d = _ptr(max_words=4)
    cp.update("alpha one two".split())
    cp.update("alpha one two three".split())            # reaches max_words: cut
    assert cp.start == 4 and cp.prev == "alpha one two three"
    cp.update("alpha one two three beta x".split())
    assert d.calls[-1] == ("beta x", "alpha one two three")


def test_pause_detector_fires_when_speech_resumes_after_a_pause():
    sr = 16000
    rng = np.random.default_rng(0)
    speech = (rng.standard_normal(sr // 10) * 3000).astype(np.int16).tobytes()   # one 100 ms chunk
    silence = np.zeros(sr // 10, dtype=np.int16).tobytes()
    pd = PauseDetector(sr)
    seq = [speech] * 5 + [silence] * 2 + [speech] + [silence] * 3 + [speech] * 2
    fired = [pd.feed(c) for c in seq]
    # 0.2 s of silence is no boundary; 0.3 s is, on the chunk where speech resumes
    assert fired == [False] * 5 + [False, False, False] + [False] * 3 + [True, False]


def test_buffer_keeps_the_whole_transcript_and_boundaries():
    b = RollingBuffer(window_seconds=1.0)
    b.push(Partial("one two", True, 0.0))
    b.push(Partial("three", True, 5.0))                 # prunes "one two" from the tail window
    b.push(Partial("four", False, 5.1))
    assert b.text == "three four"
    assert b.words() == ["one", "two", "three", "four"]
    b.boundaries.append(3)
    b.clear()
    assert b.words() == [] and b.boundaries == []


def test_transcriber_shows_every_chunk_to_the_buffer_before_stt():
    seen = []

    class Buf(RollingBuffer):
        def audio(self, chunk, sample_rate=16000):
            seen.append(chunk)

    class Src:
        sample_rate = 16000

        def chunks(self):
            yield b"\x00\x00" * 4
            yield b"\x01\x00" * 4

    class Eng:
        def transcribe_stream(self, chunks, hotwords=None):
            for _c in chunks:
                assert seen                              # the buffer saw it first
                yield Partial("x", False, 0.0)

    LiveTranscriber(Src(), Eng(), Buf()).run(lambda *_: None)
    assert len(seen) == 2


def _mapped_index():
    els = [{"id": "s1_t", "text": "Pipeline", "map_role": "title", "bbox": {"x": 0, "y": 0, "w": 1, "h": 1}},
           {"id": "s1_a", "text": "Alpha stage", "map_role": "body", "bbox": {"x": 0, "y": 2, "w": 1, "h": 1}},
           {"id": "s1_b", "text": "Beta stage", "map_role": "body", "bbox": {"x": 0, "y": 4, "w": 1, "h": 1}}]
    info = {u: {"summary": s, "talk": [], "terms": []} for u, s in (("t", "title"), ("a", "alpha"), ("b", "beta"))}
    return {"slides": [{"slide_number": 1, "elements": els,
                        "map": {"units": {"t": ["s1_t"], "a": ["s1_a"], "b": ["s1_b"]}, "info": info}},
                       {"slide_number": 2, "elements": [{"id": "s2_x", "text": "x", "bbox": {}}]}]}


def test_slide_units_reads_the_map():
    units, head, low = slide_units(_mapped_index()["slides"][0])
    assert [u["uid"] for u in units] == ["t", "a", "b"]
    assert units[1]["text"] == "Alpha stage" and units[1]["summary"] == "alpha"
    assert head == {"t": "s1_t", "a": "s1_a", "b": "s1_b"} and low == {"t"}
    units, _, _ = slide_units(_mapped_index()["slides"][0], {"w": 4.0, "h": 5.0})
    assert [u["terms"] for u in units] == [["on the left top"], ["on the left"], ["on the left bottom"]]


class _NoScores:
    """Tail-policy matcher stub: records what reaches it."""

    def __init__(self):
        self.seen = []

    def scored(self, spoken, slide_no):
        self.seen.append(spoken)
        return []


def test_controller_uses_the_clause_pointer_on_mapped_slides_only():
    m = _NoScores()
    dec = KeywordDecider({"t": "pipeline", "a": "alpha", "b": "beta"})
    c = LiveController(m, _mapped_index(), normalizer=lambda t: t, decider=dec,
                       clause_params=ClauseParams(min_words=2, gap=0.5, commit=0.0))
    c.push(Partial("now the alpha", False, 1.0))
    ev = c.feed()
    assert ev.active_id == "s1_a" and ev.tail == "now the alpha" and not m.seen
    c.slide_no = 2
    c.reset()
    c.push(Partial("now the alpha part", False, 2.0))
    c.feed()
    assert m.seen and not dec.calls[1:]                  # unmapped slide: tail policy, no decider


def test_controller_tail_pointer_ignores_maps():
    m = _NoScores()
    dec = KeywordDecider({"t": "pipeline", "a": "alpha", "b": "beta"})
    c = LiveController(m, _mapped_index(), normalizer=lambda t: t, decider=dec, pointer="tail")
    c.push(Partial("now the alpha", False, 1.0))
    c.feed()
    assert m.seen and not dec.calls


def test_controller_slide_speech_starts_at_reset():
    dec = KeywordDecider({"t": "pipeline", "a": "alpha", "b": "beta"})
    c = LiveController(_NoScores(), _mapped_index(), normalizer=lambda t: t, decider=dec,
                       clause_params=ClauseParams(min_words=2, gap=0.5, commit=0.0))
    c.push(Partial("beta beta beta", False, 1.0))
    c.reset()                                            # slide shown again: old words don't count
    c.push(Partial("beta beta beta and alpha now", False, 2.0))
    assert c.feed().active_id == "s1_a"


class Scripted:
    """Scores from a queue, one dict per decider call; none = 0."""

    def __init__(self, *calls):
        self.calls = list(calls)

    def decide(self, clause, context, units):
        return {**self.calls.pop(0), "none": 0.0}


CARD_UNITS = [{"uid": u, "text": "", "summary": "", "talk": [], "terms": [], "parents": p}
              for u, p in (("c", []), ("a", ["c"]), ("b", []))]


def _card_ptr(*calls, **kw):
    p = ClauseParams(**{"min_words": 1, "commit": 0.0, "gap": 1.0, "jump_gap": 2.0, "switch": 1.0,
                        "card": 1.0, "repair": 0.0, **kw})
    return ClausePointer(Scripted(*calls), CARD_UNITS, p)


def test_a_child_within_card_lights_its_card_and_holds_it():
    cp = _card_ptr({"a": 3.0, "c": 2.5, "b": 0.0}, {"a": 3.0, "c": 2.4, "b": 0.0})
    d = cp.update(["one"])
    assert (d.active, cp.committed) == ("c", "c")         # child a wins by 0.5 <= card: the card lights
    assert cp.update(["one", "two"]).active == "c"        # same evidence: no churn back to the child
    off = _card_ptr({"a": 3.0, "c": 2.5, "b": 0.0}, card=-1.0)
    assert off.update(["one"]).active is None             # off: a over c by 0.5 < gap, nothing lights


def test_a_child_far_above_its_card_lights_itself():
    cp = _card_ptr({"a": 5.0, "c": 2.5, "b": 0.0})
    assert cp.update(["one"]).active == "a"


def test_moving_between_a_card_and_its_child_is_no_topic_jump():
    cp = _card_ptr({"c": 3.0, "a": 0.0, "b": 0.0}, {"a": 4.0, "b": 2.5, "c": 1.0}, card=0.5)
    assert cp.update(["x"]).active == "c"
    assert cp.update(["x", "y"], [1]).active == "a"       # 1.5 over b: gap 1 suffices (own card -> child)
    cp = _card_ptr({"c": 3.0, "a": 0.0, "b": 0.0}, {"b": 4.0, "a": 2.5, "c": 1.0}, card=0.5)
    cp.update(["x"])
    assert cp.update(["x", "y"], [1]).active == "c"       # unrelated b needs jump_gap 2


def test_slide_units_reads_card_relations():
    idx = _mapped_index()
    idx["slides"][0]["map"]["relations"] = [{"from": "a", "to": "b", "type": "part_of"},
                                            {"from": "b", "to": "a", "type": "flow"}]
    units, _, _ = slide_units(idx["slides"][0])
    assert [u["parents"] for u in units] == [[], ["b"], []]


def test_the_pointer_scores_the_repaired_clause():
    seen = []

    class Rec(KeywordDecider):
        def decide(self, clause, context, units):
            seen.append(clause)
            return super().decide(clause, context, units)
    units = [{"uid": "k", "text": "Linux kernel", "summary": "", "talk": [], "terms": []},
             {"uid": "a", "text": "alpha", "summary": "", "talk": [], "terms": []}]
    cp = ClausePointer(Rec({"k": "kernel", "a": "alpha"}), units, ClauseParams(min_words=2, gap=0.5, commit=0.0, repair=0.25))
    assert cp.update("the colonel".split()).active == "k" and seen == ["the kernel"]
