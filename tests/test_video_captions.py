"""Burned-in captions + .srt sidecars (tools/video/captions.py)."""
from __future__ import annotations

import math

from PIL import Image

from tools.video.captions import MAX_CHARS, at, chunk, draw, layout, place, split_lines, srt, srt_deck, stamp, timed
from tools.video.effects import HUD, Effect
from tools.video.script import words

LONG = ("First, stage A reads the page, where a vision language O C R model turns the scan into text, "
        "handwriting included, and a computer vision detector catches the hand drawn circles the model would miss.")


def _len(ws):
    return len(" ".join(ws))


def _rows(sents, gap=10.0):
    """Word k of sentence s: start s*gap + 0.5k, 0.4 s long."""
    return [[{"word": w, "start": s * gap + 0.5 * k, "end": s * gap + 0.5 * k + 0.4} for k, w in enumerate(ws)]
            for s, ws in enumerate(sents)]


# ---- chunking / line breaking -----------------------------------------------

def test_split_lines_one_line_if_it_fits_else_a_balanced_pair():
    assert split_lines(words("So how does it actually work:")) == (tuple(words("So how does it actually work:")),)
    ws = words("Then stage B grades it, with a Qwen three model splitting")      # 57 chars
    lines = split_lines(ws)
    assert len(lines) == 2 and sum(lines, ()) == tuple(ws)
    best = min(max(_len(ws[:i]), _len(ws[i:])) for i in range(1, len(ws)))
    assert max(_len(ln) for ln in lines) == best <= MAX_CHARS
    assert _len(lines[0]) <= _len(lines[1])                       # ties: the shorter line on top


def test_long_sentence_splits_into_the_fewest_balanced_chunks():
    ws = words(LONG)                                               # 198 chars: 3 chunks at least
    chunks = chunk(ws)
    assert sum(chunks, ()) == tuple(ws)
    assert len(chunks) == 3
    for c in chunks:
        lines = split_lines(c)
        assert len(lines) <= 2 and all(_len(ln) <= MAX_CHARS for ln in lines)
    sizes = [_len(c) for c in chunks]
    assert max(sizes) - min(sizes) <= 12 and min(len(c) for c in chunks) >= 8   # no orphan chunk


def test_short_sentence_is_one_chunk_and_an_overlong_word_still_fits_somewhere():
    assert chunk(words("A graded P D F.")) == [tuple(words("A graded P D F."))]
    ws = ["see", "x" * 50, "here"]
    chunks = chunk(ws)
    assert sum(chunks, ()) == tuple(ws)
    assert all(len(ln) == 1 for c in chunks for ln in split_lines(c) if _len(ln) > MAX_CHARS)


# ---- timing -----------------------------------------------------------------

def test_chunks_run_from_their_first_word_to_the_next_chunk():
    sents = [words("So how does it work?"), words(LONG)]
    rows = _rows(sents)
    caps = timed(sents, rows)
    assert len(caps) == 4 and [w for c in caps for ln in c.lines for w in ln] == sents[0] + sents[1]
    assert caps[0].t0 == 0.0 and caps[0].t1 == caps[1].t0 == 10.0      # held through the sentence gap
    n1 = sum(len(ln) for ln in caps[1].lines)
    assert caps[1].t1 == caps[2].t0 == rows[1][n1]["start"]
    assert caps[-1].t1 == math.inf


def test_at_highlights_the_word_being_spoken():
    sents = [words("So how does it work?"), words("Then it grades.")]
    caps = timed(sents, _rows(sents))
    first = caps[0]
    assert at(-0.01, caps) is None                                   # before the first word
    assert at(0.0, caps) == (first, 0)
    assert at(0.45, caps) == (first, 0)                              # between words: the last one stays
    assert at(1.5, caps) == (first, 3)
    assert at(2.39, caps) == (first, 4) and at(2.4, caps) == (first, None)   # last word over: no highlight
    assert at(9.99, caps) == (first, None)
    assert at(10.0, caps) == (caps[1], 0)
    assert at(1e6, caps) == (caps[1], None)                          # the last caption holds


def test_punctuation_only_words_are_never_highlighted():
    sents = [["Read", "—", "then", "grade."]]
    rows = [[{"word": "Read", "start": 0.0, "end": 0.3}, {"word": "—", "start": 0.0, "end": 0.3},
             {"word": "then", "start": 0.4, "end": 0.6}, {"word": "grade.", "start": 0.6, "end": 0.9}]]
    caps = timed(sents, rows)
    assert at(0.1, caps) == (caps[0], 0)


# ---- srt --------------------------------------------------------------------

def test_srt_format_and_offsets():
    assert stamp(3661.5) == "01:01:01,500" and stamp(0.0) == "00:00:00,000" and stamp(59.9996) == "00:01:00,000"
    sents = [words("So how does it work?"), words("Then stage B grades it, with a Qwen three model splitting")]
    caps = timed(sents, _rows(sents))
    assert srt(caps, 16.25) == ("1\n00:00:00,000 --> 00:00:10,000\nSo how does it work?\n\n"
                                "2\n00:00:10,000 --> 00:00:16,250\nThen stage B grades it, with\n"
                                "a Qwen three model splitting\n\n")       # 28 / 28 chars
    assert srt(caps, 16.25, offset=20.6, first=3).startswith("3\n00:00:20,600 --> 00:00:30,600\n")
    deck = srt_deck([(caps, 16.25), (caps, 16.25)], 0.6)
    assert deck == srt(caps, 16.25) + srt(caps, 16.25, offset=16.85, first=3)


# ---- placement --------------------------------------------------------------

def test_panel_goes_to_the_top_only_when_the_bottom_would_cover_an_avoided_box():
    frame, wh = (1280, 720), (500, 80)
    bottom = place(wh, frame, [])
    assert bottom == (390, 720 - 29 - 80)                            # centred, 4 % bottom margin
    assert place(wh, frame, [(0, 0, 1280, 300)]) == bottom           # something at the top: stay
    assert place(wh, frame, [(600, 650, 100, 40)]) == (390, 29)      # covers the bottom spot: top
    # both spots covered: the one covering less (bottom 100x40 against top 500x21)
    assert place(wh, frame, [(600, 650, 100, 40), (0, 0, 1280, 50)]) == bottom


def test_layout_dodges_every_effect_live_during_the_caption():
    sents = [words("One two three."), words("Four five six."), words("Seven eight nine."),
             words("Ten eleven."), words("Twelve thirteen.")]
    low, high = (400, 620, 480, 60), (100, 300, 200, 60)    # bottom-centre, mid-left
    plan = [Effect("focus", 1.0, 10.0, low, {"spotlight": False}),               # sentence 0, at the bottom
            Effect("glide", 10.0, 20.0, high, {"from": low, "spotlight": False}),  # path starts at the bottom
            Effect("ring", 20.0, 30.0, low),                                       # a mark at the bottom: dodged
            Effect("zoom", 30.0, 33.0, high, {"lens": (500, 600, 300, 100)}),
            Effect("underline", 40.0, 50.0, (300, 640, 600, 30))]                 # underline at the bottom: dodged
    tops = [c.xy[1] < 360 for c in layout(sents, _rows(sents), plan, (1280, 720))]
    assert tops == [True, True, True, True, True]


def test_draw_is_deterministic_and_highlights_in_the_accent():
    sents = [words("So how does it work?")]
    cap = layout(sents, _rows(sents), [], (640, 360))[0]
    white = Image.new("RGB", (640, 360), (255, 255, 255))
    a, b, c, plain = white.copy(), white.copy(), white.copy(), white.copy()
    draw(a, cap, 0)
    draw(b, cap, 0)
    draw(c, cap, 1)
    draw(plain, cap, None)
    assert a.tobytes() == b.tobytes() != c.tobytes()
    x, y = cap.xy
    assert a.getpixel((x, y)) == (255, 255, 255)                      # rounded corner: left alone
    assert sum(a.getpixel((x + 8, y + 2))) < 3 * 140                  # the panel darkens the slide (60 %)
    near = lambda img: sum(1 for p in img.getdata() if all(abs(p[i] - HUD["accent"][i]) < 30 for i in range(3)))
    assert near(a) > 20 and near(plain) == 0
