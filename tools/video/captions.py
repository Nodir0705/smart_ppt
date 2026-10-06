"""Burned-in captions (karaoke HUD panel) and ``.srt`` sidecars. Pure, deterministic.

- One caption = one script sentence, its words as written (``script.words``), on
  at most 2 lines of ``MAX_CHARS`` characters. A longer sentence becomes the
  fewest consecutive chunks that each fit (``chunk``), made even by least sum
  of squared chunk lengths, so no 1-word orphan trails a long sentence. Lines
  break where the longer line is shortest (``split_lines``; ties put the
  shorter line on top). A single word over ``MAX_CHARS`` gets a line of its own.
- Timing (``timed``): a chunk shows from its first word's start to the next
  chunk's start (across sentences, so it holds through the 350 ms pause); the
  slide's last chunk holds to the end (``math.inf``; the .srt ends it at the
  wav's end). ``at`` lights the last word whose start has passed, skipping
  pure punctuation; once the chunk's final word ends, none is lit.
- Look: ``FONT`` (DejaVu Sans, a system font, never downloaded; Pillow's
  bundled font if it is missing) at ``FONT_H`` of the frame height, white with
  the spoken word in the HUD accent, on a rounded panel of the HUD halo colour
  at ``ALPHA``. The panel's masks are memoized per (chunk lines, lit word, font
  px), so a frame costs three mask pastes.
- Placement (``place``/``layout``), once per chunk so it never jumps mid-chunk:
  bottom-centred, ``MARGIN`` off the frame edge, unless it would cover a focus
  (at its 108 % snap), a glide's whole path or a zoom lens live at any moment
  of the chunk (each grown by ``GAP``); then top-centred. If both spots are
  covered, the one covering less (ties: top). Rings and underlines are not
  avoided.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import NamedTuple

from PIL import Image, ImageDraw, ImageFont

from tools.video.effects import HUD, REF_H, SNAP_FROM, _grow, _scale
from tools.video.geometry import union

MAX_CHARS = 42
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_H = 0.033          # font px / frame height
MARGIN = 0.04           # panel to the frame's bottom (or top) edge / frame height
ALPHA = 0.6
GAP = 16                # px at 1080p kept clear around a focus / lens (bracket halo + glow)
TEXT = (255, 255, 255)


class Cap(NamedTuple):
    lines: tuple        # ((word, ...), ...): 1 or 2 lines
    starts: tuple       # each word's start (slide-audio s), over the lines in order
    ends: tuple
    t0: float
    t1: float           # the next chunk's t0; math.inf for the slide's last
    xy: tuple = (0, 0)  # panel's top-left in frame px (``layout``)


def _len(ws) -> int:
    return len(" ".join(ws))


# ---- chunking / line breaking -----------------------------------------------

def split_lines(ws) -> tuple:
    ws = tuple(ws)
    if len(ws) < 2 or _len(ws) <= MAX_CHARS:
        return (ws,)
    i = min(range(1, len(ws)), key=lambda i: max(_len(ws[:i]), _len(ws[i:])))
    return (ws[:i], ws[i:])


def _fits(ws) -> bool:
    return len(ws) == 1 or max(map(_len, split_lines(ws))) <= MAX_CHARS


def chunk(ws) -> list[tuple]:
    """The fewest chunks that each fit on 2 lines, then the least sum of squared lengths."""
    ws = tuple(ws)
    best: list = [(0, 0, 0)] + [None] * len(ws)      # per prefix: (chunks, sum sq, last chunk's start)
    for j in range(1, len(ws) + 1):
        for i in range(j - 1, -1, -1):
            if j - i > 1 and _len(ws[i:j]) > 2 * MAX_CHARS + 1:
                break                                   # longer chunks never fit
            if best[i] is not None and _fits(ws[i:j]):
                c = (best[i][0] + 1, best[i][1] + _len(ws[i:j]) ** 2, i)
                if best[j] is None or c[:2] < best[j][:2]:
                    best[j] = c
    out, j = [], len(ws)
    while j:
        i = best[j][2]
        out.append(ws[i:j])
        j = i
    return out[::-1]


# ---- timing -----------------------------------------------------------------

def timed(sents, rows) -> list[Cap]:
    """Chunks of every sentence with their word times (``rows`` = timings' sentences, matching ``sents``)."""
    caps = []
    for ws, row in zip(sents, rows):
        k = 0
        for c in chunk(ws):
            r = row[k:k + len(c)]
            k += len(c)
            caps.append(Cap(split_lines(c), tuple(w["start"] for w in r), tuple(w["end"] for w in r),
                            r[0]["start"], math.inf))
    return [c._replace(t1=n.t0) for c, n in zip(caps, caps[1:])] + caps[-1:]


def at(t: float, caps) -> tuple | None:
    """``(caption, lit word index or None)`` shown at ``t``, or None."""
    for c in caps:
        if c.t0 <= t < c.t1:
            if t >= c.ends[-1]:
                return (c, None)
            ws, hi = sum(c.lines, ()), None
            for k, (w, s) in enumerate(zip(ws, c.starts)):
                if s <= t and any(ch.isalnum() for ch in w):
                    hi = k
            return (c, hi)
    return None


# ---- srt --------------------------------------------------------------------

def stamp(t: float) -> str:
    ms = round(t * 1000)
    return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def srt(caps, end: float, offset: float = 0.0, first: int = 1) -> str:
    """SubRip entries for ``caps`` (the last one ends at ``end``), shifted by ``offset`` s, numbered from ``first``."""
    return "".join(f"{first + i}\n{stamp(offset + c.t0)} --> {stamp(offset + min(c.t1, end))}\n"
                   + "\n".join(" ".join(ln) for ln in c.lines) + "\n\n" for i, c in enumerate(caps))


def srt_deck(slides, gap: float) -> str:
    """``slides`` = [(caps, wav seconds)] in deck order, ``gap`` s between them (as ``render.concat``).
    ponytail: offsets follow the wav lengths; the deck's AAC rounds each slide up by < 1 frame (21 ms)."""
    out, off, first = [], 0.0, 1
    for caps, dur in slides:
        out.append(srt(caps, dur, off, first))
        off, first = off + dur + gap, first + len(caps)
    return "".join(out)


# ---- placement --------------------------------------------------------------

def _area(a, b) -> int:
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h)


def place(wh, frame_size, avoid) -> tuple:
    """Panel top-left: bottom-centred unless that covers ``avoid``; then top, unless top covers more."""
    (w, h), (W, H) = wh, frame_size
    m, x = round(MARGIN * H), (W - w) // 2
    spots = [(x, H - m - h), (x, m)]
    cover = [sum(_area((sx, sy, w, h), b) for b in avoid) for sx, sy in spots]
    return spots[0] if cover[0] == 0 or cover[0] < cover[1] else spots[1]


def _avoid(plan, t0, t1, frame_h) -> list:
    g, out = round(GAP * frame_h / REF_H), []
    for e in plan:
        if e.t0 < t1 and t0 < e.t1 and e.kind in ("focus", "glide", "zoom"):
            b = (_scale(e.box, SNAP_FROM) if e.kind == "focus" else
                 union([e.extra["from"], e.box]) if e.kind == "glide" else e.extra["lens"])
            out.append(_grow(b, g))
    return out


def layout(sents, rows, plan, size) -> list[Cap]:
    """``timed`` captions placed clear of ``plan``'s foci and lenses (``effects.Effect``s, frame px)."""
    fs = _font_px(size[1])
    return [c._replace(xy=place(_panel(c.lines, None, fs)[0].size, size, _avoid(plan, c.t0, c.t1, size[1])))
            for c in timed(sents, rows)]


# ---- drawing ----------------------------------------------------------------

def _font_px(frame_h: int) -> int:
    return max(8, round(FONT_H * frame_h))


@lru_cache(maxsize=8)
def _font(px: int):
    try:
        return ImageFont.truetype(FONT, px)
    except OSError:
        return ImageFont.load_default(px)


@lru_cache(maxsize=256)
def _panel(lines, hi, fs) -> tuple:
    """L masks (panel, white words, lit word) for one caption state."""
    font = _font(fs)
    asc, desc = font.getmetrics()
    px, py, pitch = round(0.6 * fs), round(0.3 * fs), asc + desc
    widths = [font.getlength(" ".join(ln)) for ln in lines]
    w, h = math.ceil(max(widths)) + 2 * px, len(lines) * pitch + 2 * py
    panel, text, lit = (Image.new("L", (w, h), 0) for _ in range(3))
    ImageDraw.Draw(panel).rounded_rectangle([0, 0, w - 1, h - 1], radius=min(round(0.35 * fs), h // 2),
                                            fill=round(255 * ALPHA))
    k = 0
    for i, (ln, lw) in enumerate(zip(lines, widths)):
        x0, y = (w - lw) / 2, py + asc + i * pitch
        for j, word in enumerate(ln):
            x = x0 + (font.getlength(" ".join(ln[:j]) + " ") if j else 0)
            ImageDraw.Draw(lit if k == hi else text).text((x, y), word, fill=255, font=font, anchor="ls")
            k += 1
    return panel, text, lit


def draw(img: Image.Image, cap: Cap, hi, style=HUD) -> None:
    """Paint ``cap`` (``hi`` = lit word index or None) onto ``img`` in place."""
    panel, text, lit = _panel(cap.lines, hi, _font_px(img.height))
    img.paste(style["halo"], cap.xy, panel)
    img.paste(TEXT, cap.xy, text)
    img.paste(style["accent"], cap.xy, lit)
