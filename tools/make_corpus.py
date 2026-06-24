"""Generate a small, diverse deck corpus to catch parser generalization breaks.

The whole system was developed and tuned against one deck (`test_deck.pptx`),
so regressions that only show on *other* deck shapes go uncaught. Each deck here
is built programmatically with python-pptx, so the ground truth (which pointable
texts must survive parsing) is known *by construction* — no manual labelling.

    python3 -m tools.make_corpus                 # write decks + expected.json to corpus/
    python3 -m tools.make_corpus --out /tmp/c    # elsewhere

The acceptance harness (`tests/test_corpus.py`) regenerates these into a tmp dir
and asserts no author-intended element is silently dropped. Decks that target a
*known* parser gap (e.g. grouped shapes) carry ``known_gap`` so the harness can
xfail them until the gap is fixed, rather than failing the suite.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt


@dataclass
class CorpusDeck:
    name: str
    path: str
    # pointable strings that MUST appear in some parsed element (text or trigger)
    expected_texts: list[str] = field(default_factory=list)
    # expected element count is exact when >=0; -1 means "don't assert a count"
    expected_min_elements: int = 1
    # a documented parser gap this deck targets (harness xfails on it)
    known_gap: str | None = None
    note: str = ""


_BLANK = 6  # the "Blank" layout index in the default template


def _blank_slide(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[_BLANK])


def _textbox(shapes, text: str, x=1.0, y=1.0, w=6.0, h=1.0, size=24):
    tb = shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.text = text
    tf.paragraphs[0].font.size = Pt(size)
    return tb


# ---------- deck builders (each returns a CorpusDeck spec; writes the .pptx) ----------

def _build_multiline_body(out_dir: Path) -> CorpusDeck:
    """A single text box with several lines — exercises multi-line/line_bboxes,
    the data line-level highlighting needs and the test deck barely has."""
    prs = Presentation()
    s = _blank_slide(prs)
    tb = s.shapes.add_textbox(Inches(1), Inches(1.5), Inches(8), Inches(3))
    tf = tb.text_frame
    lines = ["The first line of the caption",
             "A second distinct line below it",
             "And a third concluding line"]
    tf.text = lines[0]
    for ln in lines[1:]:
        p = tf.add_paragraph()
        p.text = ln
    path = out_dir / "multiline_body.pptx"
    prs.save(str(path))
    return CorpusDeck("multiline_body", str(path), expected_texts=lines,
                      note="multi-paragraph body; should not lose any line")


def _build_title_and_bullets(out_dir: Path) -> CorpusDeck:
    """Title placeholder + a content box of bullets — the most common layout."""
    prs = Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[1])  # Title and Content
    s.shapes.title.text = "Quarterly Results"
    body = s.placeholders[1].text_frame
    bullets = ["Revenue grew", "Costs fell", "Margins expanded"]
    body.text = bullets[0]
    for b in bullets[1:]:
        body.add_paragraph().text = b
    path = out_dir / "title_and_bullets.pptx"
    prs.save(str(path))
    return CorpusDeck("title_and_bullets", str(path),
                      expected_texts=["Quarterly Results", *bullets],
                      note="title placeholder + multi-bullet content")


def _build_grouped_shapes(out_dir: Path) -> CorpusDeck:
    """Three text boxes inside a GROUP — SmartArt/process diagrams look like this.
    The parser now recurses GROUP shapes and maps each child from group child-
    space into slide coordinates, so all three must survive."""
    prs = Presentation()
    s = _blank_slide(prs)
    g = s.shapes.add_group_shape()
    texts = ["Alpha node", "Beta node", "Gamma node"]
    for i, t in enumerate(texts):
        tb = g.shapes.add_textbox(Inches(1 + i * 3), Inches(2), Inches(2.5), Inches(1))
        tb.text_frame.text = t
    path = out_dir / "grouped_shapes.pptx"
    prs.save(str(path))
    return CorpusDeck("grouped_shapes", str(path), expected_texts=texts,
                      expected_min_elements=3,
                      note="SmartArt-like grouped textboxes")


def _build_empty_slide(out_dir: Path) -> CorpusDeck:
    """A slide with no shapes — parsing must not crash and yields 0 elements."""
    prs = Presentation()
    _blank_slide(prs)
    path = out_dir / "empty_slide.pptx"
    prs.save(str(path))
    return CorpusDeck("empty_slide", str(path), expected_texts=[],
                      expected_min_elements=0,
                      note="0-element slide; must parse without error")


_BUILDERS = [
    _build_multiline_body,
    _build_title_and_bullets,
    _build_grouped_shapes,
    _build_empty_slide,
]


def build_corpus(out_dir: str | Path) -> list[CorpusDeck]:
    """Generate every corpus deck into `out_dir`; return their specs."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    decks = [b(out) for b in _BUILDERS]
    (out / "expected.json").write_text(
        json.dumps([asdict(d) for d in decks], indent=2)
    )
    return decks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the smartppt deck corpus.")
    ap.add_argument("-o", "--out", default="corpus", help="output dir (default: corpus)")
    args = ap.parse_args(argv)
    decks = build_corpus(args.out)
    for d in decks:
        gap = f"  [known gap: {d.known_gap}]" if d.known_gap else ""
        print(f"  {d.name:20s} {len(d.expected_texts)} pointable texts{gap}")
    print(f"wrote {len(decks)} decks + expected.json -> {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
