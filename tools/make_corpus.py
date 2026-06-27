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
    # element types that MUST be present among parsed elements (multiset subset);
    # empty means "don't assert types"
    expected_types: list[str] = field(default_factory=list)
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


def _build_centered_title_slide(out_dir: Path) -> CorpusDeck:
    """A title-slide layout: its CENTER_TITLE placeholder sits mid-page (y>1.2),
    where the geometry heuristic misclassifies it as body. Authoritative
    placeholder-type detection must still type it 'title'."""
    prs = Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[0])  # Title Slide (centred title)
    s.shapes.title.text = "Keynote Opening"
    path = out_dir / "centered_title.pptx"
    prs.save(str(path))
    return CorpusDeck("centered_title", str(path),
                      expected_texts=["Keynote Opening"],
                      expected_types=["title"],
                      note="centred title placeholder -> 'title' (heuristic would say body)")


def _build_image_alt_text(out_dir: Path) -> CorpusDeck:
    """A picture with author alt text — otherwise text-less raster content the
    presenter can still point at ("show the architecture diagram"). The parser
    reads cNvPr descr/title so the alt text becomes pointable triggers."""
    import io

    from PIL import Image

    prs = Presentation()
    s = _blank_slide(prs)
    buf = io.BytesIO()
    Image.new("RGB", (320, 240), (90, 130, 180)).save(buf, "PNG")
    buf.seek(0)
    pic = s.shapes.add_picture(buf, Inches(2), Inches(2), Inches(4), Inches(3))
    alt = "Architecture diagram of the live pipeline"
    pic._element.nvPicPr.cNvPr.set("descr", alt)
    path = out_dir / "image_alt_text.pptx"
    prs.save(str(path))
    return CorpusDeck("image_alt_text", str(path), expected_texts=[alt],
                      note="picture with alt text -> pointable without a VLM")


def _build_kpi_callouts(out_dir: Path) -> CorpusDeck:
    """Three numeric KPI callouts: an empty box with a number centred inside it,
    x3. The old slide-wide rule (>=3 pairs => diagram_label) mislabelled these as
    diagram labels; per-pair classification types each numeric callout 'stat'."""
    from pptx.enum.shapes import MSO_SHAPE

    prs = Presentation()
    s = _blank_slide(prs)
    nums = ["99%", "1.2 s", "4 GB"]
    for i, n in enumerate(nums):
        x = 0.7 + i * 3.2
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                 Inches(x), Inches(2.2), Inches(2.6), Inches(1.8))
        box.text_frame.clear()  # ensure the container itself is empty-text
        tb = s.shapes.add_textbox(Inches(x + 0.5), Inches(2.8), Inches(1.6), Inches(0.7))
        tb.text_frame.text = n
    path = out_dir / "kpi_callouts.pptx"
    prs.save(str(path))
    return CorpusDeck("kpi_callouts", str(path), expected_texts=nums,
                      expected_types=["stat", "stat", "stat"],
                      note="3 numeric callouts -> each 'stat' (not diagram_label)")


def _build_rotated_shape(out_dir: Path) -> CorpusDeck:
    """A wide-short text box rotated 90°. Its highlight must cover the rotated
    extent (tall-narrow AABB), not the unrotated wide-short box."""
    prs = Presentation()
    s = _blank_slide(prs)
    tb = s.shapes.add_textbox(Inches(2), Inches(3), Inches(6), Inches(1))
    tb.text_frame.text = "Rotated banner"
    tb.rotation = 90
    path = out_dir / "rotated_shape.pptx"
    prs.save(str(path))
    return CorpusDeck("rotated_shape", str(path),
                      expected_texts=["Rotated banner"],
                      note="90deg-rotated box -> AABB dims swap (w<h)")


def _build_grouped_bar_chart(out_dir: Path) -> CorpusDeck:
    """A clustered 2-series column chart. Each (category, series) must get its
    own distinct sub-column element, with the series name in its triggers."""
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE

    prs = Presentation()
    s = _blank_slide(prs)
    data = CategoryChartData()
    data.categories = ["Q1", "Q2", "Q3"]
    data.add_series("Revenue", (10, 20, 30))
    data.add_series("Cost", (5, 8, 12))
    s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED,
                       Inches(1), Inches(1), Inches(7), Inches(4.5), data)
    path = out_dir / "grouped_bar.pptx"
    prs.save(str(path))
    # 3 categories x 2 series = 6 chart_category + 1 chart
    return CorpusDeck("grouped_bar", str(path),
                      expected_texts=["Revenue", "Cost", "Q1", "Q3"],
                      expected_types=["chart"] + ["chart_category"] * 6,
                      note="clustered 2-series -> 6 distinct sub-column elements")


def _build_stacked_bar_chart(out_dir: Path) -> CorpusDeck:
    """A stacked 2-series column chart. Each (category, series) must get a
    distinct vertical segment stacked within its category column."""
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE

    prs = Presentation()
    s = _blank_slide(prs)
    data = CategoryChartData()
    data.categories = ["Q1", "Q2", "Q3"]
    data.add_series("Base", (10, 20, 30))
    data.add_series("Bonus", (5, 8, 12))
    s.shapes.add_chart(XL_CHART_TYPE.COLUMN_STACKED,
                       Inches(1), Inches(1), Inches(7), Inches(4.5), data)
    path = out_dir / "stacked_bar.pptx"
    prs.save(str(path))
    return CorpusDeck("stacked_bar", str(path),
                      expected_texts=["Base", "Bonus", "Q2"],
                      expected_types=["chart"] + ["chart_category"] * 6,
                      note="stacked 2-series -> 6 distinct vertical segments")


def _build_radar_chart(out_dir: Path) -> CorpusDeck:
    """A radar chart — no sound per-category bar/slice geometry. The parser must
    emit a single needs_vision chart element, not fabricated category bars."""
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE

    prs = Presentation()
    s = _blank_slide(prs)
    data = CategoryChartData()
    data.categories = ["Speed", "Power", "Range", "Cost"]
    data.add_series("Model A", (8, 6, 7, 5))
    s.shapes.add_chart(XL_CHART_TYPE.RADAR, Inches(1), Inches(1), Inches(6), Inches(4.5), data)
    path = out_dir / "radar_chart.pptx"
    prs.save(str(path))
    return CorpusDeck("radar_chart", str(path), expected_texts=[],
                      expected_types=["chart"], expected_min_elements=1,
                      note="radar -> single needs_vision chart, no fabricated bars")


def _build_pie_chart(out_dir: Path) -> CorpusDeck:
    """A single-series pie. Unlike radar, pie HAS sound geometry — each slice is
    an angular wedge — so the parser emits one value-proportional sub-region per
    category (needs_vision False), not a needs_vision punt."""
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE

    prs = Presentation()
    s = _blank_slide(prs)
    data = CategoryChartData()
    data.categories = ["North", "South", "East", "West"]
    data.add_series("Share", (40, 30, 20, 10))
    s.shapes.add_chart(XL_CHART_TYPE.PIE,
                       Inches(1), Inches(1), Inches(6), Inches(4.5), data)
    path = out_dir / "pie_chart.pptx"
    prs.save(str(path))
    return CorpusDeck("pie_chart", str(path),
                      expected_texts=["North", "South", "East", "West"],
                      expected_types=["chart"] + ["chart_category"] * 4,
                      note="pie -> 4 angular-wedge sub-regions (not needs_vision)")


def _build_nested_groups(out_dir: Path) -> CorpusDeck:
    """A textbox nested two GROUPs deep, plus one a single GROUP deep. Exercises
    the affine COMPOSITION through nesting — a depth-1 transform would leave the
    deep node in the wrong place or drop it entirely."""
    prs = Presentation()
    s = _blank_slide(prs)
    outer = s.shapes.add_group_shape()
    inner = outer.shapes.add_group_shape()
    tb = inner.shapes.add_textbox(Inches(2), Inches(2), Inches(2), Inches(1))
    tb.text_frame.text = "Deep node"
    tb2 = outer.shapes.add_textbox(Inches(5), Inches(3), Inches(2), Inches(1))
    tb2.text_frame.text = "Mid node"
    path = out_dir / "nested_groups.pptx"
    prs.save(str(path))
    return CorpusDeck("nested_groups", str(path),
                      expected_texts=["Deep node", "Mid node"],
                      expected_min_elements=2,
                      note="group-in-group; both nodes survive in slide coords")


def _build_table(out_dir: Path) -> CorpusDeck:
    """A standalone 2x3 table. Every cell must become its own pointable
    table_cell element (the test deck has a table, but only via the full deck —
    pin the contract in isolation)."""
    prs = Presentation()
    s = _blank_slide(prs)
    cells = [["Region", "Q1", "Q2"], ["North", "10", "20"]]
    t = s.shapes.add_table(2, 3, Inches(1), Inches(1), Inches(8), Inches(2)).table
    for r, row in enumerate(cells):
        for c, val in enumerate(row):
            t.cell(r, c).text = val
    path = out_dir / "table.pptx"
    prs.save(str(path))
    flat = [v for row in cells for v in row]
    return CorpusDeck("table", str(path), expected_texts=flat,
                      expected_types=["table_cell"] * 6,
                      note="2x3 table -> one table_cell per cell")


def _build_duplicate_text(out_dir: Path) -> CorpusDeck:
    """Two separate boxes with IDENTICAL text. The parser must NOT dedup them to
    one element — they are distinct pointables at distinct positions (the real
    'three Live cells' ambiguity)."""
    prs = Presentation()
    s = _blank_slide(prs)
    for x in (1.0, 6.0):
        tb = s.shapes.add_textbox(Inches(x), Inches(3), Inches(2), Inches(1))
        tb.text_frame.text = "Live"
    path = out_dir / "duplicate_text.pptx"
    prs.save(str(path))
    return CorpusDeck("duplicate_text", str(path), expected_texts=["Live"],
                      expected_min_elements=2,
                      note="identical text in 2 boxes -> 2 elements, not deduped")


def _build_unicode_text(out_dir: Path) -> CorpusDeck:
    """Non-ASCII content: em-dash, currency, accents. Parsing must preserve the
    glyphs verbatim (normalization/encoding must not drop or mangle them)."""
    prs = Presentation()
    s = _blank_slide(prs)
    tb = s.shapes.add_textbox(Inches(1), Inches(1), Inches(8), Inches(1))
    tb.text_frame.text = "Profit — €5M (déjà vu)"
    path = out_dir / "unicode_text.pptx"
    prs.save(str(path))
    return CorpusDeck("unicode_text", str(path),
                      expected_texts=["Profit — €5M (déjà vu)"],
                      note="non-ASCII (em-dash/€/accents) preserved verbatim")


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
    _build_centered_title_slide,
    _build_grouped_shapes,
    _build_image_alt_text,
    _build_kpi_callouts,
    _build_rotated_shape,
    _build_radar_chart,
    _build_grouped_bar_chart,
    _build_stacked_bar_chart,
    _build_pie_chart,
    _build_nested_groups,
    _build_table,
    _build_duplicate_text,
    _build_unicode_text,
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
