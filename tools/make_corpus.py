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
    # text only reachable with the OCR step (tesseract): the pure-parse fast test
    # skips these (it can't OCR); the slow OCR test asserts survival instead.
    requires_ocr: bool = False
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


def _build_wrapping_paragraph(out_dir: Path) -> CorpusDeck:
    """ONE long paragraph in a narrow box, so it soft-wraps across several visual
    lines WITHOUT paragraph breaks. The parser keeps it as a single element; the
    refine step is what splits it into per-line `line_bboxes` — the data
    line-level highlighting consumes. (Multi-PARAGRAPH bodies already split into
    separate elements at parse time, so this wrap case is the real multi-line
    one.) Refine needs soffice, so only the slow test exercises the geometry."""
    prs = Presentation()
    s = _blank_slide(prs)
    tb = s.shapes.add_textbox(Inches(1), Inches(1.5), Inches(3.0), Inches(2.5))
    tf = tb.text_frame
    tf.word_wrap = True
    para = ("This is a single long paragraph that must wrap across several "
            "visual lines inside one narrow text box")
    tf.text = para
    tf.paragraphs[0].font.size = Pt(20)
    path = out_dir / "wrapping_paragraph.pptx"
    prs.save(str(path))
    return CorpusDeck("wrapping_paragraph", str(path), expected_texts=[para],
                      note="one soft-wrapping paragraph -> multi-line line_bboxes")


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


def _build_titled_chart_acronym(out_dir: Path) -> CorpusDeck:
    """A titled single-series column chart plus an acronym defined once in body
    text ("Speech to Text (STT)") and used bare elsewhere ("STT Engine"). Drives
    deck-agnostic group labels (from the chart's own title) and auto-derived
    acronym expansions — neither may depend on test_deck vocabulary."""
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE

    prs = Presentation()
    s = _blank_slide(prs)
    _textbox(s.shapes, "Speech to Text (STT) runs offline", y=0.3, h=0.8)
    _textbox(s.shapes, "STT Engine", x=7.0, y=6.0, w=2.5, h=0.8, size=18)
    data = CategoryChartData()
    data.categories = ["North", "South"]
    data.add_series("Revenue", (40, 30))
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED,
                            Inches(1), Inches(1.5), Inches(6), Inches(4.5), data)
    gf.chart.has_title = True
    gf.chart.chart_title.text_frame.text = "Revenue by region (USD)"
    path = out_dir / "titled_chart_acronym.pptx"
    prs.save(str(path))
    return CorpusDeck("titled_chart_acronym", str(path),
                      expected_texts=["North", "South", "STT Engine"],
                      expected_types=["chart", "chart_category", "chart_category"],
                      note="chart title -> group label; 'Words (ACR)' -> expansion")


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


def _build_image_baked_text(out_dir: Path) -> CorpusDeck:
    """A picture with text rendered INTO the raster and NO alt text. Pure parsing
    can't reach it, but the OCR step (tesseract) recovers "NET REVENUE" — so this
    is requires_ocr: the fast test skips it, the slow OCR test asserts survival."""
    import io

    from PIL import Image, ImageDraw

    prs = Presentation()
    s = _blank_slide(prs)
    img = Image.new("RGB", (400, 200), (255, 255, 255))
    ImageDraw.Draw(img).text((20, 80), "NET REVENUE", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    s.shapes.add_picture(buf, Inches(2), Inches(2), Inches(4), Inches(2))  # no alt
    path = out_dir / "image_baked_text.pptx"
    prs.save(str(path))
    return CorpusDeck("image_baked_text", str(path),
                      expected_texts=["NET REVENUE"],
                      requires_ocr=True,
                      note="image w/ rendered text, no alt -> recovered by tesseract OCR")


def _build_image_no_text(out_dir: Path) -> CorpusDeck:
    """A picture with visual STRUCTURE but NO text (geometric shapes) and no alt.
    The precision guard: OCR readily hallucinates tokens on structured noise, so
    this must yield NO pointable text — `needs_vision` stays True. (A solid fill
    would prove nothing; shapes are what make tesseract hallucinate.)"""
    import io

    from PIL import Image, ImageDraw

    prs = Presentation()
    s = _blank_slide(prs)
    img = Image.new("RGB", (400, 300), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    dr.rectangle([30, 30, 180, 160], outline=(0, 0, 0), width=4)
    dr.ellipse([220, 40, 360, 180], outline=(20, 20, 20), width=5)
    dr.line([0, 300, 400, 0], fill=(0, 0, 0), width=3)
    dr.polygon([(100, 220), (180, 290), (40, 290)], outline=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    s.shapes.add_picture(buf, Inches(2), Inches(1.5), Inches(4), Inches(3))  # no alt
    path = out_dir / "image_no_text.pptx"
    prs.save(str(path))
    return CorpusDeck("image_no_text", str(path), expected_texts=[],
                      requires_ocr=True,
                      note="shapes, no text -> OCR must NOT hallucinate a pointable")


_MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
_A14_NS = "http://schemas.microsoft.com/office/drawing/2010/main"
_M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _build_math_textbox(out_dir: Path) -> CorpusDeck:
    """A text box holding an equation, saved the way PowerPoint saves it: wrapped
    in `mc:AlternateContent` whose Choice is the live box (the math in `a14:m`
    between ordinary runs) and whose Fallback is a picture of the whole box.
    python-pptx iterates neither branch, so the parser must resolve the Choice
    itself and read the math runs ("Let A, B be languages", math italics
    normalised). Plus an ordinary box with a soft line break (`a:br`), which
    must stay a line boundary ("TAs:\\nFadi"), not glue "TAs:" to "Fadi".
    Hand-built with lxml — python-pptx cannot author any of it."""
    import copy
    import io

    from lxml import etree
    from PIL import Image
    from pptx.oxml.ns import qn

    prs = Presentation()
    s = _blank_slide(prs)
    _textbox(s.shapes, "Regular languages", y=0.3, w=8.0, h=0.8)

    # Choice: "Let " + <a14:m>𝐴,␣𝐵</a14:m> + " be languages" (␣ = U+00A0)
    choice_sp = s.shapes.add_textbox(Inches(1), Inches(2), Inches(8), Inches(1))
    p = choice_sp.text_frame.paragraphs[0]
    p.add_run().text = "Let "
    p.add_run().text = " be languages"
    p.runs[0]._r.addnext(etree.fromstring(
        f'<a14:m xmlns:a14="{_A14_NS}"><m:oMath xmlns:m="{_M_NS}">'
        '<m:r><m:t>\U0001D434</m:t></m:r><m:r><m:t>, </m:t></m:r>'
        '<m:r><m:t>\U0001D435</m:t></m:r></m:oMath></a14:m>'))

    # Fallback: the same box filled with a picture (what PowerPoint drew). The
    # image part + relationship come from a throwaway python-pptx picture.
    buf = io.BytesIO()
    Image.new("RGB", (160, 20), (20, 40, 120)).save(buf, "PNG")
    buf.seek(0)
    pic = s.shapes.add_picture(buf, Inches(1), Inches(2), Inches(8), Inches(1))
    fallback_sp = copy.deepcopy(choice_sp._element)
    fallback_sp.remove(fallback_sp.txBody)
    fallback_sp.spPr.replace(fallback_sp.spPr.find(qn("a:noFill")), etree.fromstring(
        f'<a:blipFill xmlns:a="{_A_NS}" xmlns:r="{_R_NS}">'
        f'<a:blip r:embed="{pic._element.blip_rId}"/><a:stretch><a:fillRect/></a:stretch>'
        '</a:blipFill>'))
    pic._element.getparent().remove(pic._element)

    ac = etree.Element(f"{{{_MC_NS}}}AlternateContent", nsmap={"mc": _MC_NS, "a14": _A14_NS})
    choice = etree.SubElement(ac, f"{{{_MC_NS}}}Choice", Requires="a14")
    fallback = etree.SubElement(ac, f"{{{_MC_NS}}}Fallback")
    choice_sp._element.addprevious(ac)
    choice.append(choice_sp._element)
    fallback.append(fallback_sp)

    tb = s.shapes.add_textbox(Inches(1), Inches(4), Inches(4), Inches(1))
    p = tb.text_frame.paragraphs[0]
    p.add_run().text = "TAs:"
    p.add_line_break()
    p.add_run().text = "Fadi"

    path = out_dir / "math_textbox.pptx"
    prs.save(str(path))
    return CorpusDeck("math_textbox", str(path),
                      expected_texts=["Regular languages", "Let A, B be languages", "TAs:\nFadi"],
                      note="mc:AlternateContent equation box -> Choice text incl. math; a:br -> newline")


def _build_hidden_shapes(out_dir: Path) -> CorpusDeck:
    """Shapes PowerPoint hides (`p:cNvPr hidden="1"` — Selection Pane "eye off"):
    a hidden text box and a hidden GROUP holding a text box, beside a visible
    box. Hidden shapes are not on the rendered slide, so none may be indexed;
    the visible sibling must survive. python-pptx cannot author `hidden`."""
    prs = Presentation()
    s = _blank_slide(prs)
    _textbox(s.shapes, "Visible sibling", y=1.0)
    hidden = _textbox(s.shapes, "Hidden box", y=2.5)
    hidden._element.nvSpPr.cNvPr.set("hidden", "1")
    g = s.shapes.add_group_shape()
    g.shapes.add_textbox(Inches(1), Inches(4), Inches(4), Inches(1)).text_frame.text = "Inside hidden group"
    g._element.nvGrpSpPr.cNvPr.set("hidden", "1")
    path = out_dir / "hidden_shapes.pptx"
    prs.save(str(path))
    return CorpusDeck("hidden_shapes", str(path), expected_texts=["Visible sibling"],
                      note="hidden shape / hidden group subtree -> not indexed")


def _build_off_canvas(out_dir: Path) -> CorpusDeck:
    """A text box parked entirely below the 7.5-in slide (authors stash notes
    and leftovers off-canvas) is not on the rendered slide → not indexed. One
    that merely overhangs the bottom edge stays."""
    prs = Presentation()
    s = _blank_slide(prs)
    _textbox(s.shapes, "On the slide", y=1.0)
    _textbox(s.shapes, "Overhanging the edge", y=7.3, h=0.5)
    _textbox(s.shapes, "Parked off-canvas", y=9.0)
    path = out_dir / "off_canvas.pptx"
    prs.save(str(path))
    return CorpusDeck("off_canvas", str(path),
                      expected_texts=["On the slide", "Overhanging the edge"],
                      note="shape wholly outside the slide -> not indexed; partial overlap stays")


def _build_occluded_text(out_dir: Path) -> CorpusDeck:
    """A text box fully under a LATER opaque rectangle is not on the rendered
    slide (reveal covers, patches) → not indexed. One slide per control that
    must stay: the cover with `noFill`; 50 % alpha; the rect drawn BEFORE the
    text (under it); a rect covering the text only partially."""
    from lxml import etree
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE

    prs = Presentation()

    def slide(text, *, rect_first=False, rect_w=6.0, fill="solid"):
        s = _blank_slide(prs)
        if not rect_first:
            _textbox(s.shapes, text, x=1.0, y=2.0, w=4.0, h=1.0)
        rect = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.5),
                                  Inches(rect_w), Inches(2.0))
        rect.text_frame.clear()
        if fill == "none":
            rect.fill.background()
        else:
            rect.fill.solid()
            rect.fill.fore_color.rgb = RGBColor(0, 0, 0)
            if fill == "alpha":
                rect._element.spPr.solidFill.srgbClr.append(etree.fromstring(
                    f'<a:alpha xmlns:a="{_A_NS}" val="50000"/>'))
        if rect_first:
            _textbox(s.shapes, text, x=1.0, y=2.0, w=4.0, h=1.0)

    slide("Covered by an opaque rect")
    slide("Cover has no fill", fill="none")
    slide("Cover is half transparent", fill="alpha")
    slide("Rect is underneath", rect_first=True)
    slide("Only partly covered", rect_w=3.0)
    path = out_dir / "occluded_text.pptx"
    prs.save(str(path))
    return CorpusDeck("occluded_text", str(path),
                      expected_texts=["Cover has no fill", "Cover is half transparent",
                                      "Rect is underneath", "Only partly covered"],
                      note="text under a later opaque shape -> not indexed; 4 controls stay")


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
    _build_wrapping_paragraph,
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
    _build_titled_chart_acronym,
    _build_nested_groups,
    _build_table,
    _build_duplicate_text,
    _build_unicode_text,
    _build_image_baked_text,
    _build_image_no_text,
    _build_math_textbox,
    _build_hidden_shapes,
    _build_off_canvas,
    _build_occluded_text,
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
