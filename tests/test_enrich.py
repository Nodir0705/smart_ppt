"""Pure-Python tests for the enricher parsing helpers.

The model-driven enrichment itself is too slow / non-deterministic to test
in the suite (loads multi-GB transformer weights and runs generation), but
the answer sanitation and phrase assembly are deterministic and worth
pinning down — they're where most breakage will land if a model produces
unexpected output.
"""

from __future__ import annotations

import pytest

from src.enrich_llm import _clean_phrases as _llm_clean
from src.enrich_vlm import (
    _extract_chart_type,
    _is_none,
    _phrases_for_chart,
    _phrases_for_image,
    _sanitize,
)


# ---------- LLM trigger paraphrase cleaner ----------

class TestLLMCleanPhrases:
    def test_strips_numbering_and_bullets(self):
        raw = "1. the matcher box\n* speech to text engine\n- audio path"
        out = _llm_clean(raw, original_text="something else")
        assert "the matcher box" in out
        assert "speech to text engine" in out
        assert "audio path" in out

    def test_drops_phrases_matching_original(self):
        raw = "the matcher\nthe little matcher box\nrenders the highlight"
        out = _llm_clean(raw, original_text="the matcher")
        assert "the matcher" not in out
        assert "the little matcher box" in out

    def test_dedupes_case_insensitive(self):
        raw = "The Matcher Box\nthe matcher box\nTHE MATCHER BOX"
        out = _llm_clean(raw, original_text="x")
        assert out == ["the matcher box"]

    def test_drops_too_long_or_too_short(self):
        raw = "ok\nthe matcher\n" + ("very long " * 10).strip()
        out = _llm_clean(raw, original_text="x")
        assert "ok" not in out             # one word
        assert "the matcher" in out
        assert all(len(p.split()) <= 12 for p in out)

    def test_drops_imperative_openers(self):
        raw = (
            "show me the matcher\n"
            "highlight as primary box\n"
            "this is the parser stage\n"
            "the actual parsing step\n"
        )
        out = _llm_clean(raw, original_text="parsing")
        assert "the actual parsing step" in out
        assert not any(p.startswith("show me") for p in out)
        assert not any(p.startswith("highlight as") for p in out)
        assert not any(p.startswith("this is the") for p in out)


# ---------- VLM answer sanitation ----------

class TestSanitize:
    @pytest.mark.parametrize("raw,expected", [
        ('"Latency by stage"', "Latency by stage"),
        ("Latency by stage.", "Latency by stage"),
        ("  the matcher  ", "the matcher"),
        ("`STT`", "STT"),
        ("multiple   spaces", "multiple spaces"),
    ])
    def test_strips_punct_and_whitespace(self, raw, expected):
        assert _sanitize(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("none", True),
        ("None", True),
        ("N/A", True),
        ("no title", True),
        ("", True),
        ("Latency by stage", False),
        ("STT", False),
    ])
    def test_is_none(self, raw, expected):
        assert _is_none(raw) == expected


# ---------- chart type detection ----------

class TestExtractChartType:
    @pytest.mark.parametrize("answer,expected", [
        ("Bar", "bar"),
        ("bar chart", "bar"),
        ("Column", "bar"),       # column → bar (same vocabulary)
        ("This is a pie", "pie"),
        ("Donut chart", "pie"),  # donut → pie
        ("LINE", "line"),
        ("Line chart with markers", "line"),
        ("Area", "line"),
        ("scatter plot", "scatter"),
    ])
    def test_normalizes_common_chart_words(self, answer, expected):
        assert _extract_chart_type(answer) == expected

    def test_unknown_returns_none(self):
        assert _extract_chart_type("rainbow") is None
        assert _extract_chart_type("") is None


# ---------- phrase assembly ----------

class TestPhrasesForChart:
    def test_full_chart_info(self):
        # max_label exists but label-specific phrases are NOT generated at the
        # chart level — they live on the chart_category element (handled by
        # the parser). Only generic superlatives + chart-identifying phrases
        # appear here.
        out = _phrases_for_chart("bar", "Latency by stage (ms)", "STT")
        assert "the bar chart" in out
        assert "the bar" in out
        assert "latency by stage (ms)" in out
        assert "the latency by stage (ms) chart" in out
        assert "the largest bar" in out
        assert "the biggest bar" in out
        assert "the tallest bar" in out
        # label-specific phrase intentionally absent on chart-level
        assert "the stt bar" not in out

    def test_pie_uses_slice_vocabulary(self):
        out = _phrases_for_chart("pie", "Build effort split", "Parsing")
        assert "the pie chart" in out
        assert "the largest slice" in out
        assert "the biggest slice" in out
        # label-specific stays off the chart element
        assert "the parsing slice" not in out
        # no "tallest" for a pie
        assert not any("tallest" in p for p in out)

    def test_line_uses_point_vocabulary(self):
        out = _phrases_for_chart("line", "Accuracy vs threshold", "0.5")
        assert "the line chart" in out
        assert "the highest point" in out
        assert "the largest point" in out
        # label-specific belongs on chart_category, not chart
        assert "the 0.5 point" not in out
        assert not any("bar" in p for p in out)
        assert not any("slice" in p for p in out)

    def test_missing_title_still_yields_type_phrases(self):
        out = _phrases_for_chart("line", None, None)
        assert "the line chart" in out
        assert "the line" in out

    def test_generic_superlatives_without_max_label(self):
        # the VLM may fail to name the largest category — generic superlatives
        # should still be generated from the chart type alone
        out = _phrases_for_chart("pie", "Build effort", None)
        assert "the largest slice" in out
        assert "the biggest slice" in out
        # label-specific phrases should NOT be present
        assert not any(p.endswith("slice") and p not in
                       ("the largest slice", "the biggest slice", "the biggest slice")
                       for p in out)

    def test_no_duplicates(self):
        out = _phrases_for_chart("bar", "Bar Chart", "STT")
        # both "bar" type and "bar" in title risk duplicates — ensure unique
        assert len(out) == len(set(out))


class TestPhrasesForImage:
    def test_baseline_navigational_always_present(self):
        out = _phrases_for_image(None, None)
        for p in ("the image", "the picture", "the figure", "the visual"):
            assert p in out

    def test_kind_strips_leading_article(self):
        # "an icon" → "the icon" (not "the an icon")
        out = _phrases_for_image("an icon", None)
        assert "the icon" in out
        assert "the icon on the slide" in out
        assert "the an icon" not in out

    def test_kind_without_article_still_works(self):
        out = _phrases_for_image("photo", None)
        assert "the photo" in out
        assert "the photo on the slide" in out

    def test_description_lowercased(self):
        out = _phrases_for_image(None, "A Triangle Inside A Circle")
        assert "a triangle inside a circle" in out

    def test_no_duplicates(self):
        out = _phrases_for_image("the photo", "the photo")
        assert len(out) == len(set(out))
