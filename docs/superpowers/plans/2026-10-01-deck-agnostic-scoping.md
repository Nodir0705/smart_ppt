# Deck-agnostic Group Scoping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the test-deck-specific `_TITLE_KEYWORDS` whitelist with chart-title-derived group labels, and auto-derive acronym expansions from deck text.

**Architecture:** The parser adds `chart_title` + `series_names` to chart elements (chart `text` untouched). `group_scope.group_label_for_chart` picks a label source (chart title → sole series name → real chart text → slide title → noun fallback), cleans it, and returns `(head, full)`; scoped phrases are emitted for both. A new pure `derive_expansions(index)` regex pass finds `Words (ACR)` / `ACR (Words)` definitions; `expansions_for(index, path)` overlays `data/expansions.json` on top (file wins) and is used by both `build_index` and the `group_scope` CLI.

**Tech Stack:** Python 3, python-pptx, pytest. Offline, deterministic, CPU-only — no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-01-deck-agnostic-scoping-design.md`
**Branch:** `feat/deck-agnostic-scoping` (already created, spec committed).

---

## File map

| File | Change |
|---|---|
| `src/parser.py` | `Element` gains `chart_title`, `series_names`; new `_chart_meta(chart)`; all 3 chart `Element(...)` constructions pass them; `_element_to_dict` emits them additively |
| `src/group_scope.py` | delete `_TITLE_KEYWORDS`; new `_clean_label`, `_head_label`; `group_label_for_chart` returns `(head, full)`; `scope_index` emits phrases for both; new `derive_expansions`, `expansions_for`; CLI uses `expansions_for` |
| `src/build_index.py` | use `expansions_for` instead of `load_expansions` |
| `tools/make_corpus.py` | new `_build_titled_chart_acronym` deck |
| `tests/test_corpus.py` | parser-field tests + end-to-end scoping test on the new deck |
| `tests/test_group_scope.py` | label/expansion unit tests; update 2 existing tests |

---

### Task 0: Record the baseline

**Files:** none modified.

- [ ] **Step 1: Rebuild the scoped index on the current code**

Run: `python3 -m src.build_index test_deck.pptx`
Expected: completes, writes `out/index.scoped.json`.

- [ ] **Step 2: Run the grounding audit and save the output**

Run: `python3 -m tools.timeline_audit --index out/index.scoped.json --normalize | tee /tmp/claude-1000/-home-netvision-smartppt/45ac6273-e22d-4a7a-81cb-ce58102ee7e7/scratchpad/baseline_audit.txt`
Expected (per `tests/test_grounding_eval.py` comment): recall 38/40, 1 false highlight (s6), flicker 2. Record the exact per-slide lines for slide 4 — they are compared in Task 4.

If the numbers differ from 38 / 1 / 2, stop and report: the gate in Task 4 is defined relative to this baseline.

---

### Task 1: Parser emits `chart_title` and `series_names`

**Files:**
- Modify: `src/parser.py` (`Element` ~line 40, `_expand_chart` ~403-488, `_expand_multiseries_chart` ~491-526, `_element_to_dict` ~724)
- Modify: `tools/make_corpus.py` (new builder + `_BUILDERS` entry)
- Test: `tests/test_corpus.py`

- [ ] **Step 1: Add the corpus deck**

In `tools/make_corpus.py`, add after `_build_pie_chart`:

```python
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
```

and add `_build_titled_chart_acronym,` to `_BUILDERS` directly after `_build_pie_chart,`.

- [ ] **Step 2: Write the failing parser tests**

Append to `tests/test_corpus.py`:

```python
def test_chart_title_and_series_names_emitted(tmp_path):
    decks = {d.name: d for d in build_corpus(tmp_path)}
    index = parse_deck(decks["titled_chart_acronym"].path)
    chart = next(e for s in index["slides"] for e in s["elements"] if e["type"] == "chart")
    assert chart["chart_title"] == "Revenue by region (USD)"
    assert chart["series_names"] == ["Revenue"]
    # chart text is still the chart-type token (container embedding unchanged)
    assert chart["text"].startswith("COLUMN_CLUSTERED")


def test_untitled_chart_has_no_chart_title(tmp_path):
    decks = {d.name: d for d in build_corpus(tmp_path)}
    index = parse_deck(decks["pie_chart"].path)
    chart = next(e for s in index["slides"] for e in s["elements"] if e["type"] == "chart")
    assert chart.get("chart_title") is None
    assert chart["series_names"] == ["Share"]


def test_multiseries_chart_lists_every_series(tmp_path):
    decks = {d.name: d for d in build_corpus(tmp_path)}
    index = parse_deck(decks["grouped_bar"].path)
    chart = next(e for s in index["slides"] for e in s["elements"] if e["type"] == "chart")
    assert len(chart["series_names"]) >= 2
```

- [ ] **Step 3: Run them to verify they fail**

Run: `python3 -m pytest tests/test_corpus.py -k "chart_title or series" -v`
Expected: FAIL with `KeyError: 'chart_title'` / `KeyError: 'series_names'`.
(`test_corpus_no_silent_drops` should already PASS with the new deck — run `python3 -m pytest tests/test_corpus.py::test_corpus_no_silent_drops -v` to confirm.)

- [ ] **Step 4: Implement**

In `src/parser.py`, extend `Element` (after `multi_series`):

```python
    chart_title: str | None = None       # chart's own title (group_scope labels)
    series_names: list[str] | None = None  # chart series names (group_scope labels)
```

Add above `_expand_chart`:

```python
def _chart_meta(chart) -> dict:
    """The chart's own title and series names, for deck-agnostic group labels.

    Reads `chart_title.text_frame` only when it already exists (`has_text_frame`)
    — touching `.text_frame` otherwise would create one. A malformed chart part
    yields empty metadata rather than aborting the slide."""
    title = None
    names: list[str] = []
    try:
        if chart.has_title and chart.chart_title.has_text_frame:
            title = chart.chart_title.text_frame.text.strip() or None
        names = [str(se.name) for plot in chart.plots for se in plot.series]
    except Exception:
        pass
    return {"chart_title": title, "series_names": names}
```

In `_expand_chart`, right after `ct_simple = classify_chart_type(chart_type)`:

```python
    meta = _chart_meta(chart)
```

Add `**meta,` to the `Element(...)` call in the unsupported branch (after `needs_vision=True,`) and to `chart_el = Element(...)` (after `trigger_phrases=...`). Change the multi-series call to pass it:

```python
        return _expand_multiseries_chart(slide_no, start_idx, bbox, chart_type,
                                         categories, series_list, sub, meta)
```

and the signature + chart element in `_expand_multiseries_chart`:

```python
def _expand_multiseries_chart(slide_no, start_idx, bbox, chart_type, categories,
                              series_list, sub, meta):
    ...
    chart_el = Element(
        id=f"s{slide_no}_e{start_idx}",
        type="chart",
        text=chart_type,
        bbox=bbox,
        trigger_phrases=_phrases_for_text(chart_type) + ["chart"],
        multi_series=True,
        **meta,
    )
```

In `_element_to_dict`, after the `multi_series` block:

```python
    # additive, chart elements only — non-chart output stays byte-identical
    if e.chart_title is not None:
        d["chart_title"] = e.chart_title
    if e.series_names is not None:
        d["series_names"] = e.series_names
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_corpus.py -v`
Expected: all PASS (including the 3 new ones and `test_corpus_parse_is_deterministic`).

- [ ] **Step 6: Confirm the real deck's titles come through**

Run:
```bash
python3 -c "
from src.parser import parse_deck
ix = parse_deck('test_deck.pptx')
for s in ix['slides']:
    for e in s['elements']:
        if e['type'] == 'chart': print(e['text'], '|', e.get('chart_title'), '|', e['series_names'])
"
```
Expected:
```
COLUMN_CLUSTERED (51) | Latency by stage (ms) | ['Latency (ms)']
PIE (5) | Build effort split | ['Effort']
LINE_MARKERS (65) | None | ['Correct highlights', 'False pointing']
```

- [ ] **Step 7: Commit**

```bash
git add src/parser.py tools/make_corpus.py tests/test_corpus.py
git commit -m "feat(parser): emit chart_title + series_names on chart elements

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Group labels from chart title (drop `_TITLE_KEYWORDS`)

**Files:**
- Modify: `src/group_scope.py` (lines ~47-86 label code; ~210-238 in `scope_index`)
- Test: `tests/test_group_scope.py`

- [ ] **Step 1: Rewrite the label tests (failing)**

In `tests/test_group_scope.py`, replace `test_group_label_from_slide_title` with:

```python
def test_group_label_prefers_chart_title_and_cuts_head():
    chart = {"type": "chart", "text": "COLUMN_CLUSTERED (51)",
             "chart_title": "Latency by stage (ms)", "series_names": ["Latency (ms)"]}
    assert group_label_for_chart(chart, "Where the Time Goes") == ("latency", "latency by stage")


def test_group_label_uses_sole_series_name_when_untitled():
    chart = {"type": "chart", "text": "PIE (5)", "chart_title": None, "series_names": ["Effort"]}
    assert group_label_for_chart(chart, "Where the Time Goes") == ("effort", "effort")


def test_group_label_skips_default_series_name():
    chart = {"type": "chart", "text": "PIE (5)", "chart_title": None, "series_names": ["Series 1"]}
    assert group_label_for_chart(chart, "Where the Time Goes") == (
        "where the time goes", "where the time goes")


def test_group_label_multiseries_falls_back_to_slide_title():
    chart = {"type": "chart", "text": "LINE_MARKERS (65)", "chart_title": None,
             "series_names": ["Correct highlights", "False pointing"]}
    assert group_label_for_chart(chart, "Accuracy vs. Confidence Threshold") == (
        "accuracy", "accuracy vs. confidence threshold")


def test_group_label_noun_fallback():
    chart = {"type": "chart", "text": "COLUMN_CLUSTERED (51)"}
    assert group_label_for_chart(chart, None) == ("bar chart", "bar chart")


def test_group_label_legacy_index_uses_real_chart_text():
    # old indexes have no chart_title field; a non-chart-type text is a title
    chart = {"type": "chart", "text": "Quarterly Revenue"}
    assert group_label_for_chart(chart, "Anything") == ("quarterly revenue", "quarterly revenue")


def test_head_label_does_not_cut_at_of():
    chart = {"type": "chart", "text": "PIE (5)", "chart_title": "Share of revenue"}
    assert group_label_for_chart(chart, None) == ("share of revenue", "share of revenue")
```

Replace `test_scope_assigns_group_id_and_label` with (mirrors the real slide 4):

```python
def test_scope_assigns_group_id_and_label():
    index = {"slides": [{"slide_number": 4, "elements": [
        {"id": "s4_e0", "type": "title", "text": "Where the Time Goes", "bbox": {}, "trigger_phrases": []},
        {"id": "s4_e1", "type": "chart", "text": "COLUMN_CLUSTERED (51)", "bbox": {}, "trigger_phrases": [],
         "parent_id": None, "chart_title": "Latency by stage (ms)", "series_names": ["Latency (ms)"]},
        {"id": "s4_e3", "type": "chart_category", "text": "Match", "bbox": {}, "trigger_phrases": ["match"], "parent_id": "s4_e1"},
        {"id": "s4_e6", "type": "chart", "text": "PIE (5)", "bbox": {}, "trigger_phrases": [],
         "parent_id": None, "chart_title": "Build effort split", "series_names": ["Effort"]},
        {"id": "s4_e8", "type": "chart_category", "text": "Matching", "bbox": {}, "trigger_phrases": ["matching"], "parent_id": "s4_e6"},
    ]}]}
    out = scope_index(index, expansions={})
    e3 = out["slides"][0]["elements"][2]
    assert e3["group_id"] == "s4_e1"
    assert e3["group_label"] == "latency"
    ph3 = e3["trigger_phrases"]
    assert "the match bar" in ph3
    assert "match latency" in ph3
    assert "the match bar in the latency chart" in ph3
    # full-title variant emitted too
    assert "match latency by stage" in ph3
    assert "the match bar in the latency by stage chart" in ph3
    # the pie is labelled by ITS title, no longer by the slide-title keyword
    e8 = out["slides"][0]["elements"][4]
    assert e8["group_id"] == "s4_e6"
    assert e8["group_label"] == "build effort split"
    assert "the matching slice" in e8["trigger_phrases"]
    assert "the matching slice in the build effort split chart" in e8["trigger_phrases"]
    assert not any("latency" in p for p in e8["trigger_phrases"])


def test_scope_head_equals_full_emits_no_duplicates():
    index = {"slides": [{"slide_number": 1, "elements": [
        {"id": "c", "type": "chart", "text": "PIE (5)", "bbox": {}, "trigger_phrases": [],
         "chart_title": "Build effort split"},
        {"id": "k", "type": "chart_category", "text": "Parser", "bbox": {}, "trigger_phrases": [], "parent_id": "c"},
    ]}]}
    ph = scope_index(index, expansions={})["slides"][0]["elements"][1]["trigger_phrases"]
    assert len(ph) == len(set(ph))
    assert ph.count("the parser slice in the build effort split chart") == 1
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python3 -m pytest tests/test_group_scope.py -v`
Expected: the new label tests FAIL (function returns a `str`, e.g. `'latency' != ('latency', 'latency by stage')`); `test_scope_assigns_group_id_and_label` FAILs on the pie label.

- [ ] **Step 3: Implement the label derivation**

In `src/group_scope.py`, move `import re` to the top-level imports (after `import json`) and delete the two function-local `import re` lines (in `group_label_for_chart`, `stat_phrases`, and the acronym block of `scope_index`). Replace the whole `_TITLE_KEYWORDS` + `group_label_for_chart` section with:

```python
_CHART_TYPE_RE = re.compile(r'^[A-Z][A-Z0-9_]+\s*\(\d+\)\s*$')
# connector words that end a title's head noun phrase. "of"/"for" are excluded
# on purpose: "share of revenue" must not shrink to "share".
_CONNECTORS = {"by", "vs", "vs.", "versus", "per", "over", "across"}
# authoring-tool defaults that carry no meaning as a label
_DEFAULT_LABEL_RE = re.compile(r'^(series \d+|chart title)$')


def _clean_label(text: str) -> str:
    """Lowercase, drop parenthesised spans (units like "(ms)"), collapse spaces."""
    text = re.sub(r"\([^)]*\)", " ", text.lower())
    cleaned = " ".join(text.split())
    return "" if _DEFAULT_LABEL_RE.match(cleaned) else cleaned


def _head_label(full: str) -> str:
    """The title's head noun phrase: everything before the first connector."""
    words = full.split()
    for i, w in enumerate(words):
        if w in _CONNECTORS:
            return " ".join(words[:i]) or full
    return full


def group_label_for_chart(chart_elem: dict, slide_title: str | None) -> tuple[str, str]:
    """Return (head, full) spoken labels for a chart group, deck-agnostically.

    Source, first non-empty after cleaning wins:
      1. the chart's own title (``chart_title``, from the parser)
      2. its series name, when it has exactly one series
      3. the chart's ``text`` when it is a real title, not a "PIE (5)" token
         (indexes built before ``chart_title`` existed)
      4. the slide title
    else ``"<noun> chart"``. ``head`` is ``full`` cut at the first connector
    word ("latency by stage" -> "latency").
    """
    text = (chart_elem.get("text") or "").strip()
    series = chart_elem.get("series_names") or []
    candidates = [
        chart_elem.get("chart_title"),
        series[0] if len(series) == 1 else None,
        text if text and not _CHART_TYPE_RE.match(text) else None,
        slide_title,
    ]
    for cand in candidates:
        full = _clean_label(cand or "")
        if full:
            return _head_label(full), full
    fallback = f"{chart_noun(text)} chart"
    return fallback, fallback
```

- [ ] **Step 4: Emit phrases for both labels in `scope_index`**

In `scope_index`, inside `for elem in elements:` right after `phrases: list = elem.setdefault(...)`, add:

```python
            label_variants: list[str] = []
```

Replace the `if parent_id:` branch body with:

```python
            if parent_id:
                elem["group_id"] = parent_id
                parent_elem = elem_by_id.get(parent_id)
                if parent_elem is not None:
                    head, full = group_label_for_chart(parent_elem, slide_title)
                    elem["group_label"] = head
                    label_variants = [head] if full == head else [head, full]
                else:
                    elem["group_label"] = None
```

In the `chart_category` block, replace from `group_label = elem.get("group_label") or ""` through the last `_add_phrase(...)` of that block with:

```python
                if who and noun:
                    _add_phrase(phrases, f"the {who} {noun}")
                for gl in label_variants:
                    # avoid "... bar chart chart" when the label already ends in chart
                    gl_suffix = "" if gl.endswith("chart") else " chart"
                    if who:
                        _add_phrase(phrases, f"{who} {gl}")
                        _add_phrase(phrases, f"the {who} {noun} in the {gl}{gl_suffix}")
```

Update the `scope_index` docstring bullet for `group_label` to: `'group_label': for chart_category, the head label from group_label_for_chart of its parent chart; scoped phrases are added for both the head and the full title.`

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_group_scope.py -v`
Expected: all PASS (including unchanged `test_scope_is_idempotent`, `test_bullets_share_one_group`, `test_stat_phrases_expands_symbol_number_unit`).

- [ ] **Step 6: Confirm no other code calls `group_label_for_chart` or `_TITLE_KEYWORDS`**

Run: `grep -rn "group_label_for_chart\|_TITLE_KEYWORDS" src tools tests`
Expected: only `src/group_scope.py` (definition + the one call in `scope_index`) and `tests/test_group_scope.py`.

- [ ] **Step 7: Run the full suite**

Run: `python3 -m pytest -q`
Expected: all pass (174+ passed, 1 skipped).

- [ ] **Step 8: Commit**

```bash
git add src/group_scope.py tests/test_group_scope.py
git commit -m "feat(scope): deck-agnostic chart group labels from chart title

Drops the test_deck _TITLE_KEYWORDS whitelist (which mislabelled the
slide-4 pie as 'latency'). Labels come from chart title -> sole series
-> slide title -> noun; phrases for both head and full title.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Auto-derive acronym expansions

**Files:**
- Modify: `src/group_scope.py` (new functions after `load_expansions`; `main`)
- Modify: `src/build_index.py:22,74`
- Test: `tests/test_group_scope.py`, `tests/test_corpus.py`

- [ ] **Step 1: Write the failing unit tests**

In `tests/test_group_scope.py` change the import line to:

```python
from src.group_scope import (chart_noun, derive_expansions, expansions_for,
                             group_label_for_chart, scope_index)
```

and append:

```python
def _ix(*texts):
    return {"slides": [{"slide_number": 1, "elements": [
        {"id": f"e{i}", "type": "body", "text": t, "bbox": {}, "trigger_phrases": []}
        for i, t in enumerate(texts)]}]}


def test_derive_words_then_acronym_with_stopword():
    assert derive_expansions(_ix("Speech to Text (STT) runs offline")) == {"STT": "speech to text"}


def test_derive_skips_stopwords_and_leading_noise():
    ix = _ix("data from the Bureau of Labor Statistics (BLS)")
    assert derive_expansions(ix) == {"BLS": "bureau of labor statistics"}


def test_derive_acronym_then_words():
    assert derive_expansions(_ix("ROI (Return on Investment) doubled")) == {"ROI": "return on investment"}


def test_derive_rejects_initials_mismatch():
    assert derive_expansions(_ix("Machine Learning (API)")) == {}


def test_derive_ignores_chart_type_tokens():
    assert derive_expansions(_ix("COLUMN_CLUSTERED (51)", "PIE (5)")) == {}


def test_derive_drops_conflicting_definitions():
    ix = _ix("Speech to Text (STT)", "Single Track Transfer (STT)")
    assert derive_expansions(ix) == {}


def test_derive_same_definition_twice_is_kept():
    ix = _ix("Speech to Text (STT)", "the Speech to Text (STT) stage")
    assert derive_expansions(ix) == {"STT": "speech to text"}


def test_expansions_for_file_overrides_win(tmp_path):
    f = tmp_path / "exp.json"
    f.write_text('{"stt": "speech-to-text"}')
    ix = _ix("Speech to Text (STT)", "Bureau of Labor Statistics (BLS)")
    assert expansions_for(ix, str(f)) == {"STT": "speech-to-text",
                                          "BLS": "bureau of labor statistics"}


def test_expansions_for_missing_file_is_derived_only(tmp_path):
    ix = _ix("Speech to Text (STT)")
    assert expansions_for(ix, str(tmp_path / "nope.json")) == {"STT": "speech to text"}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python3 -m pytest tests/test_group_scope.py -v`
Expected: collection ERROR — `ImportError: cannot import name 'derive_expansions'`.

- [ ] **Step 3: Implement**

In `src/group_scope.py`, after `load_expansions`, add:

```python
# ---------------------------------------------------------------------------
# derive_expansions  — "Speech to Text (STT)" / "STT (Speech to Text)"
# ---------------------------------------------------------------------------

_ACR_IN_PARENS_RE = re.compile(r"\(([A-Z]{2,5})\)")
_ACR_THEN_PARENS_RE = re.compile(r"\b([A-Z]{2,5})\b\s*\(([^()]+)\)")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
_EXPANSION_STOPWORDS = {"to", "of", "and", "the", "for"}


def _initials_spell(words: list[str], acr: str) -> bool:
    """True if the words' initials spell `acr` — counting every word, or only
    the non-stopwords ("Bureau of Labor Statistics" -> BLS)."""
    if "".join(w[0] for w in words).upper() == acr:
        return True
    content = [w for w in words if w.lower() not in _EXPANSION_STOPWORDS]
    return bool(content) and "".join(w[0] for w in content).upper() == acr


def _expansion_before(words: list[str], acr: str) -> str | None:
    """Shortest run of trailing `words` whose initials spell `acr`."""
    for n in range(len(acr), min(len(words), len(acr) + 3) + 1):
        tail = words[-n:]
        if _initials_spell(tail, acr):
            return " ".join(tail).lower()
    return None


def derive_expansions(index: dict) -> dict[str, str]:
    """Acronym -> expansion pairs the deck defines itself, deterministically.

    Recognises "Words (ACR)" and "ACR (Words)" in any element's text, gated by an
    initials check. An acronym defined two different ways is dropped as
    ambiguous (same rule as asr_normalize.build_alias_map).
    """
    found: dict[str, set[str]] = {}
    for slide in index.get("slides", []):
        for elem in slide.get("elements", []):
            text = elem.get("text") or ""
            for m in _ACR_IN_PARENS_RE.finditer(text):
                acr = m.group(1)
                exp = _expansion_before(_WORD_RE.findall(text[:m.start()]), acr)
                if exp:
                    found.setdefault(acr, set()).add(exp)
            for m in _ACR_THEN_PARENS_RE.finditer(text):
                acr, words = m.group(1), _WORD_RE.findall(m.group(2))
                if len(words) <= len(acr) + 3 and _initials_spell(words, acr):
                    found.setdefault(acr, set()).add(" ".join(words).lower())
    return {acr: next(iter(exps)) for acr, exps in sorted(found.items()) if len(exps) == 1}


def expansions_for(index: dict, path: str = EXPANSIONS_PATH) -> dict[str, str]:
    """Deck-derived expansions overlaid with the hand-override file (file wins)."""
    return {**derive_expansions(index), **load_expansions(path)}
```

In `main` (after `index = json.load(fh)`), replace `expansions = load_expansions(args.expansions)` with:

```python
    expansions = expansions_for(index, args.expansions)
```

In `src/build_index.py` change the import on line 22 to:

```python
from .group_scope import EXPANSIONS_PATH, expansions_for, scope_index
```

and line 74 to:

```python
        final = scope_index(final, expansions=expansions_for(final, expansions_path))
```

- [ ] **Step 4: Run the unit tests to verify they pass**

Run: `python3 -m pytest tests/test_group_scope.py -v`
Expected: all PASS.

- [ ] **Step 5: Write the end-to-end corpus test**

Append to `tests/test_corpus.py` (add `from src.group_scope import expansions_for, scope_index` to the imports):

```python
def test_titled_chart_acronym_scopes_without_deck_vocab(tmp_path):
    """Group label comes from the chart's own title and STT's expansion from the
    deck's own 'Speech to Text (STT)' — no test_deck vocabulary involved."""
    decks = {d.name: d for d in build_corpus(tmp_path)}
    index = parse_deck(decks["titled_chart_acronym"].path)
    out = scope_index(index, expansions=expansions_for(index, str(tmp_path / "none.json")))
    els = {e.get("text"): e for s in out["slides"] for e in s["elements"]}
    north = els["North"]
    assert north["group_label"] == "revenue"
    assert "the north bar in the revenue chart" in north["trigger_phrases"]
    assert "the north bar in the revenue by region chart" in north["trigger_phrases"]
    engine = els["STT Engine"]["trigger_phrases"]
    assert "speech to text" in engine
    assert "speech to text engine" in engine
```

- [ ] **Step 6: Run it to verify it passes**

Run: `python3 -m pytest tests/test_corpus.py::test_titled_chart_acronym_scopes_without_deck_vocab -v`
Expected: PASS. (If `els["STT Engine"]` KeyErrors, print `list(els)` — the parser may classify/merge the textbox differently; fix the lookup, not the parser.)

- [ ] **Step 7: Full suite**

Run: `python3 -m pytest -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/group_scope.py src/build_index.py tests/test_group_scope.py tests/test_corpus.py
git commit -m "feat(scope): auto-derive acronym expansions from deck text

'Words (ACR)' / 'ACR (Words)' with an initials check; conflicting
definitions dropped; data/expansions.json still wins.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Ship gate on test_deck + record results

**Files:**
- Modify: `tasks/todo.md` (append review section), `tasks/lessons.md` (only if something surprising happened)

- [ ] **Step 1: Rebuild and inspect the new labels**

Run:
```bash
python3 -m src.build_index test_deck.pptx
python3 -c "
import json
ix = json.load(open('out/index.scoped.json'))
for s in ix['slides']:
    for e in s['elements']:
        if e['type'] == 'chart_category': print(e['id'], e['text'], '|', e['group_label'])
"
```
Expected: slide-4 bar categories `| latency`, slide-4 pie slices `| build effort split`, slide-5 points `| accuracy`.

- [ ] **Step 2: Run the audit and compare against the Task 0 baseline**

Run: `python3 -m tools.timeline_audit --index out/index.scoped.json --normalize | tee /tmp/claude-1000/-home-netvision-smartppt/45ac6273-e22d-4a7a-81cb-ce58102ee7e7/scratchpad/after_audit.txt`
then: `diff /tmp/claude-1000/-home-netvision-smartppt/45ac6273-e22d-4a7a-81cb-ce58102ee7e7/scratchpad/baseline_audit.txt /tmp/claude-1000/-home-netvision-smartppt/45ac6273-e22d-4a7a-81cb-ce58102ee7e7/scratchpad/after_audit.txt`

**Gate (all must hold):** recall ≥ 38/40, flicker ≤ 2, false highlights ≤ 1. Check slide 4 explicitly: s4_e3 (Match bar) and s4_e5 must be no worse than baseline.

If the gate fails: do NOT tune `_CONNECTORS` or add test_deck words to make it pass — that recreates the whitelist. Instead instrument slide 4 (print the live tails and top-2 scores per update, as in the 2026-06-27 lesson), report the cause, and stop for a decision.

- [ ] **Step 3: Run the opt-in eval test**

Run: `SMARTPPT_RUN_EVAL=1 python3 -m pytest tests/test_grounding_eval.py -v`
Expected: PASS.

- [ ] **Step 4: Full suite**

Run: `python3 -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Record results**

Append to `tasks/todo.md`:

```markdown
## Deck-agnostic group scoping — roadmap #9 (2026-10-01)

**Status:** complete. Spec `docs/superpowers/specs/2026-10-01-deck-agnostic-scoping-design.md`,
plan `docs/superpowers/plans/2026-10-01-deck-agnostic-scoping.md`.

- [x] parser emits `chart_title` / `series_names` (chart `text` unchanged)
- [x] `_TITLE_KEYWORDS` deleted; labels from chart title → sole series → slide title → noun; phrases for head + full
- [x] `derive_expansions` + `expansions_for` (file overrides win), wired in build_index + group_scope CLI
- [x] corpus deck `titled_chart_acronym`

**Eval (test_deck):** baseline recall 38/40 · false 1 · flicker 2 → after: <fill from after_audit.txt>.
Slide-4 pie relabelled latency → build effort split.
```

Replace `<fill from after_audit.txt>` with the actual numbers before committing.

- [ ] **Step 6: Commit**

```bash
git add tasks/todo.md
git commit -m "docs(todo): record deck-agnostic scoping results

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
