# Deck-agnostic group scoping & acronym expansions — design

**Date:** 2026-10-01 · **Roadmap item:** #9 in `docs/improvement-research-2026-06-24.md`
**Branch:** `feat/deck-agnostic-scoping` (off `master` after merging `feat/lexical-fusion`)
**Constraint:** offline, deterministic, CPU-only, no neural model.

## Problem

`src/group_scope.py` labels chart groups through `_TITLE_KEYWORDS`, a whitelist of
`test_deck.pptx`'s slide-title words (latency/time, effort/build, accuracy/threshold).
Any other deck falls through to a generic `"<noun> chart"`. The whitelist is also
wrong on the test deck itself:

| Chart (test_deck) | Real chart title in pptx | Current `group_label` |
|---|---|---|
| slide 4, COLUMN_CLUSTERED | "Latency by stage (ms)" | latency (by luck — slide title contains "time") |
| slide 4, PIE | "Build effort split" | **latency** (wrong — same slide-title keyword) |
| slide 5, LINE_MARKERS | none; series "Correct highlights", "False pointing" | accuracy (slide title) |

Root cause: the parser discards real chart titles (`src/parser.py` ~411 sets chart
`text = str(chart.chart_type)`, e.g. `"COLUMN_CLUSTERED (51)"`).

Separately, acronym expansions are hand-authored only (`data/expansions.json`, one
entry: `STT → speech to text`); nothing derives them from deck text.

## Decisions

- Chart titles reach `group_scope` as **new fields** on the chart element; chart
  `text`, `trigger_phrases` and bboxes are unchanged (container embedding unchanged).
- Group label = **short head label**; scoped phrases are emitted for **both** the head
  label and the full cleaned title.
- Expansions are **auto-derived** from deck text in this same change; the JSON file
  stays as an override that always wins.

## 1. Parser — `src/parser.py`

For every chart element (single-series, clustered, stacked, unsupported paths alike):

- `chart_title: str | None` — `chart.chart_title.text_frame.text` when
  `chart.has_title`, else `None`. Empty/whitespace → `None`.
- `series_names: list[str]` — `[s.name for plot in chart.plots for s in plot.series]`.

Serialized alongside existing element fields. No other parser behaviour changes.

## 2. Group labels — `src/group_scope.py`

Delete `_TITLE_KEYWORDS`. `group_label_for_chart(chart_elem, slide_title)` returns
`(head, full)`:

1. **Source** (first non-empty wins): `chart_title` → the series name if
   `len(series_names) == 1` → `slide_title` → `f"{chart_noun(text)} chart"`.
   The existing "chart text is a real title, not a `CHART_TYPE (n)` token" rule is
   kept as a step before the slide title, for back-compat with old indexes that lack
   `chart_title`.
2. **Clean:** lowercase, remove parenthesised spans, collapse whitespace, strip. Authoring-tool defaults ("Series N", "Chart Title") clean to empty and fall through to the next source.
3. **Head:** cut the cleaned text at the first whole-word connector in
   `{by, vs, vs., versus, per, over, across}`. `of` / `for` are excluded
   ("share of revenue" must not become "share"). Empty head → head = full.

`elem["group_label"] = head`. Existing `group_label_for_chart` tests are updated for the tuple return. For each `chart_category`, the three existing scoped
templates are emitted for `head`, then again for `full` when `full != head`
(`_add_phrase` de-dups). Bullet groups (`"bullets"`) are untouched.

Expected test_deck outcome: bar → `latency` (+ `latency by stage` phrases);
pie → `build effort split` (was `latency`); line → `accuracy`.

## 3. Acronym expansions

New pure function `derive_expansions(index) -> dict[str, str]` in `group_scope.py`,
regex over every element's `text`:

- **`Words (ACR)`** — ACR = 2–5 uppercase letters. Take the shortest run of words
  immediately before the parenthesis whose initials (case-insensitive) spell ACR;
  try counting all words first, then skipping stopwords `{to, of, and, the, for}`.
  At most `len(ACR) + 3` words considered.
- **`ACR (Words)`** — same initials check on the words inside the parenthesis.
- Expansion stored lowercased. An acronym yielding two different expansions anywhere
  in the deck is **dropped** (mirrors `asr_normalize.build_alias_map` collision rule).
- Iteration is deterministic (slides/elements in index order; result sorted keys).

Merge: `{**derive_expansions(index), **load_expansions(path)}` — file overrides win.
Applied at both call sites: `src/build_index.py:74` and `group_scope.main`.
On `test_deck.pptx` derivation finds nothing (no `Words (ACR)` text), so the STT
override still carries s6_e1.

## 4. Testing & acceptance

Unit tests (`tests/test_group_scope.py`):
- label source order: chart_title / single series / multi-series → slide title / noun fallback
- cleaning + head cut: "Latency by stage (ms)" → head `latency`, full `latency by stage`;
  "Accuracy vs. Confidence Threshold" → `accuracy`; "Share of revenue" uncut
- phrases emitted for head and full; no duplicates when equal; idempotence still holds
- expansions: both forms; stopword-skipping ("Speech to Text (STT)"); initials
  mismatch rejected ("Machine Learning (API)"); collision dropped; file override wins

Parser test: a titled chart yields `chart_title` and `series_names`; untitled → `None`.

Corpus (`tools/make_corpus.py`): one new deck with a titled chart and an
`Words (ACR)` phrase in body text; `tests/test_corpus.py` asserts the derived
group label and expansion.

**Ship gate** — `python3 -m tools.timeline_audit --index out/index.scoped.json --normalize`
on test_deck after a rebuild: recall ≥ 38/40, flicker ≤ 2, precision no worse
(1 pre-existing false on s6). s4_e3 (Match bar vs pie "Matching") and s4_e5 checked
explicitly since the pie relabel touches slide 4. Full `pytest` green.

## Out of scope

Speaker-notes as an expansion source; axis titles as a label source; tuning the
connector list beyond the set above.
