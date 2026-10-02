# Slide understanding map (approach A: map-driven matching) — design

**Date:** 2026-10-02 · **Branch:** `feat/deck-bench`

## Problem

Live highlighting follows keywords: speech is matched against the literal text
of each box. Under natural presenter speech this lights related-but-wrong boxes,
misses paraphrases, and can't use slide structure (cards, flows, captions). The
user wants each slide *understood* first — every item's role, which items form
one thing, how items connect, and how a presenter talks about them — and the
live highlight to follow that understanding.

## Decisions

- **Map author:** Claude Code agents, one agent per slide, at deck-preparation
  time (prototype; later the same map format can come from the Claude API or a
  local model). The live loop stays offline and CPU-only.
- **Approach A:** the map feeds the existing single live path (`MatchPolicy`);
  no new live algorithm. Decks without a map behave exactly as today.
- **Evaluation:** Claude Code also writes the talks; map agents never see any
  talk script and talk agents never see any map.

## 1. Map format — `out/maps/<deck_stem>/slide_NNN.json`

```json
{
  "slide": 4,
  "message": "one sentence: the slide's main point",
  "items": {"s4_e3": {"role": "card_heading", "pointable": true}, ...},
  "units": [{"uid": "u1", "members": ["s4_e2", "s4_e3", "s4_e5"],
             "summary": "Stage A reads the page with a vision-language model",
             "talk": ["the reading stage", "how we turn pixels into text", ...],
             "terms": ["stage A", "O C R"]}],
  "relations": [{"from": "u1", "to": "u2", "type": "flow"}],
  "order": ["u1", "u2", "u3"]
}
```

- `items` covers every element id on the slide. Roles: `title`,
  `section_label`, `card_heading`, `body`, `bullet`, `stat`, `caption`,
  `diagram_node`, `connector`, `table_cell`, `chart_bar`, `photo`, `icon`,
  `decoration`, `footer`, `code`.
- Every pointable item belongs to exactly one unit. Non-pointable items belong
  to none. Chart bars and table cells stay single-member units unless the agent
  sees a stronger grouping, such as a whole table row being one point.
- `talk`: 5–10 natural spoken references, as plain lowercase words.
  `terms`: spoken forms of numbers and identifiers.
- Relation types: `flow`, `part_of`, `explains`, `contrasts`, `same_row`,
  `same_col`. `order` lists every uid once.
- `tools/slide_map.py validate` checks all of the above against the slide's
  packet. An invalid map is retried once by its agent; if it is still invalid,
  that slide falls back to today's behaviour.

## 2. Generation

- `tools/slide_map.py packets <deck_dir> <map_dir>` writes one packet per slide:
  `{deck, slide, slide_size, png (absolute path), elements: [{id, type, text, bbox}]}`.
  The packet is built from the deck's built index and its rendered slide PNGs.
- A Claude Code workflow runs one agent per slide. Each agent reads the packet
  and LOOKS at the PNG, writes the map, and runs the validator before finishing.
- Decks: test_deck (7 slides), AI_Handout_Checker (12), Open-Q 845 (31).
- Agents are given no talk script, no adjudication, no benchmark result.

## 3. Applying the map (index build)

- New stage `src/slide_map.py: apply_slide_maps(index, map_dir)`. It runs in
  `build_index` after vision_link and before scope, behind a new
  `slide_map_dir=None` kwarg.
- For each element on a slide that has a map:
  - set `map_role`, `pointable`, and `map_unit` (its uid);
  - append the unit's `talk` and `terms` to the trigger phrases of the unit's
    first member (units compete as one, so one anchor is enough);
  - store the map's `message`, `relations` and `order` on the slide as
    `slide["map"]`.
- Bench variant `"map"` in `tools/deck_bench.py`:
  `{"slide_map_dir": out/maps/<deck_stem>}`, built into `out/bench/<deck>/map/`.

## 4. Live use (`src/matcher.py`, single path)

- `furniture_ids` also returns every element with `pointable: false`.
- `pointable_units`: on a slide where elements carry `map_unit`, units come from
  the map and the unit id is the first member in reading order. Other slides
  keep the heuristic rules.
- Flow hint: `PolicyConfig.flow_bonus` (default 0.03; `0` disables it) is added
  to the effective score of units that are a `flow` successor of the active
  unit. The existing same-group and stay bonuses apply unchanged.

## 5. Evaluation

- **New talk scripts:** 2 styles per deck, 6 scripts, written by agents blind to
  the maps, in the existing annotated format, as `presenter_script_<style>.json`:
  - `jumpy`: out of reading order, returns to earlier points;
  - `meaning`: refers to items by role or meaning ("the second stage", "that
    number on the right", "the photo").
- `tools/presenter_bench.py` gains `--variant` (which index directory to score)
  and `--script` (which script file).
- Compare `default` against `map` on every script for both AI decks, on lit,
  adjudicated false seconds, switches per step and lateness. For test_deck, use
  its new scripts plus the hand fixture.
- **Ship rule:** the `map` variant must not be worse on any deck/script on
  lit (−1 allowed) or adjudicated false seconds, and must be better on the total
  of lit plus switches per step across all scripts.
  - Hand fixture with the test_deck map: ≥ 38/40, false ≤ 1, flicker ≤ 2.
  - Filler holdout: ≤ 5.
  - Verbatim bench: reported, does not block.
- **Adjudication:** new false highlights on the new scripts are adjudicated by
  an agent (same tolerated/wrong rubric as before) before scoring.

## Out of scope

The story tracker (approach B), live use of the map's `order`, and producing
maps through the Claude API or a local model. These come after this prototype
proves the value.
