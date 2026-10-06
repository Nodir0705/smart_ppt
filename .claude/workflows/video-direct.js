export const meta = {
  name: 'video-direct',
  description: 'Direct the HUD effects: one agent per slide whose script or map changed writes out/video/<deck>/cues/slide_NNN.json and validates it',
  whenToUse: 'Jarvis video director, direct step (after the scripts exist). args = {deck, only?}. Writes the direct packets itself (packets-direct CLI).',
  phases: [
    { title: 'Scout', detail: 'packets-direct CLI: write packets, list slides whose cue sheet hash differs' },
    { title: 'Direct', detail: 'one agent per changed slide writes + validates its cue sheet' },
  ],
}

// Spec: docs/superpowers/specs/2026-10-06-jarvis-video-director-design.md §4-5.
// Workflow scripts cannot read files, so the CLI decides which slides are stale
// (its final "todo:" line) and a scout agent relays it.
const ROOT = '/home/netvision/smartppt'
const A = args || {}
if (!A.deck) throw new Error('args.deck is required, e.g. {deck: "AI_Handout_Checker"}')
const DECK = A.deck
const pad = n => String(n).padStart(3, '0')
const ONLY = A.only ? ` --only ${A.only}` : ''

// Spec §5 + the 2026-10-06 polish (tools/video/packets.py GUIDANCE; a test checks they match).
const GUIDANCE = 'focus on what each sentence is about, with spotlight on sentences that single one thing out. Open on the subject: the first effect lands on the first sentence that refers to something. Underline the specific line when a sentence quotes or paraphrases one line of a multi-line unit. Ring a stat or number on the word where it is spoken, but only a compact target (at most 4:1 and a quarter of the slide wide); a wide or sentence-long target gets an underline instead. Glide when the narration walks a flow. Zoom when the sentence describes a detail in a photo or text too small to read. Leave slide intros/outros and pure explanation unpointed. Prefer fewer, well-timed effects to many.'

const SCOUT = {
  type: 'object',
  properties: { todo: { type: 'array', items: { type: 'integer' } }, note: { type: 'string' } },
  required: ['todo', 'note'],
}
const RESULT = {
  type: 'object',
  properties: {
    valid: { type: 'boolean' },
    errors: { type: 'array', items: { type: 'string' } },
    cues: { type: 'integer' },
    note: { type: 'string' },
  },
  required: ['valid', 'errors', 'cues', 'note'],
}

const directPrompt = n => `You direct the HUD effects for slide ${n} of the deck "${DECK}": a narrated video in which corner brackets, a spotlight, underlines, rings and a zoom lens point at slide content on exact spoken words. Edit no file except your output cue sheet.

Inputs:
- Packet: ${ROOT}/out/video/${DECK}/packets/direct/slide_${pad(n)}.json: "png" (the slide image), "map" (message; units with members, summary, talk, terms; relations; order), "elements" (the pointable ones: id, role, text, bbox [x, y, w, h] in inches, lines as [line index, text], value), "sentences" (the narration, already tokenized: sentence s is a list of [word index, word]; use these indices exactly, never count words yourself), "vocabulary" (cue schema, effects, default ends and validator rules: read all of it) and "hash".
- LOOK at the slide image (Read the png path).
- Do NOT read presenter scripts (${ROOT}/out/bench/*/presenter*), traces, adjudications, benchmarks or ${ROOT}/tasks/lessons.md.

Write ${ROOT}/out/video/${DECK}/cues/slide_${pad(n)}.json (create the directory if needed):
{"slide": ${n}, "version": 1, "hash": "<the packet's hash, copied exactly>", "cues": [...]}
Each cue is {"effect", "target", "at": {"s", "w"}} plus the optional "until", "line", "spotlight", "region" described in vocabulary.fields. "cues": [] is right for a slide with nothing worth pointing at.

Guidance (design spec §5, verbatim):
${GUIDANCE}

Anchor "at" on the word that names the thing (its noun or number). Every cue must pass the why rule: when the anchor sentence shares no content word with the target's text, talk or terms (and is not a transition to the next flow unit), choose another target or drop the cue.

Then run: cd ${ROOT} && python3 -m tools.video validate ${DECK} --only ${n}
It must print the line "slide ${n}: VALID" (a "note: timings missing" or "stale" line is fine: the voice step may not have run yet). If it prints errors, fix the sheet and re-run; at most 2 repair rounds, then stop. A "warning: dead air" line is not an error, but act on it: move the first cue onto the sentence it names (or say in note why that sentence stays unpointed).

Return valid (from the last validator run), errors (its last error lines, [] when valid), cues (the count) and note (at most 25 words).`

phase('Scout')
const scout = await agent(`Run exactly: cd ${ROOT} && python3 -m tools.video packets-direct ${DECK}${ONLY}
Edit no file. Return todo = the numbers on its final "todo:" line ([] for "todo: none"), note = "" or the error if the command failed.`,
  { label: 'packets-direct', phase: 'Scout', schema: SCOUT, effort: 'low' })
if (!scout) return { deck: DECK, error: 'scout failed', directed: [] }
if (scout.note) log(`packets-direct: ${scout.note}`)
if (!scout.todo.length) {
  log('every cue sheet matches its packet hash: nothing to direct')
  return { deck: DECK, directed: [] }
}

phase('Direct')
const res = await pipeline(scout.todo, n => agent(directPrompt(n), { label: `direct slide ${n}`, phase: 'Direct', schema: RESULT }))
const bad = scout.todo.filter((n, i) => !(res[i] && res[i].valid))
if (bad.length) log(`still invalid after repair (render drops the failing cues): slides ${bad.join(', ')}`)
return {
  deck: DECK,
  directed: scout.todo.map((n, i) => ({ slide: n, ...(res[i] || { valid: false, note: 'agent failed' }) })),
}
