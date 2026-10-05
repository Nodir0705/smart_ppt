export const meta = {
  name: 'video-draft',
  description: 'Draft a deck narration: one agent per slide writes script/<deck>/slide_NNN.md (existing scripts are kept unless force)',
  whenToUse: 'Jarvis video director, draft step. args = {deck, force?, only?}. Writes the draft packets itself (packets-draft CLI).',
  phases: [
    { title: 'Scout', detail: 'packets-draft CLI: write packets, list slides without a script' },
    { title: 'Draft', detail: 'one agent per slide writes its script' },
  ],
}

// Spec: docs/superpowers/specs/2026-10-06-jarvis-video-director-design.md §2.
// Workflow scripts cannot read files, so the CLI decides which slides need a script
// (its final "todo:" line) and a scout agent relays it.
const ROOT = '/home/netvision/smartppt'
const A = args || {}
if (!A.deck) throw new Error('args.deck is required, e.g. {deck: "AI_Handout_Checker"}')
const DECK = A.deck
const pad = n => String(n).padStart(3, '0')
const ONLY = A.only ? ` --only ${A.only}` : ''

const SCOUT = {
  type: 'object',
  properties: {
    slides: { type: 'array', items: { type: 'integer' } },
    todo: { type: 'array', items: { type: 'integer' } },
    note: { type: 'string' },
  },
  required: ['slides', 'todo', 'note'],
}
const RESULT = {
  type: 'object',
  properties: { sentences: { type: 'integer' }, note: { type: 'string' } },
  required: ['sentences', 'note'],
}

const draftPrompt = n => `You write the spoken narration for slide ${n} of the deck "${DECK}". A text-to-speech voice reads it word for word while a video points at what each sentence mentions. Edit no file except your output.

Inputs:
- Packet: ${ROOT}/out/video/${DECK}/packets/draft/slide_${pad(n)}.json: the slide image path ("png"), the slide's main point ("message"), its units (members, summary, talk = how a presenter refers to each thing, terms = spoken forms of numbers and abbreviations), "relations" (flow = a sequence), the pointable elements (id, role, text, bbox in inches), and the previous and next slides' messages ("prev_message"/"next_message", null at the start/end of the deck).
- LOOK at the slide image (Read the png path).
- Do NOT read ${ROOT}/out/bench/*/presenter*, any trace, adjudication or benchmark file, or ${ROOT}/tasks/lessons.md. Use only the packet and the image.

Write ${ROOT}/script/${DECK}/slide_${pad(n)}.md (create the directory if needed):
- 3 to 6 sentences, one per line, nothing else: no headings, bullets, numbering or comment lines.
- Spoken-ready, because it is read exactly as written: numbers as words ("eight gigabytes", "fifteen to thirty three"), no digits, no symbols (% & / + = # * → ✓ and the like), no markdown, no brackets.
- A natural presenter: paraphrase and explain rather than read the slide aloud, say why it matters, and use transitions ("first", "then", "next", "finally") when walking a sequence. Name the specific things on the slide by their words or meaning, since the video points at whatever each sentence talks about.
- Continuity: connect from the previous slide's message (or open the talk if there is none) and, where it is natural, lead into the next slide's message (or close the talk if there is none). Do not repeat the previous slide's content.

Then check the file: 3 to 6 non-empty lines, and grep -n '[0-9]' on it prints nothing. Fix and re-check if not.
Return sentences (the count) and note (at most 20 words on anything hard about this slide).`

phase('Scout')
const scout = await agent(`Run exactly: cd ${ROOT} && python3 -m tools.video packets-draft ${DECK}${ONLY}
Edit no file. Return slides = every N from its "slide N: ..." lines, todo = the numbers on its final "todo:" line ([] for "todo: none"), note = "" or the error if the command failed.`,
  { label: 'packets-draft', phase: 'Scout', schema: SCOUT, effort: 'low' })
if (!scout || !scout.slides.length) return { deck: DECK, error: (scout && scout.note) || 'scout failed', drafted: [] }

const list = A.force ? scout.slides : scout.todo
const kept = scout.slides.filter(n => !list.includes(n))
if (kept.length) log(`keeping existing scripts (pass force to redraft): slides ${kept.join(', ')}`)

phase('Draft')
const res = await pipeline(list, n => agent(draftPrompt(n), { label: `draft slide ${n}`, phase: 'Draft', schema: RESULT }))
const failed = list.filter((n, i) => !res[i])
if (failed.length) log(`no script written for slides ${failed.join(', ')}`)
return {
  deck: DECK,
  drafted: list.map((n, i) => ({ slide: n, ...(res[i] || { failed: true }) })),
  kept,
}
