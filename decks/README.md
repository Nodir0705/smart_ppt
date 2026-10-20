Drop `.pptx` decks here, then run `python3 -m tools.deck_bench` (non-recursive: only `decks/*.pptx`).
Everything in this folder is git-ignored except this README and `holdout/SOURCES.md`.
`holdout/` is the frozen unseen test set (provenance in `holdout/SOURCES.md`); bench it with
`--decks decks/holdout`. Report: `out/bench/report_<variant>.md`.
