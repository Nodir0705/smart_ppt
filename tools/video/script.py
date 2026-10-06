"""Narration script files: ``script/<deck>/slide_NNN.md``, one sentence per line,
``#`` lines are comments. ``words`` is THE tokenizer: every step (timings, cue
``at``/``until`` word indices, director packets) indexes words through it.

``norm`` reuses ``src.stt.spoken_text`` (the same normaliser WhistleEngine applies
to Whistle output): lower case, punctuation stripped, numbers as words. It can
return several space-separated tokens ("100%" -> "one hundred percent",
"e-mail" -> "e mail"); ``timings.match`` aligns on those sub-tokens.
Writing script files is left to the drafter agents (no writer here)."""
from __future__ import annotations

from pathlib import Path

from src.stt import spoken_text


def sentences(path: str | Path) -> list[str]:
    lines = (ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines())
    return [ln for ln in lines if ln and not ln.startswith("#")]


def words(sentence: str) -> list[str]:
    return sentence.split()


def norm(word: str) -> str:
    return spoken_text(word)
