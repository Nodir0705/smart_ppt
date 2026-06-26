"""Phase-1 grounding acceptance test (slow, opt-in).

Runs the full TTS->STT->matcher->policy audit against the ground-truth cues and
asserts the three target failures are fixed, no per-slide regression below the
post-Phase-1 floor, overall accuracy holds, and flicker stays modest.

Opt-in: set SMARTPPT_RUN_EVAL=1. It needs soffice, the narration WAVs, and the
scoped index (out/index.scoped.json); skips cleanly when any are absent.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCOPED = ROOT / "out" / "index.scoped.json"
MANIFEST = ROOT / "out" / "narration" / "manifest.json"

_reasons = []
if os.environ.get("SMARTPPT_RUN_EVAL") != "1":
    _reasons.append("set SMARTPPT_RUN_EVAL=1 to run")
if shutil.which("soffice") is None:
    _reasons.append("soffice not installed")
if not SCOPED.exists():
    _reasons.append("out/index.scoped.json missing (run src.group_scope)")
if not MANIFEST.exists():
    _reasons.append("narration manifest missing (run src.tts)")

pytestmark = pytest.mark.skipif(bool(_reasons), reason="; ".join(_reasons))


def test_phase1_grounding_targets_met():
    from tools.timeline_audit import audit

    res = audit(str(SCOPED), normalize_index=True)
    slides = res["slides"]
    overall = res["overall"]

    # The three originally-failing targets must now be highlighted.
    assert "s4_e2" in slides[4]["trajectory"], "STT latency bar never lit"
    assert "s4_e3" in slides[4]["trajectory"], "Match latency bar never lit"
    assert "s6_e1" in slides[6]["trajectory"], "STT Engine node never lit"

    # No per-slide regression below the floor. Re-baselined for the bge-small
    # encoder (2026-06-18): slide 3 drops 9->8 — bge-small misses the marginal
    # s3_e5; recovering it needs gap=0.01, which spikes flicker 0->7 (a worse
    # trade). Overall stays 37/40, flicker 0, all three critical targets met.
    floor = {1: 2, 2: 6, 3: 8, 4: 7, 5: 6, 6: 5, 7: 2}
    for sno, lo in floor.items():
        assert slides[sno]["hits"] >= lo, f"slide {sno} regressed: {slides[sno]['hits']} < {lo}"

    # Headline metric is now three first-class numbers so tuning can't trade one
    # for another silently. Baseline (bge-small + scoped): recall 37/40,
    # precision 37/38 (1 false highlight on s6), flicker 2.
    assert overall["hits"] >= 36, f"recall dropped: {overall['hits']}/40"
    # precision: false highlights are elements the narration never named — recall
    # is blind to them, so guard them explicitly.
    assert overall["false"] <= 2, f"too many false highlights: {overall['false']}"
    assert overall["precision"] >= 0.95, f"precision dropped: {overall['precision']:.3f}"
    assert overall["flicker"] <= 4, f"too much flicker: {overall['flicker']}"
