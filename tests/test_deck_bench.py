import json
import os
import shutil

import pytest

import re

from src.matcher import NULL_SPEECH
from tools.deck_bench import FILLER_HOLDOUT, FILLER_TUNE, filler_script, inventory, totals


def test_inventory_counts():
    ix = {"slides": [
        {"slide_number": 1, "elements": [
            {"id": "a", "type": "title", "text": "T", "needs_vision": False},
            {"id": "b", "type": "image", "text": "alt", "needs_vision": False},
            {"id": "c", "type": "image", "text": None, "needs_vision": True}]},
        {"slide_number": 2, "elements": [
            {"id": "d", "type": "image", "text": None, "needs_vision": True}]},
        {"slide_number": 3, "elements": []},
    ]}
    inv = inventory(ix)
    assert inv == {"slides": 3, "elements": 4,
                   "by_type": {"image": 3, "title": 1}, "needs_vision": 2,
                   "images_with_alt": 1, "images_without_alt": 2,
                   "image_only_slides": 1}


def test_totals_sum_ok_decks_only():
    recs = [
        {"status": "OK", "hits": 3, "steps": 4, "relevant": 3, "false": 1, "flicker": 2,
         "filler_false": 2},
        {"status": "OK", "hits": 5, "steps": 5, "relevant": 5, "false": 0, "flicker": 0,
         "filler_false": 1},
        {"status": "FAILED: boom"},
    ]
    assert totals(recs) == {"decks_ok": 2, "decks_failed": 1, "hits": 8, "steps": 9,
                            "recall": 8 / 9, "false": 1, "precision": 8 / 9, "flicker": 2,
                            "filler_false": 3}


def test_filler_script_rotates_four_sentences_per_slide():
    sc = filler_script([1, 2, 3, 6], FILLER_HOLDOUT)
    assert [e["slide"] for e in sc] == [1, 2, 3, 6]
    assert sc[0]["segments"] == FILLER_HOLDOUT[1:5]
    assert sc[2]["segments"] == FILLER_HOLDOUT[0:4] == sc[3]["segments"]
    assert all(len(e["segments"]) == 4 for e in sc)


_STOP = set("a an the and or but so to of in on at for with by as is be it this that i you "
            "we me my let let's s just here now can see want any why really about".split())


def _content(sentences):
    return {w for s in sentences for w in re.findall(r"[a-z']+", s.lower())} - _STOP


def test_filler_holdout_is_disjoint_from_tune():
    # the holdout must not reuse the tune set's content words, or tuning leaks into it
    assert len(FILLER_HOLDOUT) == len(FILLER_TUNE) == 6
    assert not _content(FILLER_HOLDOUT) & _content(FILLER_TUNE)
    # nor with the matcher's null-hypothesis prototypes, or the holdout is tuned on
    assert not _content(FILLER_HOLDOUT) & _content(NULL_SPEECH)


@pytest.mark.skipif(os.environ.get("SMARTPPT_RUN_BENCH") != "1",
                    reason="set SMARTPPT_RUN_BENCH=1 (needs soffice, piper voice, sherpa)")
def test_bench_end_to_end(tmp_path):
    from tools.deck_bench import main
    from tools.make_corpus import build_corpus
    decks = tmp_path / "decks"
    decks.mkdir()
    for d in build_corpus(tmp_path / "corpus"):
        if d.name in ("title_and_bullets", "titled_chart_acronym"):
            shutil.copy(d.path, decks)
    out = tmp_path / "bench"
    assert main(["--decks", str(decks), "--out", str(out)]) == 0
    rep = json.loads((out / "report_default.json").read_text())
    assert sorted(r["deck"] for r in rep["decks"]) == ["title_and_bullets", "titled_chart_acronym"]
    assert all(r["status"] == "OK" and r["steps"] > 0 for r in rep["decks"])
    assert (out / "report_default.md").exists()
