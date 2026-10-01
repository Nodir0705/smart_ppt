import json
import os
import shutil

import pytest

from tools.deck_bench import inventory, totals


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
        {"status": "OK", "hits": 3, "steps": 4, "relevant": 3, "false": 1, "flicker": 2},
        {"status": "OK", "hits": 5, "steps": 5, "relevant": 5, "false": 0, "flicker": 0},
        {"status": "FAILED: boom"},
    ]
    assert totals(recs) == {"decks_ok": 2, "decks_failed": 1, "hits": 8, "steps": 9,
                            "recall": 8 / 9, "false": 1, "precision": 8 / 9, "flicker": 2}


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
