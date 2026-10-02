"""Apply per-slide understanding maps to an index (build stage, after vision_link,
before scope). Spec: docs/superpowers/specs/2026-10-02-slide-understanding-map-design.md

A map (`<map_dir>/slide_NNN.json`, format + validator in tools/slide_map.py) marks
each element's role and pointability, groups pointable elements into units, and
gives each unit spoken references. Slides without a valid map are untouched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from tools.slide_map import validate


def slide_packet(slide: dict) -> dict:
    """The part of a slide packet the validator needs (slide number + element ids)."""
    return {"slide": slide["slide_number"], "elements": [{"id": e["id"]} for e in slide["elements"]]}


def apply_slide_maps(index: dict, map_dir: Path) -> dict:
    """Mutate + return `index`: on every slide with a valid map set `map_role`,
    `pointable`, `map_unit` on its elements, add each unit's talk + terms to the
    trigger phrases of its first member in reading order (y, then x), and store
    `slide["map"] = {message, relations, order, units: {uid: [members, reading order]}}`."""
    map_dir = Path(map_dir)
    if not map_dir.is_dir():
        return index
    for slide in index.get("slides", []):
        n = slide["slide_number"]
        p = map_dir / f"slide_{n:03d}.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        errs = validate(m, slide_packet(slide))
        if errs:
            print(f"slide_map: slide {n} map invalid, skipped: {'; '.join(errs)}", file=sys.stderr)
            continue
        by_id = {e["id"]: e for e in slide["elements"]}
        pos = {e["id"]: k for k, e in enumerate(slide["elements"])}

        def reading(i: str) -> tuple:
            b = by_id[i].get("bbox") or {}
            return (b.get("y", 0.0), b.get("x", 0.0), pos[i])

        for i, it in m["items"].items():
            by_id[i].update(map_role=it["role"], pointable=it["pointable"])
        units = {}
        for u in m["units"]:
            members = sorted(u["members"], key=reading)
            units[u["uid"]] = members
            for i in members:
                by_id[i]["map_unit"] = u["uid"]
            phrases = by_id[members[0]].setdefault("trigger_phrases", [])
            have = {ph.lower() for ph in phrases}
            for ph in list(u["talk"]) + list(u.get("terms") or []):
                ph = " ".join(ph.lower().split())
                if ph and ph not in have:
                    phrases.append(ph)
                    have.add(ph)
        slide["map"] = {"message": m["message"], "relations": m.get("relations") or [],
                        "order": m["order"], "units": units}
    return index
