"""Slide understanding maps: per-slide input packets for the map agents, and the
validator every map must pass. Spec: docs/superpowers/specs/2026-10-02-slide-understanding-map-design.md

    python3 -m tools.slide_map packets out/bench/<deck>/default out/maps/<deck>
    python3 -m tools.slide_map validate out/maps/<deck>/slide_004.json out/maps/<deck>/packets/slide_004.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROLES = {"title", "section_label", "card_heading", "body", "bullet", "stat", "caption",
         "diagram_node", "connector", "table_cell", "chart_bar", "photo", "icon",
         "decoration", "footer", "code"}
RELATIONS = {"flow", "part_of", "explains", "contrasts", "same_row", "same_col"}


def write_packets(deck_dir: Path, map_dir: Path) -> int:
    """One packet per slide from the deck's built index + rendered PNGs."""
    ix = json.loads((deck_dir / "index.scoped.json").read_text())
    out = map_dir / "packets"
    out.mkdir(parents=True, exist_ok=True)
    for s in ix["slides"]:
        n = s["slide_number"]
        png = (deck_dir / "slides" / f"slide_{n:03d}.png").resolve()
        els = [{"id": e["id"], "type": e["type"], "text": e.get("text"),
                "bbox": {k: round(e["bbox"][k], 3) for k in ("x", "y", "w", "h")}}
               for e in s["elements"]]
        (out / f"slide_{n:03d}.json").write_text(json.dumps(
            {"deck": ix.get("deck"), "slide": n, "slide_size": ix.get("slide_size"),
             "png": str(png), "elements": els}, indent=1, ensure_ascii=False))
    return len(ix["slides"])


def validate(m: dict, packet: dict) -> list[str]:
    """All problems with map `m` for its slide packet; [] means valid."""
    errs: list[str] = []
    ids = {e["id"] for e in packet["elements"]}
    if m.get("slide") != packet["slide"]:
        errs.append(f"slide {m.get('slide')} != packet slide {packet['slide']}")
    if not isinstance(m.get("message"), str) or not m["message"].strip():
        errs.append("message missing")
    items = m.get("items") or {}
    if set(items) != ids:
        missing, extra = sorted(ids - set(items)), sorted(set(items) - ids)
        errs.append(f"items must cover exactly the slide's ids (missing {missing}, unknown {extra})")
    for i, it in items.items():
        if it.get("role") not in ROLES:
            errs.append(f"{i}: bad role {it.get('role')!r}")
        if not isinstance(it.get("pointable"), bool):
            errs.append(f"{i}: pointable must be true/false")
    units = m.get("units") or []
    uids = [u.get("uid") for u in units]
    if len(set(uids)) != len(uids) or not all(uids):
        errs.append("unit uids must be present and unique")
    owner: dict[str, str] = {}
    for u in units:
        for mem in u.get("members") or []:
            if mem not in ids:
                errs.append(f"{u.get('uid')}: unknown member {mem}")
            elif mem in owner:
                errs.append(f"{mem} in two units ({owner[mem]}, {u.get('uid')})")
            else:
                owner[mem] = u.get("uid")
            if mem in items and not items[mem].get("pointable"):
                errs.append(f"{u.get('uid')}: member {mem} is not pointable")
        if not u.get("members"):
            errs.append(f"{u.get('uid')}: no members")
        if not u.get("summary"):
            errs.append(f"{u.get('uid')}: summary missing")
        talk = u.get("talk") or []
        if not 3 <= len(talk) <= 12 or not all(isinstance(t, str) and t.strip() for t in talk):
            errs.append(f"{u.get('uid')}: talk needs 3-12 non-empty phrases")
        if not isinstance(u.get("terms", []), list):
            errs.append(f"{u.get('uid')}: terms must be a list")
    for i, it in items.items():
        if it.get("pointable") and i not in owner:
            errs.append(f"pointable {i} belongs to no unit")
    us = set(uids)
    for r in m.get("relations") or []:
        if r.get("type") not in RELATIONS:
            errs.append(f"bad relation type {r.get('type')!r}")
        if r.get("from") not in us or r.get("to") not in us:
            errs.append(f"relation {r.get('from')}->{r.get('to')} uses unknown uid")
    if sorted(m.get("order") or []) != sorted(uids):
        errs.append("order must list every uid exactly once")
    return errs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("packets")
    p.add_argument("deck_dir", type=Path)
    p.add_argument("map_dir", type=Path)
    v = sub.add_parser("validate")
    v.add_argument("map", type=Path)
    v.add_argument("packet", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "packets":
        print(f"{write_packets(a.deck_dir, a.map_dir)} packets -> {a.map_dir / 'packets'}")
        return 0
    errs = validate(json.loads(a.map.read_text()), json.loads(a.packet.read_text()))
    print("VALID" if not errs else "INVALID\n" + "\n".join(f"- {e}" for e in errs))
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main())
