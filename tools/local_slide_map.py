"""Slide understanding maps from a LOCAL model (Ollama on this machine's GPU) instead
of Claude agents. Same packet in, same map format out, same validator
(tools/slide_map.py). Spec: docs/superpowers/specs/2026-10-02-slide-understanding-map-design.md

Per slide: one greedy call constrained by a per-slide JSON schema (every element id a
required key, roles/relation types as enums, unit members drawn from the slide's ids),
then a deterministic fix-up of bookkeeping slips (`fixup`) and the validator; one repair
turn with the validator's remaining errors; if still invalid, a minimal valid map (each
pointable element its own unit) so the deck still builds. Raw replies go to _stats.json.

    python3 -m tools.local_slide_map gen --model qwen3.5:9b --name qwen35_9b --no-think --num-gpu 32
    python3 -m tools.local_slide_map gen --backend llamacpp --model <Qwen2.5-VL gguf> --mmproj <mmproj gguf> --name qwen25vl_7b
    python3 -m tools.local_slide_map gen --model qwen2.5:7b --name qwen25_7b_text --no-image
    python3 -m tools.local_slide_map compare out/maps_local/qwen25vl7b/AI_Handout_Checker
    python3 -m tools.local_slide_map build qwen25vl7b      # index -> out/bench_local/<name>/<deck>
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from itertools import combinations
from pathlib import Path

from tools.slide_map import RELATIONS, ROLES, validate

OLLAMA = "http://localhost:11434"
DECK = "AI_Handout_Checker"
OPTIONS = {"temperature": 0, "top_k": 1, "seed": 0, "repeat_penalty": 1.0,
           "presence_penalty": 0.0, "frequency_penalty": 0.0,   # qwen3.5 ships 1.5: it bans repeated ids
           "num_ctx": 16384, "num_predict": 6144}
IMAGE_W = 1280   # 2000 px slide PNG -> ~1/2 the vision tokens
UID = "^u[0-9]+$"

TASK = """You are preparing a slide for a live presenter tool. While the presenter talks, the tool
highlights the thing on the slide they are talking about. Write the slide's "understanding map".

For every element decide:
- role, one of: title, section_label, card_heading, body, bullet, stat, caption, diagram_node,
  connector, table_cell, chart_bar, photo, icon, decoration, footer, code.
- pointable: true if a presenter could talk about it and expect it highlighted (titles, section
  labels, headings, text, numbers, photos, diagram boxes). false for decoration, arrows/lines
  without text, footers, page numbers, logos repeated on every slide.

Then group the pointable elements into units: one unit = one thing the presenter talks about as a
whole (a card = its heading + its body text; a photo + its caption; the title + its section label).
Every pointable element belongs to exactly one unit; non-pointable elements belong to none. Keep
separate points separate (each bullet of a list is usually its own unit unless they are one idea).
For each unit write:
- summary: one sentence on what the unit says.
- talk: 5-10 natural, plain lowercase phrases a presenter would SAY when talking about this unit
  without reading it out ("the before picture", "that number on the right", "the second stage").
- terms: how numbers, acronyms and identifiers in it are spoken ("ninety five percent", "o c r").
Also give:
- message: one sentence, the slide's main point.
- relations between units: flow (a then b), part_of, explains, contrasts, same_row, same_col.
- order: every uid exactly once, in the order a presenter would naturally cover them.
Unit ids are u1, u2, u3, ...

Format example (a different slide):
{"slide": 4, "message": "one sentence: the slide's main point",
 "items": {"s4_e2": {"role": "card_heading", "pointable": true}, "s4_e3": {"role": "body", "pointable": true},
           "s4_e9": {"role": "footer", "pointable": false}},
 "units": [{"uid": "u1", "members": ["s4_e2", "s4_e3"],
            "summary": "Stage A reads the page with a vision-language model",
            "talk": ["the reading stage", "how we turn pixels into text", "the first step", "where the scan gets read", "stage a"],
            "terms": ["stage a", "o c r"]}],
 "relations": [{"from": "u1", "to": "u2", "type": "flow"}],
 "order": ["u1", "u2"]}
"""


def map_schema(packet: dict) -> dict:
    """JSON schema for this slide's map: the runtime's grammar makes every id a required
    item key, units draw members from the slide's ids, roles/relations are enums."""
    ids = [e["id"] for e in packet["elements"]]
    s = {"type": "string"}
    item = {"type": "object", "properties": {"role": {"type": "string", "enum": sorted(ROLES)},
                                             "pointable": {"type": "boolean"}},
            "required": ["role", "pointable"], "additionalProperties": False}
    unit = {"type": "object", "properties": {
        "uid": {"type": "string", "pattern": UID},
        "members": {"type": "array", "items": {"type": "string", "enum": ids}, "minItems": 1},
        "summary": s,
        "talk": {"type": "array", "items": s, "minItems": 5, "maxItems": 10},
        "terms": {"type": "array", "items": s}},
        "required": ["uid", "members", "summary", "talk", "terms"], "additionalProperties": False}
    rel = {"type": "object", "properties": {"from": {"type": "string", "pattern": UID},
                                            "to": {"type": "string", "pattern": UID},
                                            "type": {"type": "string", "enum": sorted(RELATIONS)}},
           "required": ["from", "to", "type"], "additionalProperties": False}
    return {"type": "object", "properties": {
        "slide": {"type": "integer", "enum": [packet["slide"]]},
        "message": s,
        "items": {"type": "object", "properties": {i: item for i in ids}, "required": ids,
                  "additionalProperties": False},
        "units": {"type": "array", "items": unit, "minItems": 1},
        "relations": {"type": "array", "items": rel},
        "order": {"type": "array", "items": {"type": "string", "pattern": UID}}},
        "required": ["slide", "message", "items", "units", "relations", "order"],
        "additionalProperties": False}


def prompt(packet: dict, image: bool) -> str:
    sz = packet.get("slide_size") or {}
    els = "\n".join(json.dumps({"id": e["id"], "type": e["type"], "text": e["text"],
                                "bbox": e["bbox"]}, ensure_ascii=False) for e in packet["elements"])
    see = ("The slide image is attached; LOOK at it to see cards, flows, photos and what is decoration."
           if image else "You cannot see the slide; infer its layout from element types, text and bbox positions.")
    return (f"{TASK}\n{see}\nSlide {packet['slide']}, size {sz.get('w', 0):.2f} x {sz.get('h', 0):.2f} "
            f"{sz.get('units', '')} (bbox x, y = top-left). Elements:\n{els}\n\n"
            "Answer with the JSON map only, covering every element id above.")


def slide_image(png: str) -> str:
    """Base64 PNG, downscaled to IMAGE_W wide (deterministic resize)."""
    from PIL import Image
    im = Image.open(png).convert("RGB")
    im = im.resize((IMAGE_W, round(im.height * IMAGE_W / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ollama_chat(model: str, messages: list[dict], schema: dict, think: bool | None) -> dict:
    body = {"model": model, "messages": messages, "format": schema, "stream": False,
            "options": OPTIONS, "keep_alive": "10m"}
    if think is not None:
        body["think"] = think
    req = urllib.request.Request(OLLAMA + "/api/chat", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError:   # seen once: 500 while the previous model was unloading
        time.sleep(10)
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.loads(r.read())


def unload(model: str) -> None:
    req = urllib.request.Request(OLLAMA + "/api/generate", json.dumps(
        {"model": model, "keep_alive": 0}).encode(), {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=120).read()


FALLBACK_ROLE = {"title": "title", "arrow": "decoration", "image": "photo", "bullet": "bullet",
                 "stat": "stat", "diagram_label": "section_label", "caption": "caption"}


def own_unit(e: dict, uid: str) -> dict:
    """A one-element unit described by the element's own text."""
    text = " ".join((e.get("text") or "").split())
    words = text.lower().split() or [e["type"]]
    return {"uid": uid, "members": [e["id"]], "summary": text or e["type"],
            "talk": [" ".join(words[:12]), " ".join(words[:4]), " ".join(words[-4:])], "terms": []}


def fallback_map(packet: dict) -> dict:
    """Minimal valid map: roles from element types, footers/page numbers/text-less shapes
    not pointable, every pointable element its own unit, talk from its own text."""
    h = (packet.get("slide_size") or {}).get("h") or 7.5
    items, units = {}, []
    for e in packet["elements"]:
        text = " ".join((e.get("text") or "").split())
        footer = e["bbox"]["y"] > 0.92 * h or text.isdecimal()
        role = "footer" if footer and text else FALLBACK_ROLE.get(e["type"], "body")
        pointable = not footer and (bool(text) or e["type"] == "image")
        if not pointable and role not in ("footer", "decoration"):
            role = "decoration"
        items[e["id"]] = {"role": role, "pointable": pointable}
        if pointable:
            units.append(own_unit(e, f"u{len(units) + 1}"))
    return {"slide": packet["slide"], "message": units[0]["summary"] if units else f"slide {packet['slide']}",
            "items": items, "units": units,
        "relations": [], "order": [u["uid"] for u in units]}


# ---- deterministic fix-up of the model's bookkeeping (before validation) ----

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_SYM = {"~": "about", "≈": "about", "<": "under", ">": "over", "≤": "at most", "≥": "at least"}
_UNIT = {"%": "percent", "+": "plus", "x": "times", "×": "times", "k": "thousand", "fps": "frames per second"}
_MEASURE = {"s": "second", "ms": "millisecond", "min": "minute", "h": "hour", "kb": "kilobyte",
            "mb": "megabyte", "gb": "gigabyte", "tb": "terabyte"}   # plural unless exactly one
_N = r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
_NUM = re.compile(rf"([~≈<>≤≥]?)\s*(?<![A-Za-z\d.,]){_N}(?:\s*[–—-]\s*{_N})?\s*"
                  r"(%|\+|×|(?:x|k|fps|s|ms|min|h|kb|mb|gb|tb)\b)?", re.I)


def _words(num: str) -> str:
    """'15' -> 'fifteen', '2.5' -> 'two point five' (digit by digit from a million up)."""
    whole, _, frac = num.replace(",", "").partition(".")
    n = int(whole)
    if n < 20:
        w = _ONES[n]
    elif n < 100:
        w = _TENS[n // 10] + (f" {_ONES[n % 10]}" if n % 10 else "")
    elif n < 1000:
        w = f"{_ONES[n // 100]} hundred" + (f" {_words(str(n % 100))}" if n % 100 else "")
    elif n < 10 ** 6:
        w = f"{_words(str(n // 1000))} thousand" + (f" {_words(str(n % 1000))}" if n % 1000 else "")
    else:   # ponytail: ids/years read digit by digit; add a years rule if a deck needs it
        w = " ".join(_ONES[int(d)] for d in whole)
    return w + (" point " + " ".join(_ONES[int(d)] for d in frac) if frac else "")


def spoken_terms(text: str) -> list[str]:
    """Spoken forms of the numbers in `text`: '15–33' -> 'fifteen to thirty three',
    '~1 min' -> 'about one minute', '8 GB' -> 'eight gigabytes'. Digits glued to letters
    (model names, versions) are left alone."""
    out = []
    for sym, a, b, unit in _NUM.findall(text or ""):
        u = unit.lower()
        u = _UNIT.get(u) or (_MEASURE[u] + ("" if a == "1" and not b else "s") if u in _MEASURE else "")
        w = " ".join(x for x in (_SYM.get(sym, ""), _words(a), f"to {_words(b)}" if b else "", u) if x)
        if w not in out:
            out.append(w)
    return out


NOT_POINTABLE_ROLES = {"footer", "decoration", "icon"}


def fixup(m: dict, packet: dict) -> tuple[dict, dict]:
    """(fixed copy of m, {fix: count}). Repairs bookkeeping only, never re-partitions:
    unknown ids dropped; an id in several units kept in the first (units-list order);
    a non-pointable member made pointable unless footer/decoration/icon (then removed);
    blank talk phrases dropped; empty units removed; a pointable element in no unit
    gets its own unit; relations with unknown uids dropped; order = the model's order
    (unknown/repeated uids dropped) + missing uids appended. A no-op on a valid map."""
    m = json.loads(json.dumps(m))
    fx: dict[str, int] = {}

    def hit(k: str, n: int = 1) -> None:
        if n:
            fx[k] = fx.get(k, 0) + n

    els = {e["id"]: e for e in packet["elements"]}
    items = m.get("items") or {}
    for i in [i for i in items if i not in els]:
        del items[i]
        hit("unknown_id")
    seen: set[str] = set()
    for u in m.get("units") or []:
        keep = []
        for i in u.get("members") or []:
            if i not in els:
                hit("unknown_id")
            elif i in seen:
                hit("duplicate_member")
            elif i in items and not items[i].get("pointable"):
                if items[i].get("role") in NOT_POINTABLE_ROLES:
                    hit("nonpointable_member_removed")
                    continue
                items[i]["pointable"] = True
                hit("member_made_pointable")
                keep.append(i)
            else:
                keep.append(i)
            seen.add(i)
        u["members"] = keep
        talk = [t for t in u.get("talk") or [] if isinstance(t, str) and t.strip()]
        hit("blank_talk", len(u.get("talk") or []) - len(talk))
        u["talk"] = talk
    units = [u for u in m.get("units") or [] if u["members"]]
    hit("empty_unit", len(m.get("units") or []) - len(units))
    used = {int(u["uid"][1:]) for u in units if str(u.get("uid", ""))[1:].isdecimal()}
    k = max(used, default=0)
    for i, it in items.items():
        if it.get("pointable") and i not in seen:
            k += 1
            units.append(dict(own_unit(els[i], f"u{k}"), terms=spoken_terms(els[i].get("text"))))
            hit("orphan_unit")
    m["units"] = units
    uids = [u["uid"] for u in units]
    rels = [r for r in m.get("relations") or [] if r.get("from") in uids and r.get("to") in uids]
    hit("bad_relation", len(m.get("relations") or []) - len(rels))
    m["relations"] = rels
    order = list(dict.fromkeys(x for x in m.get("order") or [] if x in uids))
    order += [x for x in uids if x not in order]
    if order != m.get("order"):
        hit("order_rebuilt")
    m["order"] = order
    return m, fx


def generate(packet: dict, chat, image: bool) -> tuple[dict, dict]:
    """(map, stats). `chat(messages, schema)` -> Ollama /api/chat response dict."""
    schema = map_schema(packet)
    msg = {"role": "user", "content": prompt(packet, image)}
    if image:
        msg["images"] = [slide_image(packet["png"])]
    messages = [msg]
    st = {"slide": packet["slide"], "calls": [], "errors": [], "fixes": [], "raw": []}
    for attempt in range(2):
        r = chat(messages, schema)
        st["calls"].append({k: r.get(k) for k in ("total_duration", "load_duration",
                                                   "prompt_eval_count", "eval_count")})
        content = r["message"]["content"]
        st["raw"].append(content)
        try:   # some runtimes/models ignore `format` and fence the JSON in ```json
            m = json.loads(content[content.find("{"):content.rfind("}") + 1])
            raw_errs = validate(m, packet)
            m, fixes = fixup(m, packet)
            errs = validate(m, packet)
        except json.JSONDecodeError as ex:
            m, raw_errs, fixes = None, [f"not valid JSON: {ex}"], {}
            errs = raw_errs
        st["errors"].append(raw_errs)   # the model's own errors, before fix-up
        st["fixes"].append(fixes)
        if not errs:
            st["result"] = ("first" if not raw_errs else "fixed") if attempt == 0 else "repaired"
            return m, st
        messages = messages + [{"role": "assistant", "content": content},
                               {"role": "user", "content": "The map is invalid:\n" + "\n".join(
                                   f"- {e}" for e in errs) + "\nReturn the corrected complete JSON map."}]
    st["result"] = "fallback"
    return fallback_map(packet), st


class LlamaCppChat:
    """`chat(messages, schema)` backed by tools/_vlm_worker.py in a llama-cpp-python venv
    (Qwen2.5-VL GGUF + mmproj, every layer on the GPU)."""

    def __init__(self, python: str, model: str, mmproj: str, ctx: int):
        worker = Path(__file__).with_name("_vlm_worker.py")
        self.p = subprocess.Popen([python, str(worker), "--model", model, "--mmproj", mmproj,
                                   "--ctx", str(ctx), "--max-tokens", str(OPTIONS["num_predict"])],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.info = json.loads(self.p.stdout.readline() or '{"ok": false, "error": "worker died"}')
        if not self.info.get("ok"):
            raise RuntimeError(self.info.get("error"))

    def __call__(self, messages: list[dict], schema: dict) -> dict:
        self.p.stdin.write(json.dumps({"messages": messages, "schema": schema}) + "\n")
        self.p.stdin.flush()
        r = json.loads(self.p.stdout.readline() or '{"error": "worker died"}')
        if "error" in r:
            raise RuntimeError(r["error"])
        return r

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=60)


class VramPeak:
    """Peak GPU memory used (MiB) while the context is open, polled via nvidia-smi."""

    def __enter__(self):
        self.peak, self._stop = 0, threading.Event()
        self.base = self._read()
        self._t = threading.Thread(target=self._poll, daemon=True)
        self._t.start()
        return self

    @staticmethod
    def _read() -> int:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        return int(out.split()[0])

    def _poll(self):
        while not self._stop.wait(0.5):
            self.peak = max(self.peak, self._read())

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()


def rescued(stats: list[dict]) -> dict:
    """{fix: slides it was applied to on a slide that fix-up made valid (result "fixed")}."""
    out: dict[str, int] = {}
    for s in stats:
        if s["result"] == "fixed":
            for k in s["fixes"][0]:
                out[k] = out.get(k, 0) + 1
    return out


def cmd_gen(a) -> int:
    if a.num_gpu is not None:
        OPTIONS["num_gpu"] = a.num_gpu
    packets = sorted(Path(a.packets).glob("slide_*.json"))
    out = Path(a.out_root) / a.name / DECK
    out.mkdir(parents=True, exist_ok=True)
    think = False if a.no_think else None

    def chat(messages, schema):
        return ollama_chat(a.model, messages, schema, think)

    stats = []
    if a.backend == "ollama":
        unload(a.model)   # VRAM baseline without the model
    with VramPeak() as vram:
        if a.backend == "llamacpp":
            chat = LlamaCppChat(a.python, a.model, a.mmproj, OPTIONS["num_ctx"])
        for p in packets:
            if a.only and int(p.stem.split("_")[1]) not in a.only:
                continue
            pk = json.loads(p.read_text())
            t0 = time.perf_counter()
            m, st = generate(pk, chat, image=not a.no_image)
            st["wall_s"] = round(time.perf_counter() - t0, 2)
            (out / p.name).write_text(json.dumps(m, indent=1, ensure_ascii=False))
            stats.append(st)
            print(f"slide {pk['slide']:3d} {st['result']:9s} {st['wall_s']:6.1f}s "
                  f"prompt {st['calls'][0]['prompt_eval_count']} out {st['calls'][-1]['eval_count']} "
                  f"fixes {st['fixes']}", flush=True)
        if a.backend == "llamacpp":
            ps = {"models": [{"name": a.model, "mmproj": a.mmproj, "runtime": chat.info}]}
            chat.close()
        else:
            ps = json.loads(urllib.request.urlopen(OLLAMA + "/api/ps").read())
            unload(a.model)
    n = len(stats)
    summary = {"model": a.model, "name": a.name, "image": not a.no_image, "slides": n,
               **{k: sum(s["result"] == r for s in stats)
                  for k, r in (("valid_first", "first"), ("fixed", "fixed"), ("repaired", "repaired"),
                               ("fallback", "fallback"))},
               "rescued_by": rescued(stats),
               "s_per_slide": round(sum(s["wall_s"] for s in stats) / max(n, 1), 1),
               "vram_base_mib": vram.base, "vram_peak_mib": vram.peak,
               "backend": a.backend,
               "runtime": [{k: x.get(k) for k in ("name", "digest", "size", "size_vram", "mmproj", "runtime")}
                             for x in ps.get("models", [])],
               "options": OPTIONS, "per_slide": stats}
    (out / "_stats.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_slide", "options")}))
    return 0


# ---- structural comparison against the reference (Claude) maps ----

def same_unit_pairs(m: dict) -> set[frozenset]:
    return {frozenset(p) for u in m["units"] for p in combinations(sorted(u["members"]), 2)}


def compare(local: dict, ref: dict) -> dict:
    """Counts for one slide: same-unit pair TP/FP/FN, pointable and role agreement."""
    lp, rp = same_unit_pairs(local), same_unit_pairs(ref)
    ids = list(ref["items"])
    li, ri = local["items"], ref["items"]
    return {"tp": len(lp & rp), "fp": len(lp - rp), "fn": len(rp - lp), "n": len(ids),
            "point_eq": sum(li[i]["pointable"] == ri[i]["pointable"] for i in ids),
            "point_tp": sum(li[i]["pointable"] and ri[i]["pointable"] for i in ids),
            "point_l": sum(li[i]["pointable"] for i in ids), "point_r": sum(ri[i]["pointable"] for i in ids),
            "role_eq": sum(li[i]["role"] == ri[i]["role"] for i in ids),
            "units_l": len(local["units"]), "units_r": len(ref["units"]),
            "confusions": [(ri[i]["role"], li[i]["role"]) for i in ids if li[i]["role"] != ri[i]["role"]]}


def compare_dirs(local_dir: Path, ref_dir: Path) -> dict:
    tot: dict = {}
    for p in sorted(ref_dir.glob("slide_*.json")):
        c = compare(json.loads((local_dir / p.name).read_text()), json.loads(p.read_text()))
        for k, v in c.items():
            tot[k] = tot.get(k, 0 if not isinstance(v, list) else []) + v
    prec = tot["tp"] / max(tot["tp"] + tot["fp"], 1)
    rec = tot["tp"] / max(tot["tp"] + tot["fn"], 1)
    conf: dict = {}
    for c in tot.pop("confusions"):
        conf[f"{c[0]}->{c[1]}"] = conf.get(f"{c[0]}->{c[1]}", 0) + 1
    return {"pair_p": round(prec, 3), "pair_r": round(rec, 3),
            "pair_f1": round(2 * prec * rec / max(prec + rec, 1e-9), 3),
            "pointable_acc": round(tot["point_eq"] / tot["n"], 3),
            "pointable_p": round(tot["point_tp"] / max(tot["point_l"], 1), 3),
            "pointable_r": round(tot["point_tp"] / max(tot["point_r"], 1), 3),
            "role_acc": round(tot["role_eq"] / tot["n"], 3),
            "units_local": tot["units_l"], "units_ref": tot["units_r"],
            "top_role_confusions": sorted(conf.items(), key=lambda kv: -kv[1])[:6]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--model", required=True, help="ollama tag, e.g. qwen2.5vl:7b")
    g.add_argument("--name", required=True, help="output folder name")
    g.add_argument("--packets", default=f"out/maps/{DECK}/packets")
    g.add_argument("--out-root", default="out/maps_local")
    g.add_argument("--backend", default="ollama", choices=("ollama", "llamacpp"),
                   help="llamacpp: --model is a GGUF path, needs --mmproj")
    g.add_argument("--mmproj", default=None)
    g.add_argument("--python", default=os.environ.get(
        "SMARTPPT_LLM_PYTHON", str(Path.home() / ".venvs/smartppt-llm-cuda/bin/python")),
        help="llama-cpp-python (CUDA) interpreter for --backend llamacpp")
    g.add_argument("--no-image", action="store_true", help="text-only: the packet, no slide image")
    g.add_argument("--no-think", action="store_true", help="send think=false (thinking models)")
    g.add_argument("--num-gpu", type=int, default=None,
                   help="pin Ollama's GPU layer count: a model that only partly fits (qwen3.5:9b: 32) "
                        "otherwise gets a split chosen from free VRAM, and its output changes with it")
    g.add_argument("--only", type=lambda s: [int(x) for x in s.split(",")], default=None)
    c = sub.add_parser("compare")
    c.add_argument("local_dir", type=Path)
    c.add_argument("--ref", type=Path, default=Path(f"out/maps/{DECK}"))
    b = sub.add_parser("build")
    b.add_argument("name")
    b.add_argument("--maps", type=Path, default=None, help="default out/maps_local/<name>/<deck>")
    a = ap.parse_args(argv)
    if a.cmd == "gen":
        return cmd_gen(a)
    if a.cmd == "compare":
        print(json.dumps(compare_dirs(a.local_dir, a.ref)))
        return 0
    from src.build_index import build_index
    build_index(f"decks/{DECK}.pptx", f"out/bench_local/{a.name}/{DECK}",
                slide_map_dir=a.maps or Path("out/maps_local") / a.name / DECK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
