"""Typed-input REPL for Stage 3 — simulate speech by typing.

Commands:
  :slide N          set current slide (1..N)
  :threshold X      override the policy threshold
  :gap X            override the runner-up gap
  :stick N          override stickiness (frames before switch)
  :reset            forget current active highlight
  :show             dump current slide's elements (id / type / text)
  :quit             exit
  <anything else>   treated as the latest spoken span
"""

from __future__ import annotations

import argparse

from .encoder import load_default_encoder
from .matcher import Index, Matcher, MatchPolicy, PolicyConfig


def _print_slide(idx: Index, slide_no: int) -> None:
    print(f"\n--- Slide {slide_no} ---")
    for ref in idx.elements_on(slide_no):
        tag = " " if ref.is_pointable else "*"  # * = unpointable (no triggers)
        print(f"  {tag} {ref.id:8s} {ref.type:15s} {(ref.text or ''):.60s}")
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Smart Pointer — typed-input matcher REPL")
    ap.add_argument("--index", default="out/index.json")
    ap.add_argument("--slide", type=int, default=1, help="initial slide number")
    ap.add_argument("--threshold", type=float, default=PolicyConfig.threshold)
    ap.add_argument("--gap", type=float, default=PolicyConfig.gap)
    ap.add_argument("--stick", type=int, default=PolicyConfig.stickiness)
    args = ap.parse_args(argv)

    print(f"loading index: {args.index}")
    enc = load_default_encoder()
    idx = Index(args.index, enc)
    matcher = Matcher(idx, enc)
    policy = MatchPolicy(matcher, PolicyConfig(
        threshold=args.threshold,
        gap=args.gap,
        stickiness=args.stick,
    ))
    slide_no = args.slide
    if slide_no not in idx._by_slide:
        print(f"slide {slide_no} not in index; using {idx.slide_numbers()[0]}")
        slide_no = idx.slide_numbers()[0]
    _print_slide(idx, slide_no)
    print("type a spoken phrase, or :help for commands, :quit to exit")
    while True:
        try:
            line = input(f"[s{slide_no}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.startswith(":"):
            cmd, *rest = line[1:].split(maxsplit=1)
            arg = rest[0] if rest else ""
            if cmd in ("q", "quit", "exit"):
                return 0
            if cmd == "help":
                print(__doc__)
                continue
            if cmd == "slide":
                try:
                    new = int(arg)
                except ValueError:
                    print("usage: :slide N"); continue
                if new not in idx._by_slide:
                    print(f"no such slide; have {idx.slide_numbers()}"); continue
                slide_no = new
                policy.reset()
                _print_slide(idx, slide_no)
                continue
            if cmd == "threshold":
                policy.cfg.threshold = float(arg); print(f"threshold = {policy.cfg.threshold}"); continue
            if cmd == "gap":
                policy.cfg.gap = float(arg); print(f"gap = {policy.cfg.gap}"); continue
            if cmd == "stick":
                policy.cfg.stickiness = int(arg); print(f"stickiness = {policy.cfg.stickiness}"); continue
            if cmd == "reset":
                policy.reset(); print("policy reset; no active highlight"); continue
            if cmd == "show":
                _print_slide(idx, slide_no); continue
            print(f"unknown command: :{cmd}")
            continue
        # treat as spoken
        res, active, why = policy.step(line, slide_no)
        active_text = ""
        if active:
            for ref in idx.elements_on(slide_no):
                if ref.id == active:
                    active_text = (ref.text or "")[:40]
                    break
        print(f"  best={res.element_id or '-':8s} score={res.score:.3f} (ru {res.runner_up_score:.3f}) → {why:6s} active={active or '-':8s} {active_text!r}")
    # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
