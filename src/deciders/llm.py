"""LLM decider — a small local instruct model picks which slide unit a clause refers to.

The model (llama.cpp GGUF, e.g. Qwen2.5-1.5B-Instruct Q4_K_M) runs in a persistent
subprocess (``tools/llm_decider_server.py``) under its own venv, so the main process
never imports llama_cpp. Protocol: one JSON object per line on stdin/stdout.

Scoring is one forward pass, no sampling: the prompt lists the slide's units as
lettered items and asks for a letter (or N = none); the score of each option is the
log-probability of its letter as the next token. The per-slide prefix (instructions +
unit list) stays in the server's KV cache, so a call only evaluates the clause tokens.

Setup (once, then fully offline):
    python3 -m venv ~/.venvs/smartppt-llm
    ~/.venvs/smartppt-llm/bin/pip install llama-cpp-python huggingface_hub
    ~/.venvs/smartppt-llm/bin/hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF \\
        qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir ~/.cache/smartppt/llm
GPU (optional): a second venv with the CUDA wheel, then
    SMARTPPT_LLM_PYTHON=~/.venvs/smartppt-llm-cuda/bin/python + LLMDecider(gpu_layers=99)
    ~/.venvs/smartppt-llm-cuda/bin/pip install llama-cpp-python \\
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

Measured on i5-14400F / RTX 4060 (AI_Handout_Checker, 226 oracle clauses): 1.5B CPU
12 threads ~135 ms/call, GPU ~10 ms; ~61-66% argmax agreement with the oracle for
both 1.5B and 3B (3B is 2x slower on CPU, no better). Call warm(units) for each slide
at startup (~2.4 s/slide on CPU) so slide switches cost ~100 ms, not a prefix eval.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENV_PY = Path(os.environ.get("SMARTPPT_LLM_PYTHON", Path.home() / ".venvs/smartppt-llm/bin/python"))
SERVER = ROOT / "tools" / "llm_decider_server.py"
MODEL_DIR = Path.home() / ".cache" / "smartppt" / "llm"
DEFAULT_MODEL = Path(os.environ.get("SMARTPPT_LLM_MODEL", MODEL_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf"))

NONE_LETTER = "N"
LETTERS = [c for c in "ABCDEFGHIJKLMOPQRSTUVWXYZ"]   # N is reserved for "none"
TEXT_CHARS = 160                                     # key slide text per unit, truncated

SYSTEM = ("You follow a live presentation. Given the items on the current slide and what "
          "the speaker just said, name the one item the speaker is talking about.")
QUESTION = ("Which item is the speaker referring to in the words after Now? "
            "Answer with the letter, or N for none.")


class LLMDeciderUnavailable(RuntimeError):
    """The venv, server script or model file is missing, or the server died."""


def _chat(role: str, text: str) -> str:
    return f"<|im_start|>{role}\n{text}<|im_end|>\n"


def unit_line(letter: str, u: dict) -> str:
    """One lettered item: the map summary plus the start of the unit's slide text."""
    text = " ".join((u.get("text") or "").split())
    if len(text) > TEXT_CHARS:
        text = text[:TEXT_CHARS].rsplit(" ", 1)[0] + " ..."
    line = f"{letter}. {u.get('summary') or text}"
    if text and u.get("summary"):
        line += f' (on slide: "{text}")'
    if u.get("terms"):
        line += " [" + ", ".join(u["terms"]) + "]"
    return line


def build_prompt(clause: str, context: str, units: list[dict]) -> tuple[str, str, list[str]]:
    """-> (prefix, suffix, option letters). prefix depends only on the slide's units,
    so the server can keep it cached across calls; the last option is NONE_LETTER."""
    if len(units) > len(LETTERS):
        raise ValueError(f"at most {len(LETTERS)} units per slide, got {len(units)}")
    letters = LETTERS[:len(units)]
    items = "\n".join(unit_line(c, u) for c, u in zip(letters, units))
    prefix = (_chat("system", SYSTEM)
              + "<|im_start|>user\nItems on the current slide:\n" + items
              + f"\n{NONE_LETTER}. none of these (greeting, transition, or nothing specific)\n\n"
              + QUESTION + "\n")
    suffix = (f'Before: "{" ".join(context.split())}"\n'
              f'Now: "{" ".join(clause.split())}"<|im_end|>\n<|im_start|>assistant\n')
    return prefix, suffix, letters + [NONE_LETTER]


def parse_scores(reply: dict, units: list[dict], options: list[str]) -> dict[str, float]:
    """Server reply {"ok": true, "scores": [...]} (one per option) -> {uid..., "none": score}."""
    if not reply.get("ok"):
        raise LLMDeciderUnavailable(f"llm server error: {reply.get('error', reply)}")
    scores = reply["scores"]
    if len(scores) != len(options):
        raise LLMDeciderUnavailable(f"llm server returned {len(scores)} scores for {len(options)} options")
    out = {u["uid"]: float(s) for u, s in zip(units, scores)}
    out["none"] = float(scores[-1])
    return out


class LLMDecider:
    """decide(clause, context, units) -> {uid: logprob, ..., "none": logprob}.

    One persistent server process per instance; close() (or use as a context manager)
    to stop it. Not thread-safe: one request in flight at a time."""

    def __init__(self, model: str | Path | None = None, *, python: str | Path | None = None,
                 server: str | Path | None = None, threads: int | None = None,
                 gpu_layers: int = 0, prefix_cache: bool = True, calibrate: bool = True,
                 use_context: bool = False):
        model = Path(model or DEFAULT_MODEL)
        python = Path(python or VENV_PY)
        server = Path(server or SERVER)
        for what, p, hint in (
                ("venv python", python, "python3 -m venv ~/.venvs/smartppt-llm && "
                 "~/.venvs/smartppt-llm/bin/pip install llama-cpp-python huggingface_hub"),
                ("server script", server, "expected tools/llm_decider_server.py in the repo"),
                ("model", model, f"download a GGUF into {MODEL_DIR} (see src/deciders/llm.py docstring) "
                 "or set SMARTPPT_LLM_MODEL")):
            if not p.exists():
                raise LLMDeciderUnavailable(f"LLM decider: {what} not found at {p} — {hint}")
        cmd = [str(python), str(server), "--model", str(model), "--gpu-layers", str(gpu_layers)]
        if threads:
            cmd += ["--threads", str(threads)]
        if not prefix_cache:
            cmd.append("--no-prefix-cache")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, bufsize=1)
        self.info = self._read()          # server's ready line: {"ok", "model", ...}
        if not self.info.get("ok"):
            self.close()
            raise LLMDeciderUnavailable(f"llm server failed to start: {self.info.get('error')}")
        self.last_ms = 0.0
        self.calibrate = calibrate
        self.use_context = use_context   # off: measured -5 pts oracle agreement with it on
        self._prior: dict[str, list[float]] = {}   # prefix -> scores for an empty clause

    def _read(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            code = self.proc.poll()
            raise LLMDeciderUnavailable(f"llm server exited (code {code}); see its stderr above")
        try:
            return json.loads(line)
        except ValueError as e:
            raise LLMDeciderUnavailable(f"llm server sent a non-protocol line: {line[:200]!r}") from e

    def decide(self, clause: str, context: str, units: list[dict]) -> dict[str, float]:
        if not units:
            return {"none": 0.0}
        prefix, suffix, options = build_prompt(clause, context if self.use_context else "", units)
        scores = self._score(prefix, suffix, options)
        if self.calibrate:
            # contextual calibration: subtract the model's letter prior for this slide
            # (scores of a content-free clause), which removes its position/"none" bias
            if prefix not in self._prior:
                if len(self._prior) > 64:
                    self._prior.clear()
                self._prior[prefix] = self._score(prefix, build_prompt("", "", units)[1], options)
            scores = [s - p for s, p in zip(scores, self._prior[prefix])]
        return parse_scores({"ok": True, "scores": scores}, units, options)

    def warm(self, units: list[dict]) -> None:
        """Evaluate (and keep) a slide's prefix ahead of time, e.g. for every slide at
        startup, so the first clause on that slide doesn't pay the prefix cost."""
        if units:
            self.decide("", "", units)

    def _score(self, prefix: str, suffix: str, options: list[str]) -> list[float]:
        try:
            self.proc.stdin.write(json.dumps({"prefix": prefix, "suffix": suffix,
                                              "options": options}) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise LLMDeciderUnavailable(f"llm server not running: {e}") from e
        reply = self._read()
        self.last_ms = float(reply.get("ms", 0.0))
        return list(parse_scores(reply, [{"uid": o} for o in options[:-1]], options).values())

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
