"""llm_decider_server.py — model side of src/deciders/llm.py. Runs INSIDE
~/.venvs/smartppt-llm (llama-cpp-python), never imported by the main process.

stdin/stdout JSON lines. First line out: {"ok": true, "model": ..., "threads": ...}
(or {"ok": false, "error": ...}). Then per request
    {"prefix": str, "suffix": str, "options": ["A", ..., "N"]}
->  {"ok": true, "scores": [logprob per option], "ms": float, "prefix_hit": bool}
Each option's score is the log-softmax (over the full vocab) of its first token at
the position after prefix+suffix: one forward pass, no sampling. The prefix's KV
cache is kept and reused while consecutive requests share it, and saved per prefix
(LRU) so returning to a slide restores its KV instead of re-evaluating it."""
import argparse
import ctypes
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, default=12)  # best measured on i5-14400F (6P+4E)
    ap.add_argument("--gpu-layers", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--no-prefix-cache", action="store_true")
    ap.add_argument("--states", type=int, default=32, help="saved per-slide prefix states (LRU)")
    a = ap.parse_args()
    out = sys.stdout
    sys.stdout = sys.stderr                  # nothing but protocol lines on the real stdout

    def send(obj):
        out.write(json.dumps(obj) + "\n")
        out.flush()

    try:
        import llama_cpp
        llm = llama_cpp.Llama(model_path=a.model, n_ctx=a.ctx, n_threads=a.threads,
                              n_threads_batch=a.threads, n_gpu_layers=a.gpu_layers,
                              logits_all=False, verbose=False)
    except Exception as e:                   # missing lib / bad model file
        send({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return
    n_vocab = llm.n_vocab()
    send({"ok": True, "model": os.path.basename(a.model), "threads": a.threads,
          "gpu_layers": a.gpu_layers, "prefix_cache": not a.no_prefix_cache})

    tok = lambda s: llm.tokenize(s.encode(), add_bos=False, special=True)
    states = OrderedDict()                   # prefix string -> saved KV state (LRU)
    cached = None                            # prefix string currently in the KV cache
    for line in sys.stdin:
        if not line.strip():
            continue
        t0 = time.perf_counter()
        try:
            req = json.loads(line)
            prefix, suf = req["prefix"], tok(req["suffix"])
            opt = [tok(o)[0] for o in req["options"]]
            hit = cached == prefix
            if a.no_prefix_cache:
                pre = tok(prefix)
                llm.reset()
                llm.eval(pre + suf)
            else:
                if not hit:
                    cached = None
                    if prefix in states:     # slide seen before: restore its KV (a memcpy)
                        llm._ctx.kv_cache_seq_rm(-1, 0, -1)
                        buf, _n = states[prefix]
                        if not llama_cpp.llama_state_seq_set_data(llm.ctx, buf, len(buf), 0):
                            raise RuntimeError("restoring prefix KV failed")
                        states.move_to_end(prefix)
                    else:
                        pre = tok(prefix)
                        if len(pre) + len(suf) >= a.ctx:
                            raise ValueError(f"prompt {len(pre) + len(suf)} tokens > ctx {a.ctx}")
                        llm.reset()
                        llm.eval(pre)
                        size = llama_cpp.llama_state_seq_get_size(llm.ctx, 0)
                        buf = (ctypes.c_uint8 * size)()   # just this prefix's KV cells
                        llama_cpp.llama_state_seq_get_data(llm.ctx, buf, size, 0)
                        states[prefix] = (buf, len(pre))
                        if len(states) > a.states:
                            states.popitem(last=False)
                    cached = prefix
                llm.n_tokens = states[prefix][1]   # eval() drops the KV past n_tokens
                llm.eval(suf)
            ptr = llama_cpp.llama_get_logits_ith(llm.ctx, -1)
            logits = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)),
                                           shape=(n_vocab,)).astype(np.float64)
            m = logits.max()
            lse = m + np.log(np.exp(logits - m).sum())
            send({"ok": True, "scores": [float(logits[i] - lse) for i in opt],
                  "ms": round((time.perf_counter() - t0) * 1000, 2), "prefix_hit": hit})
        except Exception as e:
            cached = None
            send({"ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
