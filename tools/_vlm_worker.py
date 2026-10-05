"""_vlm_worker.py — llama.cpp side of tools/local_slide_map.py (--backend llamacpp).
Runs INSIDE a llama-cpp-python venv with CUDA (e.g. ~/.venvs/smartppt-llm-cuda), never
imported by the main process.

stdin/stdout JSON lines. First line out: {"ok": true, ...} or {"ok": false, "error": ...}.
Then per request {"messages": [Ollama-style {role, content, images?: [b64 png]}], "schema": {...}}
->  Ollama-shaped {"message": {"content": str}, "prompt_eval_count", "eval_count", "total_duration"}.
Greedy, seed 0, output constrained to `schema` by llama.cpp's grammar."""
import argparse
import json
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--image-max-tokens", type=int, default=1280,
                    help="bounds the vision encoder's compute buffer (the 8 GB fit)")
    ap.add_argument("--max-tokens", type=int, default=6144)
    a = ap.parse_args()
    out = sys.stdout
    sys.stdout = sys.stderr                  # nothing but protocol lines on the real stdout

    def send(obj):
        out.write(json.dumps(obj) + "\n")
        out.flush()

    try:
        import llama_cpp
        import llama_cpp.mtmd_cpp as mtmd_cpp
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler
        default = mtmd_cpp.mtmd_context_params_default

        def params():
            p = default()
            p.image_max_tokens = a.image_max_tokens
            return p
        mtmd_cpp.mtmd_context_params_default = params
        llm = llama_cpp.Llama(model_path=a.model, chat_handler=Qwen25VLChatHandler(a.mmproj, verbose=False),
                              n_ctx=a.ctx, n_gpu_layers=-1, n_batch=512, flash_attn=True,
                              seed=0, verbose=False)
    except Exception as e:                   # missing lib / bad model file
        send({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return
    send({"ok": True, "model": a.model, "mmproj": a.mmproj, "ctx": a.ctx})
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        msgs = []
        for m in req["messages"]:
            if m.get("images"):
                content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b}}
                           for b in m["images"]] + [{"type": "text", "text": m["content"]}]
                msgs.append({"role": m["role"], "content": content})
            else:
                msgs.append({"role": m["role"], "content": m["content"]})
        t0 = time.perf_counter()
        try:
            r = llm.create_chat_completion(
                messages=msgs, temperature=0.0, top_k=1, seed=0, repeat_penalty=1.0,
                max_tokens=a.max_tokens, response_format={"type": "json_object", "schema": req["schema"]})
            u = r.get("usage") or {}
            send({"message": {"content": r["choices"][0]["message"]["content"]},
                  "prompt_eval_count": u.get("prompt_tokens"), "eval_count": u.get("completion_tokens"),
                  "total_duration": int((time.perf_counter() - t0) * 1e9), "load_duration": 0})
        except Exception as e:
            send({"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
