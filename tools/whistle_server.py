"""whistle_server.py — model side of src/stt.WhistleEngine. Runs INSIDE
~/.venvs/smartppt-whistle (pip: cactus-needle), never imported by the main process.

Binary stdin, JSON-lines stdout. First line out: {"ok": true, "weights": ...}
(or {"ok": false, "error": ...}). Then per request: a header line {"n": <bytes>}
followed by n bytes of 16 kHz mono PCM16 (<= 30 s; the engine sends <= 5.12 s)
->  {"ok": true, "text": str, "ms": float}. English, no keyword biasing.
Exits when stdin closes. The parent sets NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1
HF_HUB_OFFLINE=1 (the module-level needle API phones home by default)."""
import argparse
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--language", default="en")
    a = ap.parse_args()
    out = sys.stdout
    sys.stdout = sys.stderr                  # nothing but protocol lines on the real stdout

    def send(obj):
        out.write(json.dumps(obj) + "\n")
        out.flush()

    try:
        import numpy as np
        from needle.agent.whistle import Whistle
        model = Whistle(weights=a.weights)
    except Exception as e:                   # missing package / bad weights
        send({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return
    send({"ok": True, "weights": os.path.basename(a.weights)})

    inp = sys.stdin.buffer
    for line in iter(inp.readline, b""):
        if not line.strip():
            continue
        t0 = time.perf_counter()
        try:
            n = int(json.loads(line)["n"])
            pcm = inp.read(n)
            if len(pcm) != n:                # parent went away mid-request
                return
            audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
            text = model.transcribe(audio, language=a.language)["text"] if n else ""
            send({"ok": True, "text": text, "ms": 1000 * (time.perf_counter() - t0)})
        except Exception as e:
            send({"ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
