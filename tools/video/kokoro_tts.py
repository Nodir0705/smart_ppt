"""Kokoro-82M (Apache-2.0) one-shot synthesis for ``tools.video.voice.Kokoro``.

Runs under ~/.venvs/smartppt-kokoro (pip: kokoro-onnx), never in the repo's python.
argv = model.onnx voices.bin voice_id; stdin = JSON list of sentences; stdout = a JSON
list of byte lengths, a newline, then each sentence's int16 24 kHz mono PCM, concatenated.
The v1.0 graph has no random ops, so the same sentence gives the same samples."""
import json
import sys

import numpy as np
from kokoro_onnx import Kokoro

model, voices, vid = sys.argv[1:4]
k = Kokoro(model, voices)
pcm = [(np.clip(k.create(s, voice=vid, lang="en-us")[0], -1.0, 1.0) * 32767).astype("<i2").tobytes()
       for s in json.load(sys.stdin)]
sys.stdout.buffer.write(json.dumps([len(p) for p in pcm]).encode() + b"\n" + b"".join(pcm))
