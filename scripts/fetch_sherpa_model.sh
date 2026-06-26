#!/usr/bin/env bash
# Fetch the streaming STT model used by src/stt.py SherpaEngine (the default
# engine; Vosk is the offline fallback when this model is absent).
#
# One-time download into ~/.cache/sherpa-onnx (mirrors Vosk's cache). After this
# the runtime is fully offline. Override the location with SMARTPPT_SHERPA_MODEL.
set -euo pipefail

NAME="sherpa-onnx-streaming-zipformer-en-2023-06-26"
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${NAME}.tar.bz2"
DEST="${SMARTPPT_SHERPA_MODEL:-$HOME/.cache/sherpa-onnx/$NAME}"

if [ -f "$DEST/tokens.txt" ]; then
  echo "sherpa model already present at $DEST"
  exit 0
fi

mkdir -p "$(dirname "$DEST")"
tmp="$(mktemp -d)"
echo "downloading $NAME …"
wget -q "$URL" -O "$tmp/model.tar.bz2"
tar xjf "$tmp/model.tar.bz2" -C "$tmp"
mkdir -p "$DEST"
# keep only the int8 model + tokens (the deterministic, CPU-fast variant)
cp "$tmp/$NAME"/*.int8.onnx "$tmp/$NAME/tokens.txt" "$DEST/"
rm -rf "$tmp"
echo "installed sherpa model -> $DEST"
