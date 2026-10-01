"""_paddle_worker.py — runs INSIDE ~/.venvs/smartppt-paddle, never imported by
the main smartppt venv (keeps paddleocr/paddlepaddle out of the repo's deps).

Loads one PaddleOCR pipeline (model loads once, reused across every image in
--spec), OCRs each image, writes raw detections to --out. tools/ocr_bakeoff.py
(running under the regular python) reads that JSON and converts px->inches.

    python _paddle_worker.py --model mobile --spec spec.json --out out.json

spec.json: [{"key": "<slide_no or icon label>", "image": "/abs/path.png"}, ...]
out.json:  {"results": {"<key>": [{"text","poly","conf","img_w","img_h"}, ...]},
            "seconds_total": float, "model_info": {...}}
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from paddleocr import PaddleOCR
from PIL import Image

# PP-OCRv5 det+rec, English. Disable orientation/unwarping/textline-angle
# classifiers: slide PNGs are already upright — measuring det+rec only.
_MODEL_NAMES = {
    "mobile": dict(text_detection_model_name="PP-OCRv5_mobile_det",
                   text_recognition_model_name="PP-OCRv5_mobile_rec"),
    "server": dict(text_detection_model_name="PP-OCRv5_server_det",
                   text_recognition_model_name="PP-OCRv5_server_rec"),
}


def build_pipeline(model: str) -> PaddleOCR:
    return PaddleOCR(
        device="cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,  # oneDNN PIR executor crashes on this paddle/paddleocr combo (CPU-only run)
        **_MODEL_NAMES[model],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(_MODEL_NAMES), required=True)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    spec = json.loads(args.spec.read_text())
    ocr = build_pipeline(args.model)

    results = {}
    t0 = time.perf_counter()
    for item in spec:
        key, img_path = item["key"], item["image"]
        with Image.open(img_path) as im:
            img_w, img_h = im.size
        pred = ocr.predict(img_path)
        dets = []
        if pred:
            r = pred[0]
            texts = r.get("rec_texts", []) if isinstance(r, dict) else r["rec_texts"]
            scores = r.get("rec_scores", []) if isinstance(r, dict) else r["rec_scores"]
            polys = r.get("rec_polys", r.get("dt_polys", [])) if isinstance(r, dict) else r["rec_polys"]
            for text, score, poly in zip(texts, scores, polys):
                if not text:
                    continue
                dets.append({
                    "text": text,
                    "conf": float(score),
                    "poly": [[float(x), float(y)] for x, y in poly],
                    "img_w": img_w,
                    "img_h": img_h,
                })
        results[key] = dets
    elapsed = time.perf_counter() - t0

    out = {
        "results": results,
        "seconds_total": elapsed,
        "model_info": {
            "model": args.model,
            **_MODEL_NAMES[args.model],
        },
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
