"""Model side of the vision bake-off. Runs under ~/.venvs/smartppt-vision (GPU, fp16,
greedy, offline). One (model, task) per process; writes raw/<file>.json + raw/meta/<file>.json
(pass 2 -> raw/pass2/<file>.json, used only for the determinism diff).

    ~/.venvs/smartppt-vision/bin/python tools/vision_runner.py --model florence2-base --task ocr
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # reproducible offline: cached revisions only
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import vision_bakeoff as vb  # noqa: E402

REPOS = {
    "florence2-base": "microsoft/Florence-2-base",
    "florence2-large": "microsoft/Florence-2-large",
    "smolvlm-256m": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "smolvlm-500m": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "smolvlm2-2.2b": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "vit-base": "google/vit-base-patch16-224",
    "clip-b32": "openai/clip-vit-base-patch32",              # ships pytorch_model.bin only
    "clip-b32-laion": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",  # open_clip ViT-B-32 weights
    "yolov8-doclaynet": "hantian/yolo-doclaynet",
}
YOLO_WEIGHTS = "yolov8m-doclaynet.pt"
OCR_PROMPT = "Read all text in this slide."
CAP_PROMPT = "Describe this image in one short phrase."
OCR_MAX_NEW = 1024
CAP_MAX_NEW = 40


def seed_all() -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def crop(job) -> Image.Image:
    return Image.open(job["png"]).convert("RGB").crop(tuple(job["crop"]))


# ------------------------------------------------------------------ models

class Florence:
    def __init__(self, repo, path):
        from unittest.mock import patch
        from transformers import AutoModelForCausalLM, AutoProcessor
        from transformers.dynamic_module_utils import get_imports

        def no_flash(f):  # remote code lists flash_attn as a hard import; it is optional
            return [i for i in get_imports(f) if i != "flash_attn"]
        with patch("transformers.dynamic_module_utils.get_imports", no_flash):
            self.m = AutoModelForCausalLM.from_pretrained(
                path, trust_remote_code=True, torch_dtype=torch.float16,
                attn_implementation="eager").cuda().eval()
        self.p = AutoProcessor.from_pretrained(path, trust_remote_code=True)

    @torch.inference_mode()
    def __call__(self, im, task, max_new=1024):
        inp = self.p(text=task, images=im, return_tensors="pt").to("cuda", torch.float16)
        g = self.m.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"],
                            max_new_tokens=max_new, num_beams=1, do_sample=False)
        txt = self.p.batch_decode(g, skip_special_tokens=False)[0]
        return self.p.post_process_generation(txt, task=task, image_size=im.size)[task]


class SmolVLM:
    def __init__(self, repo, path):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.p = AutoProcessor.from_pretrained(path)
        self.m = AutoModelForImageTextToText.from_pretrained(
            path, torch_dtype=torch.float16, _attn_implementation="eager").cuda().eval()

    @torch.inference_mode()
    def __call__(self, im, prompt, max_new):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.p.apply_chat_template(msgs, add_generation_prompt=True)
        inp = self.p(text=text, images=[im], return_tensors="pt").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].half()
        g = self.m.generate(**inp, max_new_tokens=max_new, do_sample=False, num_beams=1)
        new = g[0, inp["input_ids"].shape[1]:]
        return self.p.decode(new, skip_special_tokens=True).strip(), len(new) >= max_new


# ------------------------------------------------------------------- tasks

def ocr_jobs(decks, limit):
    jobs = []
    for deck, t in decks.items():
        for s in t["slides"]:
            png = vb.slide_png(t, s["slide_number"])
            w, h = Image.open(png).size
            jobs.append({"deck": deck, "slide": str(s["slide_number"]), "png": str(png),
                         "scale": vb.px_per_inch(w, h, t["slide_size"])})
    return jobs[:limit] if limit else jobs


def timed(jobs, fn):
    """Run fn(job) over jobs; seconds/item excludes the first (warm-up) call."""
    out, times = [], []
    for j in jobs:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out.append(fn(j))
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    rest = times[1:] or times
    return out, sum(rest) / len(rest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=REPOS)
    ap.add_argument("--task", required=True, choices=["ocr", "caption", "clip", "layout"])
    ap.add_argument("--pass", dest="pass_no", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prompt", default=OCR_PROMPT, help="SmolVLM OCR prompt (sensitivity check)")
    ap.add_argument("--suffix", default="", help="appended to the output engine name")
    a = ap.parse_args()
    seed_all()
    repo = REPOS[a.model]
    path = snapshot_download(repo, local_files_only=True)
    revision = Path(path).name
    decks = vb.load_decks()
    results: dict[str, dict] = {}   # file stem -> payload
    meta: dict = {"model": a.model, "repo": repo, "revision": revision, "task": a.task,
                  "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}

    if a.model.startswith("florence"):
        M = Florence(repo, path)
        if a.task == "ocr":
            def f(j):
                im = Image.open(j["png"]).convert("RGB")
                r = M(im, "<OCR_WITH_REGION>")
                return [{"text": lab.replace("</s>", "").strip(),
                         "bbox": vb.px_to_inch_box(*vb.quad_to_xyxy(q), j["scale"]), "conf": None}
                        for q, lab in zip(r["quad_boxes"], r["labels"])
                        if lab.replace("</s>", "").strip()]
            jobs = ocr_jobs(decks, a.limit)
            outs, sec = timed(jobs, f)
            results[a.model] = ocr_payload(jobs, outs, sec)
            meta.update(unit="line", has_boxes=True, prompt="<OCR_WITH_REGION>")
        elif a.task == "caption":
            jobs = img_jobs(decks, a.limit)
            outs, sec = timed(jobs, lambda j: {"caption": M(crop(j), "<CAPTION>", 64).strip(), "conf": None})
            results[f"captions_{a.model}"] = cap_payload(jobs, outs)
            meta.update(prompt="<CAPTION>")
        elif a.task == "layout":
            jobs = ocr_jobs(decks, a.limit)
            for tag, task in (("od", "<OD>"), ("rp", "<REGION_PROPOSAL>")):
                def f(j, task=task):
                    r = M(Image.open(j["png"]).convert("RGB"), task)
                    return [{"bbox": vb.px_to_inch_box(*b, j["scale"]), "label": lab, "conf": None}
                            for b, lab in zip(r["bboxes"], r["labels"])]
                outs, sec_t = timed(jobs, f)
                results[f"layout_{a.model}-{tag}"] = layout_payload(jobs, outs)
                meta[f"seconds_per_item_{tag}"] = sec_t
            sec = None
        else:
            raise SystemExit(f"{a.model} has no task {a.task}")

    elif a.model.startswith("smolvlm"):
        M = SmolVLM(repo, path)
        if a.task == "ocr":
            capped = 0

            def f(j):
                nonlocal capped
                txt, hit = M(Image.open(j["png"]).convert("RGB"), a.prompt, OCR_MAX_NEW)
                capped += hit
                return [{"text": ln.strip(), "bbox": None, "conf": None}
                        for ln in txt.splitlines() if ln.strip()]
            jobs = ocr_jobs(decks, a.limit)
            outs, sec = timed(jobs, f)
            results[a.model + a.suffix] = ocr_payload(jobs, outs, sec)
            meta.update(unit="line", has_boxes=False, prompt=a.prompt, max_new_tokens=OCR_MAX_NEW,
                        hit_token_cap=f"{capped}/{len(jobs)}")
        elif a.task == "caption":
            jobs = img_jobs(decks, a.limit)
            outs, sec = timed(jobs, lambda j: {"caption": M(crop(j), CAP_PROMPT, CAP_MAX_NEW)[0], "conf": None})
            results[f"captions_{a.model}"] = cap_payload(jobs, outs)
            meta.update(prompt=CAP_PROMPT)
        else:
            raise SystemExit(f"{a.model} has no task {a.task}")

    elif a.model == "vit-base":
        from transformers import ViTForImageClassification, ViTImageProcessor
        p = ViTImageProcessor.from_pretrained(path)
        m = ViTForImageClassification.from_pretrained(path, torch_dtype=torch.float16).cuda().eval()

        @torch.inference_mode()
        def f(j):
            x = p(images=crop(j), return_tensors="pt")["pixel_values"].cuda().half()
            pr = m(pixel_values=x).logits.float().softmax(-1)[0]
            top = pr.topk(3)
            labs = [m.config.id2label[int(i)].split(",")[0].strip() for i in top.indices]
            return {"caption": ", ".join(labs), "conf": round(float(top.values[0]), 4)}
        jobs = img_jobs(decks, a.limit)
        outs, sec = timed(jobs, f)
        results[f"captions_{a.model}"] = cap_payload(jobs, outs)
        meta.update(prompt="top-3 ImageNet labels")

    elif a.model.startswith("clip-b32"):
        from transformers import CLIPModel, CLIPProcessor
        p = CLIPProcessor.from_pretrained(path)
        m = CLIPModel.from_pretrained(path, torch_dtype=torch.float16).cuda().eval()

        @torch.inference_mode()
        def f(j):
            ids, texts = zip(*j["candidates"])
            inp = p(text=list(texts), images=crop(j), return_tensors="pt", padding=True,
                    truncation=True, max_length=77).to("cuda")
            inp["pixel_values"] = inp["pixel_values"].half()
            o = m(**inp)
            sims = o.logits_per_image[0].float().tolist()
            return {"scores": {i: round(s, 4) for i, s in zip(ids, sims)}}
        jobs = [j for j in img_jobs(decks, None) if j["set"] == "content" and len(j["candidates"]) >= 1]
        jobs = jobs[:a.limit] if a.limit else jobs
        outs, sec = timed(jobs, f)
        payload: dict = {}
        for j, o in zip(jobs, outs):
            payload.setdefault(j["deck"], {})[j["id"]] = {"slide": j["slide"], **o}
        results[f"clip_{a.model}"] = payload

    elif a.model == "yolov8-doclaynet":
        from ultralytics import YOLO
        m = YOLO(str(Path(path) / YOLO_WEIGHTS))

        def f(j):
            r = m.predict(j["png"], imgsz=1024, conf=0.25, device=0, half=True, verbose=False)[0]
            return [{"bbox": vb.px_to_inch_box(*map(float, b), j["scale"]),
                     "label": r.names[int(c)], "conf": round(float(s), 4)}
                    for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        jobs = ocr_jobs(decks, a.limit)
        outs, sec = timed(jobs, f)
        results[f"layout_{a.model}"] = layout_payload(jobs, outs)
        meta.update(weights=YOLO_WEIGHTS, imgsz=1024, conf=0.25)
    else:
        raise SystemExit(a.model)

    meta["vram_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
    raw = vb.OUT / "raw"
    dest = raw if a.pass_no == 1 else raw / "pass2"
    for stem, payload in results.items():
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"{stem}.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
        if a.pass_no == 1:
            (raw / "meta").mkdir(parents=True, exist_ok=True)
            m_ = dict(meta, seconds_per_item=sec if sec is not None else meta.get(
                f"seconds_per_item_{stem.rsplit('-', 1)[-1]}"))
            (raw / "meta" / f"{stem}.json").write_text(json.dumps(m_, indent=1))
        print(f"wrote {dest / stem}.json  vram={meta['vram_peak_mb']:.0f}MB", flush=True)
    return 0


def img_jobs(decks, limit):
    jobs = vb.image_jobs(decks)
    return jobs[:limit] if limit else jobs


def ocr_payload(jobs, outs, sec):
    d: dict = {"_unit": "line", "_seconds_per_slide": sec}
    for j, o in zip(jobs, outs):
        d.setdefault(j["deck"], {})[j["slide"]] = o
    return d


def cap_payload(jobs, outs):
    d: dict = {}
    for j, o in zip(jobs, outs):
        d.setdefault(j["deck"], {})[j["id"]] = {"slide": j["slide"], "set": j["set"], **o}
    return d


def layout_payload(jobs, outs):
    d: dict = {}
    for j, o in zip(jobs, outs):
        d.setdefault(j["deck"], {})[j["slide"]] = o
    return d


if __name__ == "__main__":
    sys.exit(main())
