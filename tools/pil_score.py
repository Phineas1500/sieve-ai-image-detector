#!/usr/bin/env python3
"""Reference scorer: Pillow preprocessing (resize shorter side 440 bilinear,
center-crop 384, ImageNet normalize) + onnxruntime, calibrated score =
sigmoid(z + bias). The extension's resampler is built to match this
pipeline byte-for-byte; eval/e2e/parity-test.mjs checks that.

  pil_score.py --model M --bias B pilref  > eval/e2e/pil_reference_<tag>.json
  pil_score.py --model M --bias B battery            # audit #41/#45 transforms
  pil_score.py --model M --bias B files a.jpg b.png  # one line per file
"""
import argparse, glob, io, json, os, sys
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageEnhance, ImageOps

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "..", "eval", "e2e", "sample_images")


class Scorer:
    def __init__(self, model, bias):
        self.sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name
        self.bias = bias

    def logit(self, img):
        img = img.convert("RGB")
        w, h = img.size
        s = 440 / min(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
        w, h = img.size
        l, t = (w - 384) // 2, (h - 384) // 2
        x = np.asarray(img.crop((l, t, l + 384, t + 384)), dtype=np.float32) / 255.0
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
        return float(self.sess.run(None, {self.iname: x})[0].ravel()[0])

    def score(self, img):
        return float(1 / (1 + np.exp(-(self.logit(img) + self.bias))))


def jpeg(img, q):
    buf = io.BytesIO(); img.convert("RGB").save(buf, "JPEG", quality=q); buf.seek(0)
    return Image.open(buf).convert("RGB")

def deepfry(img):
    out = img.convert("RGB")
    for _ in range(6):
        out = jpeg(ImageEnhance.Color(ImageEnhance.Contrast(out).enhance(1.25)).enhance(1.3), 12)
    return out

def thumb_up(img, t):
    w, h = img.size; s = t / min(w, h)
    small = jpeg(img.convert("RGB").resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR), 70)
    return small.resize((w, h), Image.BILINEAR)

def resize_max(img, m):
    w, h = img.size
    if max(w, h) <= m: return img.convert("RGB")
    s = m / max(w, h)
    return jpeg(img.convert("RGB").resize((round(w * s), round(h * s)), Image.BILINEAR), 85)

def collage2(img):
    w, h = img.size; out = Image.new("RGB", (w * 2, h * 2))
    for dx in (0, w):
        for dy in (0, h): out.paste(img.convert("RGB"), (dx, dy))
    return jpeg(out, 85)

def letterbox(img):
    w, h = img.size; tw, th = w, round(w * 9 / 16)
    if th < h: tw, th = round(h * 16 / 9), h
    out = Image.new("RGB", (tw, th)); out.paste(img.convert("RGB"), ((tw - w) // 2, (th - h) // 2))
    return out

REAL_T = [("native", lambda i: i), ("deepfry", deepfry), ("resize512", lambda i: resize_max(i, 512)),
          ("jpeg_q40", lambda i: jpeg(i, 40)), ("corner60", lambda i: i.convert("RGB").crop((0, 0, round(i.size[0] * .6), round(i.size[1] * .6)))),
          ("collage2x", collage2), ("letterbox", letterbox), ("up64", lambda i: thumb_up(i, 64))]
AI_T = [("native", lambda i: i), ("thumb128", lambda i: jpeg(thumb_up(i, 128), 80)),
        ("grayscale", lambda i: ImageOps.grayscale(i).convert("RGB")), ("jpeg_q40", lambda i: jpeg(i, 40))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--bias", type=float, required=True)
    ap.add_argument("mode", choices=["pilref", "battery", "files"]); ap.add_argument("paths", nargs="*")
    a = ap.parse_args()
    sc = Scorer(a.model, a.bias)
    if a.mode == "pilref":
        files = sorted(f for f in os.listdir(SAMPLES) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
        out = {"sample_images": [{"file": f, "score": sc.score(Image.open(os.path.join(SAMPLES, f)))} for f in files]}
        json.dump(out, sys.stdout, indent=1)
    elif a.mode == "battery":
        for label, pat, battery in (("REAL", "0000005*.jpg", REAL_T), ("AI", "fake_*.png", AI_T)):
            for f in sorted(glob.glob(os.path.join(SAMPLES, pat)))[:4]:
                img = Image.open(f)
                print(f"{label} {os.path.basename(f):24s} " + " ".join(f"{n}={sc.score(t(img)):.3f}" for n, t in battery))
    else:
        for p in a.paths:
            try: print(f"{sc.score(Image.open(p)):.4f}\t{p}")
            except Exception as e: print(f"ERR\t{p}\t{e}")


if __name__ == "__main__":
    main()
