"""Preprocessing parity check: Python (PIL bilinear, CF test protocol) vs the
extension's browser pipeline (canvas resize + WebGPU), on identical files.

Usage:
  python eval/parity.py --model dev_model/commfor-model-384.onnx \
      --images eval/e2e/sample_images --browser-scores eval/e2e/scores.json
"""
import argparse
import json
import os

import numpy as np
import onnxruntime as ort
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(path, resize=440, crop=384):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = resize / min(w, h)
    img = img.resize((max(crop, round(w * s)), max(crop, round(h * s))), Image.BILINEAR)
    w, h = img.size
    x0, y0 = (w - crop) // 2, (h - crop) // 2
    img = img.crop((x0, y0, x0 + crop, y0 + crop))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--browser-scores", required=True)
    args = ap.parse_args()

    browser = {r["file"]: r["score"] for r in json.load(open(args.browser_scores))}
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    print(f"{'file':38s} {'python':>8s} {'browser':>8s} {'diff':>8s}")
    diffs = []
    for f in sorted(browser):
        p = os.path.join(args.images, f)
        logit = sess.run(None, {iname: preprocess(p)})[0].ravel()[0]
        py = float(1 / (1 + np.exp(-logit)))
        br = browser[f]
        diffs.append(abs(py - br))
        flag = "  <-- CHECK" if abs(py - br) > 0.05 else ""
        print(f"{f:38s} {py:8.4f} {br:8.4f} {abs(py - br):8.4f}{flag}")
    print(f"\nmax |diff| = {max(diffs):.4f}   mean = {np.mean(diffs):.4f}")
    if max(diffs) > 0.05:
        print("WARNING: browser/python drift exceeds 0.05 — calibrate on the browser pipeline")
    else:
        print("OK: pipelines agree; Python-side calibration transfers to the extension")


if __name__ == "__main__":
    main()
