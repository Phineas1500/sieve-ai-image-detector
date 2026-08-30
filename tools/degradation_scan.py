#!/usr/bin/env python3
"""False-trigger scan for the extension's degraded-input gate (v0.13).

Ports offscreen.js degradationStats() (luma; sampled rows; 8px-grid block
energy ratio; d12 = mean |1px luma diff| / mean |2px luma diff|, ~0.5 for
anything interpolated up, higher wherever pixel-scale texture survives) and
measures how often ordinary images would trip the gate (block >= 1.8 or
d12 < 0.528) across the
benchmark manifests under the clean / web / hard delivery views that
eval_logits.py uses. Run on the training node:

  degradation_scan.py --manifests coco,openfake_test,openfake_reddit,synthbuster,cartoons_heldout --per 500
"""
import argparse
import csv
import os
import random
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training", "standalone"))
from eval_logits import _degrade  # noqa: E402

DATA = os.path.expanduser(os.environ.get("DATA", "~/data"))
BLOCK_T, D12_T = 1.8, 0.528


def stats(img):
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = a.shape[:2]
    if w < 24 or h < 24:
        return 1.0, 1.0
    g = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    sy = max(1, h // 192)
    ys = np.arange(1, h - 1, sy)
    c = g[ys, 1:w - 1]; r = g[ys, 2:w]; l = g[ys, 0:w - 2]; u = g[ys - 1, 1:w - 1]; d = g[ys + 1, 1:w - 1]
    dx = np.abs(r - c); dy = np.abs(d - c)
    xs = np.arange(1, w - 1)
    bx = dx[:, xs % 8 == 7].mean() / (dx.mean() + 1e-6) if (xs % 8 == 7).any() else 1.0
    rows_b = ys % 8 == 7
    by = dy[rows_b].mean() / (dy.mean() + 1e-6) if rows_b.any() else 1.0
    d12 = dx.mean() / (np.abs(r - l).mean() + 1e-6)
    lap = 4 * c - l - r - u - d
    hf = lap.var() / (c.var() + 1e-6)  # the v0.13-rc gate; kept for comparison
    return float((bx + by) / 2), float(d12), float(hf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default="coco,openfake_test,openfake_reddit,synthbuster,cartoons_heldout")
    ap.add_argument("--per", type=int, default=500)
    ap.add_argument("--views", default="clean,web,hard")
    ap.add_argument("--out", default=f"{DATA}/results/degradation_scan.csv")
    a = ap.parse_args()
    rng = random.Random(0)
    rows_out = []
    for m in a.manifests.split(","):
        with open(f"{DATA}/manifests/{m}.csv") as f:
            items = list(csv.DictReader(f))
        rng.shuffle(items)
        items = items[:a.per]
        for view in a.views.split(","):
            trips = {"block": 0, "d12": 0, "any": 0}
            by_label = {0: [0, 0], 1: [0, 0]}
            n = 0
            for it in items:
                try:
                    img = _degrade(Image.open(it["path"]).convert("RGB"), view)
                except Exception:
                    continue
                b, d12, hf = stats(img)
                n += 1
                tb, th = b >= BLOCK_T, d12 < D12_T
                trips["block"] += tb; trips["d12"] += th; trips["any"] += (tb or th)
                y = int(it["label"]); by_label[y][0] += 1; by_label[y][1] += (tb or th)
                rows_out.append([m, view, it["source"], y, round(b, 3), round(d12, 4), round(hf, 4)])
            print(f"{m:18s} {view:5s} n={n:4d}  block>={BLOCK_T}: {100*trips['block']/max(n,1):5.2f}%  d12<{D12_T}: {100*trips['d12']/max(n,1):5.2f}%  any: {100*trips['any']/max(n,1):5.2f}%   "
                  f"real-trip {by_label[0][1]}/{by_label[0][0]}  ai-trip {by_label[1][1]}/{by_label[1][0]}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["manifest", "view", "source", "label", "block", "d12", "hf"]); w.writerows(rows_out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
