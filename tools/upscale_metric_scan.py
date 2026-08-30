#!/usr/bin/env python3
"""Which cheap pixel statistic best separates upscaled-small images (the case
the v0.13 degraded-input gate exists for) from native images, including
smooth AI renders? This is the comparison that replaced the Laplacian
"hf" gate with d12 before v0.13.0 shipped.

Negatives: native clean + web views of benchmark/category samples.
Positives: the same images downscaled to side s and bicubic-upscaled back
(raw / JPEG q75). Reports, per metric, the catch rate on each positive
condition at the threshold that trips <=1/2/5% of negatives, plus the trip
rate on the smooth openfake_reddit AI subset at that threshold.

  PER=150 NPROC=16 upscale_metric_scan.py     # on the training node, ~2 min

Result (2026-08-30, 1050 images, 2100 negatives): at 1% native false trips
  hf   (lapVar/globalVar)      demotes 8.5% clean / 3.2% web of reddit AI; catches 82% of 64px raw upscales
  lg   (lapVar/gradVar)        3.2% / 3.2%; 96% of 64px raw
  d12  (mean|d1| / mean|d2|)   2.1% / 1.1%; 99% of 64px raw, 95.5% of 64px q75, 79% of 128px raw  <- shipped, t=0.528
  p99, p999 (peak |lap| / sd)  9.6-11.7% / 2.1-3.2%; 83-92% of 64px raw
"""
import csv
import io
import os
import random
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training", "standalone"))
from eval_logits import _degrade  # noqa: E402

DATA = os.path.expanduser(os.environ.get("DATA", "~/data"))
MANIFESTS = "coco,openfake_test,openfake_reddit,synthbuster,cartoons_heldout,small_pos2_heldout,small_neg_heldout".split(",")
PER = int(os.environ.get("PER", "150"))
SIDES = (64, 96, 128, 160)
METRICS = ["hf", "lg", "d12", "p99", "p999"]


def metrics(img):
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = a.shape[:2]
    if w < 24 or h < 24:
        return None
    g = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    sy = max(1, h // 192)
    ys = np.arange(1, h - 1, sy)
    c = g[ys, 1:w - 1]; r = g[ys, 2:w]; l = g[ys, 0:w - 2]; u = g[ys - 1, 1:w - 1]; d = g[ys + 1, 1:w - 1]
    lap = 4 * c - l - r - u - d
    cv = c.var() + 1e-6
    sd = np.sqrt(cv)
    dx = r - c; dy = d - c
    al = np.abs(lap)
    return dict(
        hf=lap.var() / cv,
        lg=lap.var() / (dx.var() + dy.var() + 1e-6),
        d12=np.abs(dx).mean() / (np.abs(r - l).mean() + 1e-6),
        p99=np.percentile(al, 99) / sd,
        p999=np.percentile(al, 99.9) / sd,
    )


def work(args):
    m, it = args
    out = []
    try:
        im = Image.open(it["path"]).convert("RGB")
    except Exception:
        return out
    y = int(it["label"])
    for view in ("clean", "web"):
        mm = metrics(_degrade(im, view))
        if mm:
            out.append((m, it["source"], y, f"native_{view}", mm))
    w, h = im.size
    for s in SIDES:
        sc = s / min(w, h)
        if sc >= 1:
            continue
        small = im.resize((max(1, round(w * sc)), max(1, round(h * sc))), Image.BICUBIC)
        up = small.resize((w, h), Image.BICUBIC)
        for jpeg in (None, 75):
            x = up
            if jpeg:
                buf = io.BytesIO(); up.save(buf, "JPEG", quality=jpeg); buf.seek(0); x = Image.open(buf).convert("RGB")
            mm = metrics(x)
            if mm:
                out.append((m, it["source"], y, f"up{s}_{'raw' if jpeg is None else 'q' + str(jpeg)}", mm))
    return out


def main():
    rng = random.Random(0)
    tasks = []
    for m in MANIFESTS:
        with open(f"{DATA}/manifests/{m}.csv") as f:
            items = list(csv.DictReader(f))
        rng.shuffle(items)
        tasks += [(m, it) for it in items[:PER]]
    rows = []
    with Pool(int(os.environ.get("NPROC", str(os.cpu_count() or 8)))) as p:
        for i, out in enumerate(p.imap_unordered(work, tasks, chunksize=4)):
            rows += out
            if i % 100 == 0:
                print(f"{i}/{len(tasks)}", flush=True)
    with open(f"{DATA}/results/upscale_metrics.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["manifest", "source", "label", "cond"] + METRICS)
        for m, src, y, cond, mm in rows:
            w.writerow([m, src, y, cond] + [f"{mm[k]:.5f}" for k in METRICS])
    neg = [r for r in rows if r[3].startswith("native")]
    conds = sorted({r[3] for r in rows if not r[3].startswith("native")}, key=lambda c: (int(c[2:].split("_")[0]), c))
    print(f"\nnegatives n={len(neg)} (native clean+web over {len(tasks)} images); positives per condition ~{len(tasks)}")
    for k in METRICS:
        nv = np.array([r[4][k] for r in neg])
        print(f"\nmetric {k}: trip when value < t")
        for fp in (0.01, 0.02, 0.05):
            t = np.percentile(nv, 100 * fp)
            line = f"  fp={fp:.2f} t={t:.4f}"
            for sub in ("native_clean", "native_web"):
                rv = np.array([r[4][k] for r in neg if r[0] == "openfake_reddit" and r[2] == 1 and r[3] == sub])
                line += f"  reddit-AI {sub[7:]}: {100 * (rv < t).mean():4.1f}%"
            print(line)
            print("      catch: " + "  ".join(f"{c}: {100 * (np.array([r[4][k] for r in rows if r[3] == c]) < t).mean():5.1f}%" for c in conds))
            per = {}
            for r in neg:
                per.setdefault((r[0], r[2]), []).append(r[4][k] < t)
            print("      native trips: " + "  ".join(f"{m}/{y}: {100 * np.mean(v):4.1f}%" for (m, y), v in sorted(per.items())))
    print("METRIC-SCAN-DONE")


if __name__ == "__main__":
    main()
