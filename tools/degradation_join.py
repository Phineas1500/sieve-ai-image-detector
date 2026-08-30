#!/usr/bin/env python3
"""Join degradation_scan.csv (seeded sample, same rng as degradation_scan.py)
with ft58s logits and report what the v0.13 degraded-input gate actually
changes at the 0.65 threshold: AI images that would go red -> amber ("unsure")
and real false positives rescued to amber. Also sweeps the d12 threshold
against synthetic small-upscale positives (the case the gate exists for).
(upscale_metric_scan.py is the wider metric comparison that picked d12.)

  BIAS=0.30 degradation_join.py
"""
import csv
import io
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from degradation_scan import stats  # noqa: E402  (block, d12, hf)

DATA = os.path.expanduser(os.environ.get("DATA", "~/data"))
T = math.log(0.65 / 0.35)
BIAS = float(os.environ.get("BIAS", "0.30"))
MANIFESTS = "coco,openfake_test,openfake_reddit,synthbuster,cartoons_heldout,small_pos2_heldout,small_neg_heldout".split(",")
BENCH = {"coco", "openfake_test", "openfake_reddit", "synthbuster"}
PER = 500
D12_TS = [0.528, 0.533, 0.541]
COL = os.environ.get("COL", "d12")
BLOCK_T = 1.8


def load_z(name):
    p = f"{DATA}/results/{name}"
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return {r["path"]: float(r["z"]) for r in csv.DictReader(f)}


def main():
    zs = {v: load_z(f"ft58s_{v}_logits.csv") for v in ("clean", "web", "hard")}
    zcat = load_z("ft58scat2_clean_logits.csv")
    rng = random.Random(0)
    sample = {}
    for m in MANIFESTS:
        with open(f"{DATA}/manifests/{m}.csv") as f:
            items = list(csv.DictReader(f))
        rng.shuffle(items)
        sample[m] = items[:PER]
    rows = defaultdict(list)
    with open(f"{DATA}/results/degradation_scan.csv") as f:
        for r in csv.DictReader(f):
            rows[(r["manifest"], r["view"])].append(r)

    print(f"gate effect at 0.65 (bias {BIAS}); amb@t = AI images scored red that the gate demotes to amber with {COL}<t; resc@t = real FPs rescued to amber")
    print(f"{'manifest':18s} {'view':5s} | AI red  " + " ".join(f"amb@{t:<5}" for t in D12_TS) + " | real FP " + " ".join(f"resc@{t:<5}" for t in D12_TS))
    for m in MANIFESTS:
        for view in ("clean", "web", "hard"):
            rs = rows.get((m, view))
            if not rs:
                continue
            items = sample[m]
            if len(rs) != len(items):
                print(f"  ! {m} {view}: {len(rs)} scan rows vs {len(items)} sampled (scan skipped some); aligning by prefix")
            z = zs[view] if m in BENCH else (zcat if view == "clean" else {})
            if not z:
                continue
            ai_red = real_fp = miss = 0
            amb = [0] * len(D12_TS)
            resc = [0] * len(D12_TS)
            for r, it in zip(rs, items):
                assert r["source"] == it["source"], (m, view, r["source"], it["source"])
                zz = z.get(it["path"])
                if zz is None:
                    miss += 1
                    continue
                red = zz + BIAS >= T
                b, v, y = float(r["block"]), float(r[COL]), int(r["label"])
                if y == 1:
                    ai_red += red
                else:
                    real_fp += red
                if not red:
                    continue
                for i, t in enumerate(D12_TS):
                    trip = b >= BLOCK_T or v < t
                    if y == 1:
                        amb[i] += trip
                    else:
                        resc[i] += trip
            print(f"{m:18s} {view:5s} | {ai_red:6d}  " + " ".join(f"{a:9d}" for a in amb) + f" | {real_fp:7d} " + " ".join(f"{a:10d}" for a in resc) + (f"  (no score for {miss})" if miss else ""))

    print(f"\nsynthetic upscale positives: coco clean sample (200), downscaled to side s and bicubic-upscaled back; fraction with {COL} < t")
    imgs = sample["coco"][:200]
    for jpeg in (None, 75):
        print(f"  {'raw upscale' if jpeg is None else f'upscale + JPEG q{jpeg}'}")
        for s in (64, 96, 128, 160, 192, 256):
            hfs = []
            for it in imgs:
                im = Image.open(it["path"]).convert("RGB")
                w, h = im.size
                sc = s / min(w, h)
                if sc >= 1:
                    continue
                small = im.resize((max(1, round(w * sc)), max(1, round(h * sc))), Image.BICUBIC)
                up = small.resize((w, h), Image.BICUBIC)
                if jpeg:
                    buf = io.BytesIO(); up.save(buf, "JPEG", quality=jpeg); buf.seek(0); up = Image.open(buf).convert("RGB")
                hfs.append(stats(up)[1 if COL == "d12" else 2])
            hfs = np.array(hfs)
            print(f"    side {s:3d}: n={len(hfs):3d} " + " ".join(f"<{t}: {100 * (hfs < t).mean():5.1f}%" for t in D12_TS) + f"  median {np.median(hfs):.4f}")

    print(f"\nnative {COL} percentiles p1/p5/p10/p25 (clean view):")
    for m in MANIFESTS:
        rs = rows.get((m, "clean"))
        if not rs:
            continue
        for lab in (0, 1):
            h = np.array([float(r[COL]) for r in rs if int(r["label"]) == lab])
            if len(h) == 0:
                continue
            print(f"  {m:18s} label {lab} n={len(h):3d}: " + "  ".join(f"{np.percentile(h, p):.4f}" for p in (1, 5, 10, 25)))


if __name__ == "__main__":
    main()
