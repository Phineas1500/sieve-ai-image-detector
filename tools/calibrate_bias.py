#!/usr/bin/env python3
"""Pick the calibration bias for a model from its benchmark logit CSVs:
grid over bias, maximize the WORST-tier balanced accuracy at the fixed 0.65
threshold subject to per-tier TNR >= 0.95 (the operating-point definition
used since ft44s; recorded per model in dev_model/soups/biases.txt).

  calibrate_bias.py --dir dev_model/ft10_results --tag ft10s [--grid -1:1:0.02]
"""
import argparse
import csv
import math

import numpy as np

T = math.log(0.65 / 0.35)


def tier(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r["z"]) for r in rows]), np.array([int(r["label"]) for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--grid", default="-1:1:0.02")
    ap.add_argument("--min-tnr", type=float, default=0.95)
    a = ap.parse_args()
    lo, hi, step = (float(x) for x in a.grid.split(":"))
    tiers = {v: tier(f"{a.dir}/{a.tag}_{v}_logits.csv") for v in ("clean", "web", "hard")}
    best = None
    for b in np.arange(lo, hi + 1e-9, step):
        bas, tnrs = [], []
        for v, (z, y) in tiers.items():
            pred = z + b >= T
            tpr = pred[y == 1].mean(); tnr = (~pred[y == 0]).mean()
            bas.append((tpr + tnr) / 2); tnrs.append(tnr)
        ok = min(tnrs) >= a.min_tnr
        cand = (min(bas), ok, round(float(b), 4), bas, tnrs)
        if ok and (best is None or cand[0] > best[0]):
            best = cand
    if best is None:
        print(f"no bias satisfies TNR >= {a.min_tnr} on all tiers"); return
    _, _, b, bas, tnrs = best
    print(f"{a.tag}: bias {b}")
    for (v, _), ba, tnr in zip(tiers.items(), bas, tnrs):
        print(f"  {v:5s} BA {100*ba:.1f}  TNR {100*tnr:.1f}")


if __name__ == "__main__":
    main()
