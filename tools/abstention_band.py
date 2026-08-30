#!/usr/bin/env python3
"""What a distance-to-threshold "unsure" band would do on the benchmark
(agentatwork's #42 proposal): for each tier, the error rate inside score
bands around 0.65 versus outside, and the errors suppressed by abstaining on
the X% of images nearest the threshold.

  abstention_band.py --dir dev_model/soups --model ft58s --bias 0.30
"""
import argparse
import csv
import math

import numpy as np

T = 0.65


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--model", default="ft58s"); ap.add_argument("--bias", type=float, required=True)
    a = ap.parse_args()
    for view in ("clean", "web", "hard"):
        with open(f"{a.dir}/{a.model}_{view}_logits.csv") as f:
            rows = list(csv.DictReader(f))
        z = np.array([float(r["z"]) for r in rows]) + a.bias
        y = np.array([int(r["label"]) for r in rows])
        p = 1 / (1 + np.exp(-z))
        pred = p >= T
        err = pred != (y == 1)
        n = len(p)
        print(f"\n{view}: n={n} errors={err.sum()} ({100*err.mean():.2f}%)  [AI {int((y==1).sum())}, real {int((y==0).sum())}]")
        print(f"  {'band':14s} {'images':>7s} {'share':>6s} {'errors':>7s} {'err rate':>9s}  {'AI in band':>10s} {'real in band':>12s}")
        for lo, hi in ((0.50, 0.65), (0.65, 0.70), (0.65, 0.75), (0.65, 0.80), (0.55, 0.75), (0.80, 1.01)):
            m = (p >= lo) & (p < hi)
            print(f"  [{lo:.2f},{hi:.2f})   {m.sum():7d} {100*m.mean():5.1f}% {err[m].sum():7d} {100*err[m].mean() if m.any() else 0:8.1f}%  {int((y[m]==1).sum()):10d} {int((y[m]==0).sum()):12d}")
        u = np.abs(p - T)
        order = np.argsort(u)
        print("  abstain on nearest X% (by |score-0.65|): errors suppressed / total, correct verdicts suppressed, lift vs random")
        for frac in (0.05, 0.10, 0.15):
            k = int(round(frac * n)); idx = order[:k]
            caught = err[idx].sum(); ok = k - caught
            print(f"    {int(frac*100):3d}%: {caught}/{err.sum()} ({100*caught/max(1,err.sum()):.1f}%)  correct suppressed {ok} ({100*ok/n:.1f}% of all)  lift {(caught/max(1,err.sum()))/frac:.2f}   score range [{p[idx].min():.3f}, {p[idx].max():.3f}]")
        # asymmetric: only above-threshold band (what an amber "AI but near threshold" would do)
        for hi in (0.70, 0.75):
            m = (p >= T) & (p < hi)
            fp = ((y == 0) & m).sum(); tp = ((y == 1) & m).sum()
            print(f"  amber band [0.65,{hi:.2f}): {m.sum()} images = {fp} FPs rescued + {tp} TPs demoted  (all FPs at this tier: {((y==0)&pred).sum()}; TPs: {((y==1)&pred).sum()})")


if __name__ == "__main__":
    main()
