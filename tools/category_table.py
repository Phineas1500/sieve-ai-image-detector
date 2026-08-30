#!/usr/bin/env python3
"""Per-category accuracy table for the README (audit #48): real-photo accuracy
and AI recall per held-out slice, at the shipped calibration and the fixed
0.65 threshold. Reads the logit CSVs that eval_logits.py writes.

  category_table.py --dir <results dir> --cat ft58scat2_clean_logits.csv --bench ft58s --bias 0.30
"""
import argparse
import csv
import math
from collections import defaultdict

T = math.log(0.65 / 0.35)
FACE = {"face_ffhq256", "celeb_portrait", "selfie_lowq", "press_photo"}


def acc_by_source(path, bias):
    out = defaultdict(lambda: [0, 0, 0])  # n, correct, label
    for r in csv.DictReader(open(path)):
        z = float(r["z"])
        y = int(r["label"])
        if z != z:
            continue
        pred = z + bias >= T
        a = out[r["source"]]
        a[0] += 1
        a[1] += pred == (y == 1)
        a[2] = y
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--bias", type=float, required=True)
    ap.add_argument("--dir", default=".")
    a = ap.parse_args()
    cat = acc_by_source(f"{a.dir}/{a.cat}", a.bias)
    bench = {v: acc_by_source(f"{a.dir}/{a.bench}_{v}_logits.csv", a.bias) for v in ("clean", "web", "hard")}
    print("#### Real-photo accuracy by held-out category (clean delivery)\n")
    print("| category | images | real-photo accuracy |\n|---|---|---|")
    for s, (n, c, _) in sorted((s, v) for s, v in cat.items() if v[2] == 0 and not s.endswith("_small")):
        print(f"| {s}{' — faces' if s in FACE else ''} | {n} | {100 * c / n:.1f}% |")
    print("\n#### Real-photo accuracy on the benchmark real sets, per delivery tier\n")
    print("| real set | clean | web | hard |\n|---|---|---|---|")
    for s in sorted(k for k, v in bench["clean"].items() if v[2] == 0):
        print(f"| {s} | " + " | ".join(f"{100 * bench[v][s][1] / bench[v][s][0]:.1f}%" for v in ("clean", "web", "hard")) + " |")
    print("\n#### AI recall by held-out category (clean delivery)\n")
    print("| category | images | AI recall |\n|---|---|---|")
    for s, (n, c, _) in sorted((s, v) for s, v in cat.items() if v[2] == 1 and not s.endswith("_small")):
        print(f"| {s} | {n} | {100 * c / n:.1f}% |")
    print("\n#### Small-resolution copies (180–320px shorter side, JPEG q45–80)\n")
    print("| slice | images | accuracy |\n|---|---|---|")
    for s, (n, c, _) in sorted((s, v) for s, v in cat.items() if s.endswith("_small")):
        print(f"| {s} | {n} | {100 * c / n:.1f}% |")


if __name__ == "__main__":
    main()
