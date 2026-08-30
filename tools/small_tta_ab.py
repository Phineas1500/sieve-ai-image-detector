#!/usr/bin/env python3
"""A/B of the v0.13 small-image TTA (fit view averaged with the standard view
for in-band images below min_side) on the local small held-out sets scored
through the real extension (eval/e2e/score-dir.mjs, with and without
--no-small-tta). Reports accuracy at 0.65 per slice and on the TTA-fired
subset only (the rest is identical by construction).

  small_tta_ab.py dev_model/small_heldout
"""
import json
import os
import sys

T = 0.65
SLICES = [("ext_small2_small_pos2_heldout", 1), ("ext_small2_small_neg_heldout", 0), ("ext_small_pos_heldout", 1)]


def load(p):
    with open(p) as f:
        d = json.load(f)
    return d["results"] if isinstance(d, dict) and "results" in d else d


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "dev_model/small_heldout"
    tot = {"tta": [0, 0], "notta": [0, 0]}
    for tag, y in SLICES:
        a, b = f"{d}/{tag}_tta.json", f"{d}/{tag}_notta.json"
        if not (os.path.exists(a) and os.path.exists(b)):
            print(f"{tag}: pending"); continue
        A, B = load(a), load(b)
        keys = [k for k in A if A.get(k) and B.get(k) and A[k].get("score") is not None and B[k].get("score") is not None]
        acc = lambda D: sum((D[k]["score"] >= T) == (y == 1) for k in keys) / max(1, len(keys))
        fired = [k for k in keys if A[k].get("tta")]
        accf = lambda D: sum((D[k]["score"] >= T) == (y == 1) for k in fired) / max(1, len(fired))
        print(f"{tag:32s} n={len(keys):4d}  acc notta {100*acc(B):5.1f}%  tta {100*acc(A):5.1f}%  ({100*(acc(A)-acc(B)):+.1f})   | tta fired on {len(fired):3d}: notta {100*accf(B):5.1f}% -> tta {100*accf(A):5.1f}%")
        tot["tta"][0] += sum((A[k]["score"] >= T) == (y == 1) for k in keys); tot["tta"][1] += len(keys)
        tot["notta"][0] += sum((B[k]["score"] >= T) == (y == 1) for k in keys); tot["notta"][1] += len(keys)
    if tot["tta"][1]:
        print(f"{'all':32s} n={tot['tta'][1]:4d}  acc notta {100*tot['notta'][0]/tot['notta'][1]:5.1f}%  tta {100*tot['tta'][0]/tot['tta'][1]:5.1f}%")


if __name__ == "__main__":
    main()
