#!/usr/bin/env python3
"""Uniform weight averaging ("model soup") of same-recipe checkpoints.

Single fine-tunes carry ~±1 logit of boundary noise from sampler seeds, which
flip-flops borderline exemplars run to run; averaging same-data runs cancels
that noise while keeping shared category knowledge. Only sensible for
checkpoints trained on the SAME data mix — averaging across different mixes
dilutes what only one run learned (measured, not theory).

Usage: soup_pt.py --ckpts a.pt,b.pt --out soup.pt
"""
import argparse
import os

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

sds = [torch.load(os.path.expanduser(p), map_location="cpu") for p in args.ckpts.split(",")]
sds = [s.get("model_state_dict", s) for s in sds]
avg = {}
for k in sds[0]:
    avg[k] = (sum(s[k].float() for s in sds) / len(sds)).to(sds[0][k].dtype)
out = os.path.expanduser(args.out)
os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save({"model_state_dict": avg}, out)
print(f"souped {len(sds)} checkpoints -> {out}")
