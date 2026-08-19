#!/usr/bin/env python3
"""Small-scale positives for scale-fragile generators (ft5, sweep finding):
MJ7 recall collapses 90%->35% at 256px and GPT-image-class fakes behave the
same, while blanket thumbnail *augmentation* hasn't closed it. This writes
dedicated downscaled COPIES of fragile-generator positives into the training
pool so small-resolution fingerprints are represented in the data itself.

Reads the existing openfake_train manifest (materialize_train.py must have
run), picks fragile-generator fakes, downscales to 180-320px shorter side with
JPEG q45-80, writes DATA/small_pos + manifests/small_pos.csv (label 1).
"""
import csv
import io
import os
import random

from materialize_eval import DATA, _write_manifest

FRAGILE = ("of_midjourney-7", "of_gpt-image-2", "of_gpt-image-1.5", "of_seedream-v5.0",
           "of_nano-banana-pro")
CAP = 8000
HELDOUT = 800


def main():
    from PIL import Image

    marker = f"{DATA}/small_pos/.done"
    if os.path.exists(marker):
        print("small_pos: done marker present, skipping")
        return
    with open(f"{DATA}/manifests/openfake_train.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["source"] in FRAGILE]
    rng = random.Random(5)
    rng.shuffle(rows)
    rows = rows[:CAP + HELDOUT]
    os.makedirs(f"{DATA}/small_pos/train", exist_ok=True)
    os.makedirs(f"{DATA}/small_pos/heldout", exist_ok=True)
    out = []
    for i, r in enumerate(rows):
        try:
            img = Image.open(r["path"]).convert("RGB")
        except Exception:
            continue
        t = rng.randint(180, 320)
        w, h = img.size
        if min(w, h) > t:
            s = t / min(w, h)
            img = img.resize((max(64, round(w * s)), max(64, round(h * s))), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(45, 80))
        split = "heldout" if i < HELDOUT else "train"
        p = f"{DATA}/small_pos/{split}/{r['source']}_{i:05d}.jpg"
        with open(p, "wb") as f:
            f.write(buf.getvalue())
        out.append((p, r["source"] + "_small", split))
        if len(out) % 2000 == 0:
            print(f"  {len(out)}/{len(rows)}")
    for split, name in (("train", "small_pos"), ("heldout", "small_pos_heldout")):
        rows_m = [[p, 1, src, name] for p, src, sp in out if sp == split]
        _write_manifest(f"{DATA}/manifests/{name}.csv", rows_m)
        print(f"{name}: {len(rows_m)} rows")
    open(marker, "w").close()


if __name__ == "__main__":
    main()
