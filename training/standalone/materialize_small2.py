#!/usr/bin/env python3
"""Small-resolution copies of the NEW category data (ft8).

ft7 diluted the thumbnail fix: small_pos (8.8k downscaled OpenFake positives)
was 1.7% of a 496k set and the 128px round-trip regressed. This mirrors it for
the sources added since — wild Nano Banana Pro / GPT-Image-2, edits, Ideogram,
StyleGAN faces — as small_pos2 (label 1), and adds the missing counterpart,
small_neg (label 0): downscaled + recompressed REAL photos (press, product,
low-light, portraits, FFHQ, thumbnails, COCO), so "small" stops being a cue
for "AI" (issue #52, audit #48). Heldouts come from the category heldout
split (never trained on).
"""
import csv
import os
import random
import shutil
import sys

from materialize_eval import DATA, _write_manifest
from materialize_small import downscale

POS_SOURCES = {"ai_nanobanana_pro_wild", "ai_gptimage2_wild", "ai_nanobanana", "ai_gpt_edit",
               "ai_ideogram", "ai_stylegan_face", "ai_nanobanana_edit"}
NEG_SOURCES = {"press_photo", "product_catalog", "lowlight_phone", "degraded_indoor", "selfie_lowq",
               "celeb_portrait", "yt_thumbnail", "face_ffhq256", "game_screenshot", "product_photo",
               "photo_hires_real", "photo_unsplash"}
CAP, HELDOUT, COCO_EXTRA = 8000, 800, 3000


def rows(manifest, sources):
    with open(f"{DATA}/manifests/{manifest}.csv") as f:
        return [r for r in csv.DictReader(f) if r["source"] in sources]


def build(name, label, train_rows, heldout_rows, seed):
    rng = random.Random(seed)
    for split, src_rows, cap in (("train", train_rows, CAP), ("heldout", heldout_rows, HELDOUT)):
        d = f"{DATA}/small2/{name}/{split}"
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        rng.shuffle(src_rows)
        out = []
        for i, r in enumerate(src_rows[:cap]):
            try:
                b = downscale(rng, r["path"])
            except Exception:
                continue
            p = f"{d}/{r['source']}_{i:05d}.jpg"
            with open(p, "wb") as f:
                f.write(b)
            out.append([p, label, r["source"] + "_small", name if split == "train" else name + "_heldout"])
        mname = name if split == "train" else name + "_heldout"
        _write_manifest(f"{DATA}/manifests/{mname}.csv", out)
        print(f"{mname}: {len(out)} rows")


def main():
    marker = f"{DATA}/small2/.done"
    if os.path.exists(marker) and "--force" not in sys.argv:
        print("small2: done marker present, skipping")
        return
    build("small_pos2", 1, rows("cartoons_train", POS_SOURCES), rows("cartoons_heldout", POS_SOURCES), 11)
    neg_train = rows("cartoons_train", NEG_SOURCES)
    with open(f"{DATA}/manifests/coco_train.csv") as f:
        coco = list(csv.DictReader(f))
    random.Random(12).shuffle(coco)
    neg_train += coco[:COCO_EXTRA]
    build("small_neg", 0, neg_train, rows("cartoons_heldout", NEG_SOURCES), 13)
    open(marker, "w").close()


if __name__ == "__main__":
    main()
