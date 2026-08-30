#!/usr/bin/env python3
"""Small-resolution copies of the ft10 sources (same recipe as small2):
small_pos3 (label 1) from the FLUX.1-dev renders — FLUX at 256px was the one
modern generator still missed on the independent corpus (#53) — and
small_neg3 (label 0) from the new reals (CelebA-HQ, Pexels, Open Images,
retouched), so the new real categories are also seen in the thumbnail regime.
bm-flux-offline is natively small and is not downscaled again.
"""
import os
import sys

from materialize_eval import DATA
from materialize_small2 import build, rows

POS3 = {"ai_flux_dev", "ai_flux_face"}
NEG3 = {"face_celebahq", "photo_pexels", "photo_openimages", "photo_retouched"}


def main():
    marker = f"{DATA}/small2/.done3"
    if os.path.exists(marker) and "--force" not in sys.argv:
        print("small3: done marker present, skipping")
        return
    build("small_pos3", 1, rows("cartoons_train", POS3), rows("cartoons_heldout", POS3), 30)
    build("small_neg3", 0, rows("cartoons_train", NEG3), rows("cartoons_heldout", NEG3), 31)
    open(marker, "w").close()


if __name__ == "__main__":
    main()
