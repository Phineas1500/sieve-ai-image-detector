#!/usr/bin/env python3
"""Zip-packaged AI positives (ft6): bitmind/Nano-banana-150k and
bitmind/ideogram-27k ship as a single zip on the main branch; the HF parquet
preview carries no pixels, so the parquet ingest yields nothing. Download the
zip to SCRATCH, walk members in sorted order with a stride, write with the
usual prefix + tail heldout into DATA/cartoons (manifests() maps by prefix).

Nano-banana-150k is an identity-preserving EDIT set: output/<edit_type>/ are
the Nano Banana edits (positives); orignal/ are the source photos, whose
provenance is not documented, so they are not used as reals.
"""
import os
import re
import shutil
import zipfile

from materialize_eval import DATA, SCRATCH
from materialize_cartoons import manifests

SOURCES = [
    {"prefix": "nb150k", "repo": "bitmind/Nano-banana-150k", "file": "AI.zip",
     "member_re": r"/output/.*\.(jpe?g|png|webp)$", "stride": 3, "train": 20000, "heldout": 2000},
    {"prefix": "ideo", "repo": "bitmind/ideogram-27k", "file": "ideogram-27k.zip",
     "member_re": r"\.(jpe?g|png|webp)$", "stride": 2, "train": 6000, "heldout": 600},
    # ft10: CC0 Pexels photography at 768p (lifestyle / smartphone-look reals,
    # the clean-photo-fp category). Despite the name the zip holds only the
    # photos (109,971 JPEGs under images/) plus an attributes json.
    {"prefix": "pexl", "repo": "cj-mills/pexels-110k-768p-min-jpg-depth-anything-large-hf",
     "file": "pexels-110k-768p-min-jpg-depth-anything-large-hf.zip",
     "member_re": r"/images/.*\.jpe?g$", "stride": 12, "train": 8000, "heldout": 800},
]


def ingest(spec):
    from huggingface_hub import hf_hub_download

    marker = f"{DATA}/cartoons/.{spec['prefix']}_zip_done"
    if os.path.exists(marker):
        print(f"{spec['prefix']}: done marker present, skipping")
        return
    os.makedirs(f"{DATA}/cartoons/train", exist_ok=True)
    os.makedirs(f"{DATA}/cartoons/heldout", exist_ok=True)
    cache = f"{SCRATCH}/zip_{spec['prefix']}"
    local = hf_hub_download(spec["repo"], spec["file"], repo_type="dataset",
                            token=os.environ.get("HF_TOKEN") or None, local_dir=cache)
    print(f"  {spec['prefix']}: downloaded {os.path.getsize(local) / 1e9:.1f} GB", flush=True)
    pat = re.compile(spec["member_re"], re.I)
    cap = spec["train"] + spec["heldout"]
    written = []
    with zipfile.ZipFile(local) as z:
        members = sorted(n for n in z.namelist() if pat.search(n))
        print(f"  {spec['prefix']}: {len(members)} matching members, stride {spec['stride']}", flush=True)
        for n in members[::spec["stride"]]:
            if len(written) >= cap:
                break
            b = z.read(n)
            if len(b) < 1024:
                continue
            p = f"{DATA}/cartoons/train/{spec['prefix']}_{len(written):06d}{os.path.splitext(n)[1].lower()}"
            with open(p, "wb") as f:
                f.write(b)
            written.append(p)
            if len(written) % 2000 == 0:
                print(f"  {spec['prefix']}: {len(written)}/{cap}", flush=True)
    shutil.rmtree(cache, ignore_errors=True)
    ho = min(spec["heldout"], len(written) // 5)
    for p in written[-ho:] if ho else []:
        shutil.move(p, f"{DATA}/cartoons/heldout/{os.path.basename(p)}")
    open(marker, "w").close()
    print(f"{spec['prefix']}: {len(written) - ho} train / {ho} heldout")


if __name__ == "__main__":
    import traceback
    for spec in SOURCES:
        try:
            ingest(spec)
        except Exception:
            traceback.print_exc()
            print(f"FAILED: {spec['prefix']}")
    manifests()
    print("ZIP DONE")
