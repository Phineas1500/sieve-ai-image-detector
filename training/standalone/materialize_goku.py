#!/usr/bin/env python3
"""In-the-wild Nano Banana Pro / GPT-Image-2 generations (ft6.1).

The field misses that survived ft6 are casual "snapshot" fakes delivered
through social media. Goku-OpenLab curates exactly that: prompts + the
generated images as posted on X (CC-BY-4.0), sharded under
<root>/images/<n>/<ID>_<k>.jpg with a metadata.jsonl index. Unlike the
bitmind Nano-banana-150k set (B&W identity-preserving headshots), these span
everyday scenes, selfies, products, memes — the aesthetic users report.

Samples up to `per_prompt` renders per prompt, SFW only, strided across the
whole index; writes <prefix>_* (label 1) into DATA/cartoons with the usual
tail heldout. manifests() maps by prefix.
"""
import concurrent.futures as cf
import json
import os
import shutil
import time

import requests

from materialize_eval import DATA, SCRATCH
from materialize_cartoons import manifests

SOURCES = [
    {"prefix": "nbpro", "repo": "Goku-OpenLab/nano-banana-pro-prompts-datasets",
     "train": 15000, "heldout": 1500, "per_prompt": 2},
    {"prefix": "gpt2", "repo": "Goku-OpenLab/gpt-image-2-prompts-datasets",
     "train": 15000, "heldout": 1500, "per_prompt": 2},
]


def candidates(meta_path, per_prompt):
    out = []
    with open(meta_path) as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            rating = ((e.get("spec") or {}).get("safety_rating") or "").lower()
            if rating and "safe" not in rating:
                continue
            imgs = ((e.get("media") or {}).get("images") or [])[:per_prompt]
            out.extend(p for p in imgs if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
    return out


def ingest(spec):
    from huggingface_hub import hf_hub_download

    marker = f"{DATA}/cartoons/.{spec['prefix']}_done"
    if os.path.exists(marker):
        print(f"{spec['prefix']}: done marker present, skipping")
        return
    os.makedirs(f"{DATA}/cartoons/train", exist_ok=True)
    os.makedirs(f"{DATA}/cartoons/heldout", exist_ok=True)
    cache = f"{SCRATCH}/goku_{spec['prefix']}"
    token = os.environ.get("HF_TOKEN") or None
    if not token:
        print("  WARNING: no HF_TOKEN — per-file downloads are rate-limited (HTTP 429) when anonymous", flush=True)
    meta = hf_hub_download(spec["repo"], "metadata.jsonl", repo_type="dataset", token=token, local_dir=cache)
    cands = candidates(meta, spec["per_prompt"])
    cap = spec["train"] + spec["heldout"]
    stride = max(1, len(cands) // cap)
    picked = cands[::stride][:cap]
    print(f"  {spec['prefix']}: {len(cands)} candidate images, stride {stride} -> {len(picked)}", flush=True)

    base = f"https://huggingface.co/datasets/{spec['repo']}/resolve/main/"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    sess = requests.Session()

    def fetch(job):
        i, path = job
        dest = f"{DATA}/cartoons/train/{spec['prefix']}_{i:06d}{os.path.splitext(path)[1].lower()}"
        if os.path.exists(dest):
            return dest
        for attempt in range(6):
            try:
                r = sess.get(base + path, headers=headers, timeout=60)
                if r.status_code == 200 and len(r.content) > 1024:
                    with open(dest, "wb") as f:
                        f.write(r.content)
                    return dest
                if r.status_code == 429:
                    time.sleep(float(r.headers.get("Retry-After", 15)) + attempt * 5)
                    continue
                if r.status_code in (404, 403):
                    return None
            except requests.RequestException:
                pass
            time.sleep(2 * (attempt + 1))
        return None

    written = []
    with cf.ThreadPoolExecutor(4) as ex:
        for k, dest in enumerate(ex.map(fetch, list(enumerate(picked)))):
            if dest:
                written.append(dest)
            if (k + 1) % 2000 == 0:
                print(f"  {spec['prefix']}: {k + 1}/{len(picked)} ({len(written)} ok)", flush=True)
    written.sort()
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
    print("GOKU DONE")
