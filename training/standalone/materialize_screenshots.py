#!/usr/bin/env python3
"""Screenshot hard-real data for ft3 (runs on the training node).

  - wave-ui-25k real UI screenshots: first N shards -> train, LAST shard ->
    held-out eval (shard-disjoint split)
  - synthesized chat screenshots: downloaded tar (generated locally by
    eval/e2e/chat-screenshot-synth.mjs) -> train + held-out slice

Manifests: screenshots_train.csv (label 0), screenshots_heldout.csv (label 0).
Env: DATA, SCRATCH, HF_TOKEN as usual.
Usage: python3 -u materialize_screenshots.py [--synth-url URL]
"""
import argparse
import os
import random
import shutil
import tarfile

from materialize_eval import DATA, SCRATCH, _download, _sniff_ext, _write_manifest

TRAIN_CAP = 12000
HELDOUT_CAP = 1000
SYNTH_HELDOUT = 300


def waveui():
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    marker = f"{DATA}/screenshots/.waveui_done"
    if os.path.exists(marker):
        print("waveui: done marker present, skipping")
        return
    token = os.environ.get("HF_TOKEN") or None
    files = sorted(
        f for f in HfApi(token=token).list_repo_files("agentsea/wave-ui-25k", repo_type="dataset")
        if f.startswith("data/train-") and f.endswith(".parquet")
    )
    train_shards, heldout_shard = files[:-1], files[-1]
    os.makedirs(f"{DATA}/screenshots/train", exist_ok=True)
    os.makedirs(f"{DATA}/screenshots/heldout", exist_ok=True)

    def extract(shards, outdir, cap, prefix):
        n = 0
        for fpath in shards:
            if n >= cap:
                break
            local = hf_hub_download("agentsea/wave-ui-25k", fpath, repo_type="dataset",
                                    token=token, local_dir=f"{SCRATCH}/waveui")
            pf = pq.ParquetFile(local)
            for batch in pf.iter_batches(batch_size=32, columns=["image"]):
                col = batch.column(0)
                for i in range(len(col)):
                    if n >= cap:
                        break
                    b = col[i]["bytes"].as_py()
                    if b is None:
                        continue
                    n += 1
                    with open(f"{outdir}/{prefix}_{n:06d}{_sniff_ext(b)}", "wb") as f:
                        f.write(b)
            os.remove(local)
            print(f"  waveui {prefix}: {fpath} done ({n}/{cap})")
        return n

    nt = extract(train_shards, f"{DATA}/screenshots/train", TRAIN_CAP, "waveui")
    nh = extract([heldout_shard], f"{DATA}/screenshots/heldout", HELDOUT_CAP, "waveui_ho")
    shutil.rmtree(f"{SCRATCH}/waveui", ignore_errors=True)
    open(marker, "w").close()
    print(f"waveui: {nt} train / {nh} heldout")


def synth(url):
    marker = f"{DATA}/screenshots/.synth_done"
    if os.path.exists(marker):
        print("synth: done marker present, skipping")
        return
    tar_path = f"{SCRATCH}/synth_chats.tar.gz"
    _download(url, tar_path, "synthesized chat screenshots")
    tmp = f"{SCRATCH}/synth_extract"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    with tarfile.open(tar_path) as t:
        t.extractall(tmp, filter="data")
    imgs = []
    for root, _, files in os.walk(tmp):
        imgs += [os.path.join(root, f) for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    random.Random(0).shuffle(imgs)
    os.makedirs(f"{DATA}/screenshots/train", exist_ok=True)
    os.makedirs(f"{DATA}/screenshots/heldout", exist_ok=True)
    for i, p in enumerate(imgs):
        dest = "heldout" if i < SYNTH_HELDOUT else "train"
        shutil.copy(p, f"{DATA}/screenshots/{dest}/synth_{i:05d}{os.path.splitext(p)[1]}")
    shutil.rmtree(tmp, ignore_errors=True)
    os.remove(tar_path)
    open(marker, "w").close()
    print(f"synth: {len(imgs) - SYNTH_HELDOUT} train / {SYNTH_HELDOUT} heldout")


def manifests():
    for split, name in (("train", "screenshots_train"), ("heldout", "screenshots_heldout")):
        d = f"{DATA}/screenshots/{split}"
        rows = [[os.path.join(d, f), 0,
                 "screenshot_synth" if f.startswith("synth") else "screenshot_ui", name]
                for f in sorted(os.listdir(d)) if not f.startswith(".")]
        _write_manifest(f"{DATA}/manifests/{name}.csv", rows)
        print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-url", default="")
    args = ap.parse_args()
    waveui()
    if args.synth_url:
        synth(args.synth_url)
    manifests()
    print("ALL DONE")
