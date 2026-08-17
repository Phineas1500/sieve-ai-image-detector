#!/usr/bin/env python3
"""Standalone eval-benchmark materializer (no Modal required).

Ports the Modal pipeline's downloaders for plain GPU-provider nodes
(e.g. givemeanode): SynthBuster (Zenodo), COCO val2017, and capped
OpenFake core/test + reddit/test subsets — same directory layout and
manifest format as the Modal volume, so eval scripts run unchanged.

Env:
  DATA     output root (default ~/data) — put this on the persistent volume
  SCRATCH  large re-downloadable temp space (default /scratch, falls back to DATA/tmp)
  HF_TOKEN optional Hugging Face token

Usage:
  python3 -u materialize_eval.py [--only synthbuster,coco,openfake_test,openfake_reddit]
"""
import argparse
import csv
import os
import random
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor

import requests

DATA = os.path.expanduser(os.environ.get("DATA", "~/data"))
SCRATCH = os.environ.get("SCRATCH", "/scratch")
if not (os.path.isdir(SCRATCH) and os.access(SCRATCH, os.W_OK)):
    SCRATCH = os.path.join(DATA, "tmp")
os.makedirs(DATA, exist_ok=True)
os.makedirs(SCRATCH, exist_ok=True)


def _sniff_ext(b: bytes) -> str:
    if b[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return ".webp"
    return ".jpg"


def _write_manifest(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "source", "subset"])
        w.writerows(rows)


def _download(url, dest, desc):
    if os.path.exists(dest):
        print(f"{desc}: archive already present")
        return
    print(f"{desc}: downloading -> {dest}")
    tmp = dest + ".part"
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 22):
                f.write(chunk)
                done += len(chunk)
                if done % (1 << 30) < (1 << 22):
                    print(f"  {desc}: {done / 1e9:.1f}GB")
    os.replace(tmp, dest)


def synthbuster():
    marker = f"{DATA}/synthbuster/.done"
    if os.path.exists(marker):
        print("synthbuster: done marker present, skipping")
        return
    zpath = f"{SCRATCH}/synthbuster.zip"
    _download("https://zenodo.org/api/records/10066460/files/synthbuster.zip/content", zpath, "synthbuster (12.4GB)")
    print("synthbuster: extracting...")
    os.makedirs(f"{DATA}/synthbuster", exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(f"{DATA}/synthbuster")
    os.remove(zpath)
    rows = []
    for root, _, files in os.walk(f"{DATA}/synthbuster"):
        for fn in files:
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")):
                p = os.path.join(root, fn)
                rows.append([p, 1, f"sb_{os.path.basename(os.path.dirname(p))}", "synthbuster"])
    _write_manifest(f"{DATA}/manifests/synthbuster.csv", rows)
    open(marker, "w").close()
    print(f"synthbuster: {len(rows)} fake images")


def coco():
    marker = f"{DATA}/real/coco_val2017/.done"
    if os.path.exists(marker):
        print("coco: done marker present, skipping")
        return
    zpath = f"{SCRATCH}/val2017.zip"
    _download("http://images.cocodataset.org/zips/val2017.zip", zpath, "coco val2017 (778MB)")
    os.makedirs(f"{DATA}/real/coco_val2017", exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(f"{DATA}/real/coco_val2017")
    os.remove(zpath)
    imgs = []
    for root, _, files in os.walk(f"{DATA}/real/coco_val2017"):
        imgs += [os.path.join(root, fn) for fn in files if fn.lower().endswith(".jpg")]
    random.Random(0).shuffle(imgs)
    rows = [[p, 0, "coco_val2017", "coco"] for p in imgs[:3000]]
    _write_manifest(f"{DATA}/manifests/coco.csv", rows)
    open(marker, "w").close()
    print(f"coco: {len(rows)} real images (of {len(imgs)})")


def openfake(config, split, fake_cap_per_model, real_cap, workers=4):
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    subset = "openfake_reddit" if config == "reddit" else f"openfake_{split}"
    marker = f"{DATA}/{subset}/.done"
    if os.path.exists(marker):
        print(f"{subset}: done marker present, skipping")
        return
    token = os.environ.get("HF_TOKEN") or None
    files = sorted(
        f for f in HfApi(token=token).list_repo_files("ComplexDataLab/OpenFake", repo_type="dataset")
        if f.startswith(f"{config}/{split}-") and f.endswith(".parquet")
    )
    n_groups = min(workers * 3, len(files))
    groups = [files[i::n_groups] for i in range(n_groups)]
    fake_cap_pm_group = max(2, -(-fake_cap_per_model * 2 // n_groups))
    real_cap_group = max(10, -(-real_cap * 13 // (10 * n_groups)))
    outdir = f"{DATA}/{subset}"
    os.makedirs(f"{outdir}/fake", exist_ok=True)
    os.makedirs(f"{outdir}/real", exist_ok=True)
    print(f"{subset}: {len(files)} shards in {n_groups} groups, {workers} parallel")

    def worker(gid):
        rows, per_model, nreal = [], {}, 0
        cache = f"{SCRATCH}/hf_g{gid}"
        for fpath in groups[gid]:
            local = hf_hub_download("ComplexDataLab/OpenFake", fpath, repo_type="dataset",
                                    token=token, local_dir=cache)
            pf = pq.ParquetFile(local)
            for batch in pf.iter_batches(batch_size=32, columns=["image", "label", "model"]):
                imgs, labels, models = batch.column(0), batch.column(1), batch.column(2)
                for i in range(len(labels)):
                    label = labels[i].as_py()
                    model = (models[i].as_py() or "unknown").replace("/", "_")
                    if label == "fake":
                        if per_model.get(model, 0) >= fake_cap_pm_group:
                            continue
                        per_model[model] = per_model.get(model, 0) + 1
                        lab, sub, idx = 1, "fake", per_model[model]
                    else:
                        if nreal >= real_cap_group:
                            continue
                        nreal += 1
                        lab, sub, idx = 0, "real", nreal
                    b = imgs[i]["bytes"].as_py()
                    if b is None:
                        continue
                    p = f"{outdir}/{sub}/g{gid}_{model}_{idx:06d}{_sniff_ext(b)}"
                    with open(p, "wb") as f:
                        f.write(b)
                    rows.append([p, lab, f"of_{model}" if lab else f"of_real_{model}", subset])
            os.remove(local)
            print(f"  {subset} g{gid}: finished {fpath} (kept {len(rows)})")
        shutil.rmtree(cache, ignore_errors=True)
        return rows

    all_rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for rows in ex.map(worker, range(n_groups)):
            all_rows += rows
    _write_manifest(f"{DATA}/manifests/{subset}.csv", all_rows)
    open(marker, "w").close()
    n_fake = sum(1 for r in all_rows if r[1] == 1)
    print(f"{subset}: TOTAL {n_fake} fake / {len(all_rows) - n_fake} real")


STAGES = {
    "synthbuster": synthbuster,
    "coco": coco,
    "openfake_test": lambda: openfake("core", "test", 500, 4000),
    "openfake_reddit": lambda: openfake("reddit", "test", 2000, 2000),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(STAGES))
    args = ap.parse_args()
    for name in args.only.split(","):
        STAGES[name.strip()]()
    print("ALL DONE")
