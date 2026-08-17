#!/usr/bin/env python3
"""Standalone training-data materializer for the ft2 retrain (no Modal).

Builds on the same node/volume as materialize_eval.py:
  - OpenFake core/train sampled subset, web-tier re-encoded, with boosted
    per-generator caps on the measured-weak painterly/frontier families
  - OpenFake core/validation subset (original bytes)
  - COCO train2017 real photos (subsampled)
  - WikiArt hard-real paintings

Env: DATA, SCRATCH, HF_TOKEN (as materialize_eval.py).
"""
import argparse
import io
import os
import random
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor

from materialize_eval import DATA, SCRATCH, _download, _sniff_ext, _write_manifest

# ft1 measured weak spots: painterly MJ7-style content and frontier generators.
# Boost their share of the training mix; everything else keeps the default cap.
CAP_BOOST = {
    "midjourney-7": 6000, "illustrious": 3000, "z-image-turbo": 3000,
    "seedream-v5.0": 3000, "nano-banana-pro": 3000, "gpt-image-2": 3000,
    "gpt-image-1.5": 3000, "recraft-v3": 2500, "flux.2-klein-9b": 3000,
    "sora-2": 2000, "veo-3": 2000,
}
DEFAULT_FAKE_CAP = 1200


def _fake_cap(model):
    return CAP_BOOST.get(model, DEFAULT_FAKE_CAP)


def _reencode_web_tier(b: bytes) -> bytes:
    from PIL import Image

    img = Image.open(io.BytesIO(b)).convert("RGB")
    w, h = img.size
    if min(w, h) > 768:
        s = 768 / min(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _openfake_split(split, subset, sample_files, n_groups, workers, real_cap,
                    fake_cap_fn, web_tier, fixed_fake_cap=None):
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    marker = f"{DATA}/{subset}/.done"
    if os.path.exists(marker):
        print(f"{subset}: done marker present, skipping")
        return
    token = os.environ.get("HF_TOKEN") or None
    all_files = sorted(
        f for f in HfApi(token=token).list_repo_files("ComplexDataLab/OpenFake", repo_type="dataset")
        if f.startswith(f"core/{split}-") and f.endswith(".parquet")
    )
    if sample_files and sample_files < len(all_files):
        step = len(all_files) / sample_files
        files = [all_files[int(i * step)] for i in range(sample_files)]
    else:
        files = all_files
    n_groups = min(n_groups, len(files))
    groups = [files[i::n_groups] for i in range(n_groups)]
    real_cap_group = max(10, -(-real_cap * 13 // (10 * n_groups)))
    outdir = f"{DATA}/{subset}"
    os.makedirs(f"{outdir}/fake", exist_ok=True)
    os.makedirs(f"{outdir}/real", exist_ok=True)
    print(f"{subset}: {len(files)}/{len(all_files)} shards in {n_groups} groups")

    def group_fake_cap(model):
        cap = fixed_fake_cap if fixed_fake_cap else fake_cap_fn(model)
        return max(2, -(-cap * 2 // n_groups))

    def worker(gid):
        rows, per_model, nreal = [], {}, 0
        cache = f"{SCRATCH}/hf_t{gid}"
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
                        if per_model.get(model, 0) >= group_fake_cap(model):
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
                    if web_tier:
                        try:
                            b = _reencode_web_tier(b)
                        except Exception:
                            continue
                        ext = ".jpg"
                    else:
                        ext = _sniff_ext(b)
                    p = f"{outdir}/{sub}/t{gid}_{model}_{idx:06d}{ext}"
                    with open(p, "wb") as f:
                        f.write(b)
                    rows.append([p, lab, f"of_{model}" if lab else f"of_real_{model}", subset])
            os.remove(local)
            print(f"  {subset} t{gid}: {fpath} done (kept {len(rows)})")
        shutil.rmtree(cache, ignore_errors=True)
        return rows

    all_rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for rows in ex.map(worker, range(n_groups)):
            all_rows += rows
    _write_manifest(f"{DATA}/manifests/{subset}.csv", all_rows)
    open(marker, "w").close()
    n_fake = sum(1 for r in all_rows if r[1] == 1)
    per = {}
    for r in all_rows:
        if r[1] == 1:
            per[r[2]] = per.get(r[2], 0) + 1
    print(f"{subset}: TOTAL {n_fake} fake / {len(all_rows) - n_fake} real")
    for k in sorted(per, key=per.get, reverse=True)[:15]:
        print(f"  {k}: {per[k]}")


def openfake_train():
    _openfake_split("train", "openfake_train", sample_files=192, n_groups=16,
                    workers=4, real_cap=70000, fake_cap_fn=_fake_cap, web_tier=True)


def openfake_validation():
    _openfake_split("validation", "openfake_validation", sample_files=8, n_groups=8,
                    workers=4, real_cap=3000, fake_cap_fn=None, web_tier=False,
                    fixed_fake_cap=200)


def coco_train(keep=60000):
    marker = f"{DATA}/real/coco_train2017/.done"
    if os.path.exists(marker):
        print("coco_train: done marker present, skipping")
        return
    zpath = f"{SCRATCH}/train2017.zip"
    _download("http://images.cocodataset.org/zips/train2017.zip", zpath, "coco train2017 (19GB)")
    os.makedirs(f"{DATA}/real/coco_train2017", exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(f"{DATA}/real/coco_train2017")
    os.remove(zpath)
    imgs = []
    for root, _, files in os.walk(f"{DATA}/real/coco_train2017"):
        imgs += [os.path.join(root, fn) for fn in files if fn.lower().endswith(".jpg")]
    random.Random(0).shuffle(imgs)
    for p in imgs[keep:]:
        os.remove(p)
    rows = [[p, 0, "coco_train2017", "coco_train"] for p in imgs[:keep]]
    _write_manifest(f"{DATA}/manifests/coco_train.csv", rows)
    open(marker, "w").close()
    print(f"coco_train: kept {len(rows)} of {len(imgs)}")


def wikiart(cap=15000):
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    marker = f"{DATA}/wikiart/.done"
    if os.path.exists(marker):
        print("wikiart: done marker present, skipping")
        return
    token = os.environ.get("HF_TOKEN") or None
    files = sorted(
        f for f in HfApi(token=token).list_repo_files("huggan/wikiart", repo_type="dataset")
        if f.endswith(".parquet")
    )
    os.makedirs(f"{DATA}/wikiart", exist_ok=True)
    rows, n = [], 0
    for fpath in files:
        if n >= cap:
            break
        local = hf_hub_download("huggan/wikiart", fpath, repo_type="dataset",
                                token=token, local_dir=f"{SCRATCH}/hf_wiki")
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
                p = f"{DATA}/wikiart/wikiart_{n:06d}{_sniff_ext(b)}"
                with open(p, "wb") as f:
                    f.write(b)
                rows.append([p, 0, "wikiart", "wikiart"])
        os.remove(local)
        print(f"  wikiart: {fpath} done ({n}/{cap})")
    shutil.rmtree(f"{SCRATCH}/hf_wiki", ignore_errors=True)
    _write_manifest(f"{DATA}/manifests/wikiart.csv", rows)
    open(marker, "w").close()
    print(f"wikiart: {len(rows)} real paintings")


STAGES = {
    "openfake_train": openfake_train,
    "openfake_validation": openfake_validation,
    "coco_train": coco_train,
    "wikiart": wikiart,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(STAGES))
    args = ap.parse_args()
    for name in args.only.split(","):
        STAGES[name.strip()]()
    os.system(f"df -h {DATA} | tail -1")
    print("ALL DONE")
