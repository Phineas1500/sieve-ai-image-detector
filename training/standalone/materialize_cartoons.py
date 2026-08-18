#!/usr/bin/env python3
"""Cartoon/game-art hard-real data for ft4 (runs on the training node).

ft3 field reports showed a category hole: flat-color cartoon frames and game
art score as AI (Club Penguin 0.9997, South Park 0.9997/0.92). ft4 adds
scene-level hard reals in that style, plus AI-generated pixel art on the
positive side so the boundary stays generator-based within the style:

  - badigadiii/game_screenshots_11k         real game art/screens  label 0
  - CasperLD/tom_and_jerry_..._512          real cel animation     label 0
  - cgarciae/cartoonset (CC-BY-4.0)         flat vector avatars    label 0
  - jainr3/diffusiondb-pixelart (CC0)       AI pixel art           label 1

Heldout split is a contiguous tail per source (adjacent animation frames are
near-duplicates; a random split would leak). Strides subsample across each
source's range instead of taking only its head.

Manifests: cartoons_train.csv / cartoons_heldout.csv.
Env: DATA, SCRATCH, HF_TOKEN as usual.
"""
import os
import re
import shutil

from materialize_eval import DATA, SCRATCH, _sniff_ext, _write_manifest

SOURCES = [
    {"prefix": "game", "repo": "badigadiii/game_screenshots_11k", "config": "default",
     "stride": 1, "train": 8000, "heldout": 800},
    {"prefix": "tj", "repo": "CasperLD/tom_and_jerry_cartoons_with_blip_captions_512",
     "config": "default", "stride": 2, "train": 6000, "heldout": 600},
    {"prefix": "cset", "repo": "cgarciae/cartoonset", "config": "100k",
     "stride": 8, "train": 3000, "heldout": 300},
    {"prefix": "pxai", "repo": "jainr3/diffusiondb-pixelart", "config": "2k_all",
     "stride": 1, "train": 1800, "heldout": 200},
    # ft4.1: modern flat-TV-cartoon style (South Park / Club Penguin reports
    # survived ft4 — cel film, procedural avatars, and game thumbnails don't
    # span crisp-outline flat digital 2D; these frames do)
    {"prefix": "sp", "repo": "CasperLD/south_park_cartoons_with_blip_captions_512",
     "config": "default", "stride": 2, "train": 5000, "heldout": 500},
    {"prefix": "fg", "repo": "CasperLD/family_guy_cartoons_with_blip_captions_512",
     "config": "default", "stride": 4, "train": 4000, "heldout": 400},
    {"prefix": "at", "repo": "CasperLD/adventure_time_cartoons_with_blip_captions_512",
     "config": "default", "stride": 4, "train": 3000, "heldout": 300},
    {"prefix": "av", "repo": "lumenggan/avatar-the-last-airbender",
     "config": "default", "stride": 6, "train": 2000, "heldout": 200},
    {"prefix": "spc", "repo": "leffff/south-park-character-png-dataset",
     "config": "default", "stride": 1, "train": 860, "heldout": 90},
    # ft4.2: (a) AI cartoon/illustration positives to recover the illustration
    # recall ft4.1 gave back, and to teach the AI side of mascot/sticker art;
    # (b) HD stills (fine-texture gap), gradient emoji/mascot vector reals,
    # and CGI object renders (issue #15 category)
    {"prefix": "aicart", "repo": "poloclub/diffusiondb", "config": "2m_random_50k",
     "stride": 1, "train": 6000, "heldout": 600, "text_col": "prompt",
     "text_re": r"cartoon|illustrat|sticker|mascot|chibi|pixar|disney|anime|comic|vector art|flat design|emoji|cel[- ]?shad|clip ?art|character design"},
    {"prefix": "wallp", "repo": "puruchinera/anime_wallpapers", "config": "default",
     "stride": 3, "train": 4000, "heldout": 400},
    {"prefix": "emoji", "repo": "valhalla/emoji-dataset", "config": "default",
     "stride": 1, "train": 2400, "heldout": 250},
    {"prefix": "render", "repo": "tyhuang/ShapeNet_Rendering", "config": "default",
     "stride": 12, "train": 4000, "heldout": 400},
]

# prefix -> (label, source) for manifest rows
TAXONOMY = {
    "game": (0, "game_art"),
    "tj": (0, "cartoon_frame"),
    "cset": (0, "cartoon_vector"),
    "pxai": (1, "ai_pixelart"),
    "sp": (0, "cartoon_southpark"),
    "fg": (0, "cartoon_familyguy"),
    "at": (0, "cartoon_adventuretime"),
    "av": (0, "cartoon_avatar"),
    "spc": (0, "cartoon_charart"),
    "aicart": (1, "ai_cartoon"),
    "wallp": (0, "anime_wallpaper"),
    "emoji": (0, "emoji_vector"),
    "render": (0, "cgi_render"),
}


def ingest(spec):
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    marker = f"{DATA}/cartoons/.{spec['prefix']}_done"
    if os.path.exists(marker):
        print(f"{spec['prefix']}: done marker present, skipping")
        return
    token = os.environ.get("HF_TOKEN") or None
    # auto-converted parquet branch: uniform layout <config>/<split>/NNNN.parquet
    files = sorted(
        f for f in HfApi(token=token).list_repo_files(
            spec["repo"], repo_type="dataset", revision="refs/convert/parquet")
        if f.startswith(spec["config"] + "/") and "train" in f and f.endswith(".parquet")
    )
    if not files:
        raise RuntimeError(f"{spec['prefix']}: no parquet shards on convert branch")
    os.makedirs(f"{DATA}/cartoons/train", exist_ok=True)
    os.makedirs(f"{DATA}/cartoons/heldout", exist_ok=True)
    total_cap = spec["train"] + spec["heldout"]
    cache = f"{SCRATCH}/cart_{spec['prefix']}"
    written, seen = [], 0
    for fpath in files:
        if len(written) >= total_cap:
            break
        local = hf_hub_download(spec["repo"], fpath, repo_type="dataset",
                                revision="refs/convert/parquet", token=token, local_dir=cache)
        pf = pq.ParquetFile(local)
        names = pf.schema_arrow.names
        col_name = next(c for c in ("image", "img_bytes", "png") if c in names)
        cols = [col_name] + ([spec["text_col"]] if spec.get("text_col") else [])
        text_re = re.compile(spec["text_re"], re.I) if spec.get("text_re") else None
        for batch in pf.iter_batches(batch_size=32, columns=cols):
            col = batch.column(0)
            txt = batch.column(1) if text_re else None
            for i in range(len(col)):
                if len(written) >= total_cap:
                    break
                if text_re and not text_re.search(txt[i].as_py() or ""):
                    continue
                seen += 1
                if (seen - 1) % spec["stride"]:
                    continue
                b = col[i].as_py() if col_name == "img_bytes" else col[i]["bytes"].as_py()
                if b is None:
                    continue
                p = f"{DATA}/cartoons/train/{spec['prefix']}_{len(written):06d}{_sniff_ext(b)}"
                with open(p, "wb") as f:
                    f.write(b)
                written.append(p)
        os.remove(local)
        print(f"  {spec['prefix']}: {fpath} done ({len(written)}/{total_cap})")
    shutil.rmtree(cache, ignore_errors=True)
    for p in written[-spec["heldout"]:]:
        shutil.move(p, f"{DATA}/cartoons/heldout/{os.path.basename(p)}")
    open(marker, "w").close()
    print(f"{spec['prefix']}: {len(written) - spec['heldout']} train / "
          f"{min(spec['heldout'], len(written))} heldout")


def manifests():
    for split, name in (("train", "cartoons_train"), ("heldout", "cartoons_heldout")):
        d = f"{DATA}/cartoons/{split}"
        rows = []
        for f in sorted(os.listdir(d)):
            if f.startswith("."):
                continue
            label, source = TAXONOMY[f.split("_")[0]]
            rows.append([os.path.join(d, f), label, source, name])
        _write_manifest(f"{DATA}/manifests/{name}.csv", rows)
        n_fake = sum(1 for r in rows if r[1] == 1)
        print(f"{name}: {len(rows)} rows ({n_fake} fake / {len(rows) - n_fake} real)")


if __name__ == "__main__":
    for spec in SOURCES:
        ingest(spec)
    manifests()
    print("ALL DONE")
