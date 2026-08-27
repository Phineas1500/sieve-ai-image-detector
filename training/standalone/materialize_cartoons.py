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
    # ft4.3: properly-sized modern-generator positives in the cartoon style
    # space (diffusiondb's partial config only yielded ~1.2k). MJ v6 prompt-
    # filtered, plus Niji (MJ's anime/cartoon model) where every image is
    # flat-cartoon AI by construction.
    {"prefix": "mjc", "repo": "brivangl/midjourney-v6-llava", "config": "default",
     "stride": 1, "train": 5000, "heldout": 500, "text_col": "prompt",
     "text_re": r"cartoon|illustrat|sticker|mascot|chibi|pixar|disney|anime|comic|vector art|flat design|emoji|cel[- ]?shad|clip ?art|character design|kawaii|doodle"},
    {"prefix": "niji", "repo": "Korakoe/NijiJourney-Prompt-Pairs", "config": "default",
     "stride": 1, "train": 3400, "heldout": 350},
    {"prefix": "niji2", "repo": "p1atdev/nijijourney", "config": "default",
     "stride": 1, "train": 1650, "heldout": 180},
    # ft4.4: processed-real imagery — the dominant field FP category (issues
    # #17-#22): designed YouTube thumbnail composites (2023 scrape, pre-AI-
    # adoption in thumbnail design), retouched celebrity portraits (CelebA-HQ,
    # pre-2015), and painterly AI positives to anchor the painterly boundary.
    {"prefix": "ytt", "repo": "vargr/yt_thumbnail_dataset", "config": "default",
     "stride": 3, "train": 8000, "heldout": 800},
    {"prefix": "celeb", "repo": "Chris1/celebA-HQ", "config": "default",
     "stride": 4, "train": 4000, "heldout": 400},
    {"prefix": "aipnt", "repo": "poloclub/diffusiondb", "config": "2m_random_50k",
     "stride": 1, "train": 5000, "heldout": 500, "text_col": "prompt",
     "text_re": r"painting|watercolor|oil on canvas|impressionis|acrylic|gouache|artstation|concept art|fantasy art|digital painting|matte painting"},
    # ft5 (category sweep findings): the Recraft structural gap gets a
    # dedicated positive set (Rapidata preference pairs: every row carries a
    # Recraft-side image); designed promo composites, pop-art/halftone print,
    # and degraded reals close the top FP categories; selfie2anime provides
    # both low-quality real selfies (imageA) and GAN anime outputs (imageB).
    {"prefix": "rcf3", "repo": "Rapidata/Recraft-v3-24-7-25_t2i_human_preference",
     "config": "default", "stride": 7, "train": 8000, "heldout": 800,
     "pair": [["image1", "model1"], ["image2", "model2"]], "pair_match": "recraft"},
    {"prefix": "rcf2", "repo": "Rapidata/Recraft-V2_t2i_human_preference",
     "config": "default", "stride": 4, "train": 2500, "heldout": 250,
     "pair": [["image1", "model1"], ["image2", "model2"]], "pair_match": "recraft"},
    {"prefix": "popart", "repo": "luethan2025/WikiArt-Pop-Art", "config": "default",
     "stride": 1, "train": 1350, "heldout": 130},
    {"prefix": "poster", "repo": "skvarre/movie_posters-100k", "config": "default",
     "stride": 4, "train": 6000, "heldout": 600, "text_col": "release_date",
     "text_re": r"^(19|200|201)"},
    {"prefix": "indoor", "repo": "FatimaSaadNaik/indoor-scenes-dataset", "config": "default",
     "stride": 3, "train": 4000, "heldout": 400},
    {"prefix": "selfa", "repo": "huggan/selfie2anime", "config": "default",
     "stride": 1, "train": 3000, "heldout": 300, "img_col": "imageA"},
    {"prefix": "gana", "repo": "huggan/selfie2anime", "config": "default",
     "stride": 1, "train": 3000, "heldout": 300, "img_col": "imageB"},
    {"prefix": "prodph", "repo": "rajuptvs/ecommerce_products_clip", "config": "default",
     "stride": 1, "train": 2200, "heldout": 220},
    # --- ft6: photoreal / edit-class positives + hard reals for the FP clusters
    # (studio product shots, press portraits, low-light phone photos, game
    # captures). See materialize_edits.py / materialize_commons.py for the two
    # non-parquet sources (gptedit/editsrc, press); materialize_zip.py for the
    # zip-packaged bitmind sets (nb150k, ideo).
    {"prefix": "fswap", "repo": "bitmind/face-swap", "config": "default",
     "stride": 1, "train": 7000, "heldout": 700, "img_col": "image"},
    {"prefix": "abo", "repo": "suvadityamuk/amazon-berkeley-objects", "config": "images_original",
     "stride": 2, "train": 10000, "heldout": 1000},
    {"prefix": "steam", "repo": "taesiri/SteamScreenshots_Compressed", "config": "default",
     "stride": 1, "train": 9000, "heldout": 900, "img_col": "jpg"},
    {"prefix": "lowl", "repo": "ARM4588/Lowlight-Smartphone-Dataset", "config": "default",
     "stride": 1, "train": 5000, "heldout": 500, "img_col": "png"},
    {"prefix": "lowl2", "repo": "ishicode/low-light-dataset", "config": "default",
     "stride": 1, "train": 520, "heldout": 60},
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
    "mjc": (1, "ai_cartoon_mj6"),
    "niji": (1, "ai_niji"),
    "niji2": (1, "ai_niji"),
    "ytt": (0, "yt_thumbnail"),
    "celeb": (0, "celeb_portrait"),
    "aipnt": (1, "ai_painterly"),
    "rcf3": (1, "ai_recraft3"),
    "rcf2": (1, "ai_recraft2"),
    "popart": (0, "popart_print"),
    "poster": (0, "promo_poster"),
    "indoor": (0, "degraded_indoor"),
    "selfa": (0, "selfie_lowq"),
    "gana": (1, "ai_gan_anime"),
    "prodph": (0, "product_photo"),
    "nb150k": (1, "ai_nanobanana"),
    "fswap": (1, "ai_faceswap"),
    "ideo": (1, "ai_ideogram"),
    "abo": (0, "product_catalog"),
    "steam": (0, "game_screenshot"),
    "lowl": (0, "lowlight_phone"),
    "lowl2": (0, "lowlight_phone"),
    "gptedit": (1, "ai_gpt_edit"),
    "editsrc": (0, "edit_source_photo"),
    "press": (0, "press_photo"),
    "nbpro": (1, "ai_nanobanana_pro_wild"),
    "gpt2": (1, "ai_gptimage2_wild"),
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
        pair = spec.get("pair")
        if pair:
            cols = [c for p_ in pair for c in p_]
        else:
            col_name = spec.get("img_col") or next(c for c in ("image", "img_bytes", "png", "jpg") if c in names)
            cols = [col_name] + ([spec["text_col"]] if spec.get("text_col") else [])
        text_re = re.compile(spec["text_re"], re.I) if spec.get("text_re") else None
        for batch in pf.iter_batches(batch_size=32, columns=cols):
            data = {c: batch.column(j) for j, c in enumerate(cols)}
            nrows = len(batch.column(0))
            for i in range(nrows):
                if len(written) >= total_cap:
                    break
                if pair:
                    # preference-pair row: take the side whose model matches
                    b = None
                    for icol, mcol in pair:
                        m = data[mcol][i].as_py() or ""
                        if spec["pair_match"] in str(m).lower():
                            v = data[icol][i]
                            b = v.as_py() if not hasattr(v, "__getitem__") else v["bytes"].as_py()
                            break
                    if b is None:
                        continue
                    seen += 1
                    if (seen - 1) % spec["stride"]:
                        continue
                else:
                    if text_re and not text_re.search(data[spec["text_col"]][i].as_py() or ""):
                        continue
                    seen += 1
                    if (seen - 1) % spec["stride"]:
                        continue
                    raw = data[col_name][i].as_py()
                    # image struct {bytes,path} / raw bytes / list-of-ints
                    # (bitmind/face-swap stores JPEG bytes as an int list)
                    b = (raw.get("bytes") if isinstance(raw, dict)
                         else bytes(raw) if isinstance(raw, (list, tuple)) else raw)
                if b is None:
                    continue
                p = f"{DATA}/cartoons/train/{spec['prefix']}_{len(written):06d}{_sniff_ext(b)}"
                with open(p, "wb") as f:
                    f.write(b)
                written.append(p)
        os.remove(local)
        print(f"  {spec['prefix']}: {fpath} done ({len(written)}/{total_cap})")
    shutil.rmtree(cache, ignore_errors=True)
    # a source that under-fills (exhausted or filter-limited) must not end up
    # mostly-heldout: cap the tail split at 20% of what was actually collected
    ho = min(spec["heldout"], len(written) // 5)
    if ho:
        for p in written[-ho:]:
            shutil.move(p, f"{DATA}/cartoons/heldout/{os.path.basename(p)}")
    open(marker, "w").close()
    print(f"{spec['prefix']}: {len(written) - ho} train / {ho} heldout")


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
    import traceback
    failed = []
    for spec in SOURCES:
        try:
            ingest(spec)
        except Exception:  # one bad source must not block the rest (markers make reruns cheap)
            traceback.print_exc()
            failed.append(spec["prefix"])
    manifests()
    print("FAILED SOURCES:", ", ".join(failed) if failed else "none")
    print("ALL DONE")
