#!/usr/bin/env python3
"""Standalone fine-tune (ft2) — port of the Modal finetune() with one addition:
a thumbnail-regime augmentation path (hard downscale to 180-320px shorter side)
targeting the measured MJ7-at-thumbnail-scale weakness.

Env: DATA (default ~/data). Requires torch/timm/pillow in the venv.
Usage:
  python3 -u train.py --run-name ft2 --max-steps 4500 --val-every 750 \
      --train-manifests openfake_train,coco_train,wikiart --val-manifest openfake_validation
"""
import argparse
import re
import csv
import io
import json
import os
import random as pyrandom
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

DATA = os.path.expanduser(os.environ.get("DATA", "~/data"))


def _degrade(pil_img, view):
    from PIL import Image

    if view == "clean":
        return pil_img
    target, q = (768, 60) if view == "web" else (512, 40)
    w, h = pil_img.size
    m = max(w, h)
    if m > target:
        s = target / m
        pil_img = pil_img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _train_augment(pil_img, rng, input_size, resize_size, thumb_p=0.25):
    from PIL import Image

    interps = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
    # thumbnail regime (new in ft2): hard downscale like search-result thumbs
    if rng.random() < thumb_p:
        w, h = pil_img.size
        t = rng.randint(180, 320)
        if min(w, h) > t:
            s = t / min(w, h)
            pil_img = pil_img.resize((max(64, round(w * s)), max(64, round(h * s))), rng.choice(interps))
    elif rng.random() < 0.7:
        s = rng.uniform(0.4, 1.3)
        w, h = pil_img.size
        pil_img = pil_img.resize((max(64, round(w * s)), max(64, round(h * s))), rng.choice(interps))
    for k in range(2):
        if rng.random() < (0.85 if k == 0 else 0.3):
            buf = io.BytesIO()
            # modern CDNs (YouTube, Twitter) re-encode as WebP/AVIF; WebP's
            # VP8 artifacts stand in for that family (AVIF encode is too slow
            # for the loader). JPEG stays the dominant web codec.
            fmt = "WEBP" if rng.random() < 0.25 else "JPEG"
            pil_img.convert("RGB").save(buf, format=fmt, quality=rng.randint(30, 95))
            buf.seek(0)
            pil_img = Image.open(buf).convert("RGB")
    w, h = pil_img.size
    scale = resize_size / min(w, h)
    pil_img = pil_img.resize((max(input_size, round(w * scale)), max(input_size, round(h * scale))), Image.BILINEAR)
    w, h = pil_img.size
    x0 = rng.randint(0, w - input_size) if w > input_size else 0
    y0 = rng.randint(0, h - input_size) if h > input_size else 0
    pil_img = pil_img.crop((x0, y0, x0 + input_size, y0 + input_size))
    if rng.random() < 0.5:
        pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    return pil_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="ft2")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--input-size", type=int, default=384)
    ap.add_argument("--hf-repo", default="OwensLab/commfor-model-384")
    ap.add_argument("--train-manifests", default="openfake_train,coco_train,wikiart")
    ap.add_argument("--val-manifest", default="openfake_validation")
    ap.add_argument("--max-steps", type=int, default=4500)
    ap.add_argument("--val-every", type=int, default=750)
    ap.add_argument("--val-cap", type=int, default=6000)
    ap.add_argument("--thumb-p", type=float, default=0.25)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--backbone", default="cf", help="cf | dinov2_b14 | dinov2_b14_336 | clip_b16 | dinov2_s14")
    ap.add_argument("--freeze-blocks", type=int, default=0,
                    help="freeze embeddings + the first N transformer blocks (timm backbones)")
    ap.add_argument("--exclude-sources", default="",
                    help="regex; training items whose source matches are dropped (ablations)")
    ap.add_argument("--degrade-view", default="",
                    help="apply the benchmark's web/hard delivery path (resize + JPEG q60/q40) at "
                         "train time to matching sources with the given probability: comma list of "
                         "<source-regex>=<p>, e.g. '^(ai_biggan|ai_glide)=0.5'. ft10: the audit found "
                         "the 2021-22 cohort and faces learned at clean did not transfer to degraded "
                         "delivery; the standard augmentation alone was not enough for small slices.")
    ap.add_argument("--init-ckpt", default="",
                    help="initialise from a training checkpoint (model_state_dict) instead of the HF base")
    ap.add_argument("--boost", default="",
                    help="oversample by source: comma list of <source-regex>=<factor>, e.g. "
                         "'^(ai_nanobanana|ai_gpt_edit|press_photo)$=3'. Applied before fake/real "
                         "balancing, so the class mix stays 50/50.")
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from torchvision import transforms

    from backbones import build, freeze_blocks, sizes

    Image.MAX_IMAGE_PIXELS = None
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.backbone != "cf":
        args.input_size, _ = sizes(args.backbone)
    C = args.input_size
    resize_size = 440 if C == 384 else 256
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def load_items(names):
        items = []
        for n in names.split(","):
            with open(f"{DATA}/manifests/{n}.csv") as f:
                items += list(csv.DictReader(f))
        return items

    train_items = load_items(args.train_manifests)
    if args.exclude_sources:
        ex = re.compile(args.exclude_sources)
        before = len(train_items)
        train_items = [it for it in train_items if not ex.search(it.get("source", ""))]
        print(f"exclude-sources {args.exclude_sources}: dropped {before - len(train_items)} items")
    val_items = load_items(args.val_manifest)
    if args.val_cap and len(val_items) > args.val_cap:
        pyrandom.Random(0).shuffle(val_items)
        val_items = val_items[:args.val_cap]
    n_fake = sum(1 for it in train_items if int(it["label"]) == 1)
    n_real = len(train_items) - n_fake
    print(f"train: {len(train_items)} ({n_fake} fake / {n_real} real); val: {len(val_items)}")

    degrade_rules = [(re.compile(k), float(v)) for k, v in
                     (d.split("=") for d in args.degrade_view.split(",") if d)]

    def degrade_p(source):
        return max([p_ for pat, p_ in degrade_rules if pat.search(source)] or [0.0])

    if degrade_rules:
        n_dv = sum(1 for it in train_items if degrade_p(it.get("source", "")) > 0)
        print(f"degrade-view: {n_dv} items eligible ({args.degrade_view})")

    class TrainDS(Dataset):
        def __len__(self):
            return len(train_items)

        def __getitem__(self, i):
            it = train_items[i]
            rng = pyrandom.Random((hash(it["path"]) ^ pyrandom.getrandbits(32)) & 0xFFFFFFFF)
            try:
                img = Image.open(it["path"]).convert("RGB")
                dv = degrade_p(it.get("source", "")) if degrade_rules else 0.0
                if dv and rng.random() < dv:
                    img = _degrade(img, "hard" if rng.random() < 0.5 else "web")
                img = _train_augment(img, rng, C, resize_size, args.thumb_p)
                return norm(transforms.functional.to_tensor(img)), float(it["label"])
            except Exception:
                return torch.zeros(3, C, C), -1.0

    class ValDS(Dataset):
        def __len__(self):
            return len(val_items)

        def __getitem__(self, i):
            it = val_items[i]
            try:
                img = _degrade(Image.open(it["path"]).convert("RGB"), "web")
                w, h = img.size
                s = resize_size / min(w, h)
                img = img.resize((max(C, round(w * s)), max(C, round(h * s))), Image.BILINEAR)
                w, h = img.size
                img = img.crop(((w - C) // 2, (h - C) // 2, (w - C) // 2 + C, (h - C) // 2 + C))
                return norm(transforms.functional.to_tensor(img)), float(it["label"])
            except Exception:
                return torch.zeros(3, C, C), -1.0

    boosts = [(re.compile(k), float(v)) for k, v in
              (b.split("=") for b in args.boost.split(",") if b)]
    def boost(it):
        return max([f for pat, f in boosts if pat.search(it.get("source", ""))] or [1.0])
    item_boost = [boost(it) for it in train_items]
    mass_fake = sum(b for it, b in zip(train_items, item_boost) if int(it["label"]) == 1)
    mass_real = sum(b for it, b in zip(train_items, item_boost) if int(it["label"]) == 0)
    weights = [(0.5 / max(mass_fake, 1e-9) if int(it["label"]) == 1 else 0.5 / max(mass_real, 1e-9)) * b
               for it, b in zip(train_items, item_boost)]
    if boosts:
        boosted = sum(1 for b in item_boost if b > 1)
        print(f"boost: {boosted} items boosted ({args.boost})")
    sampler = WeightedRandomSampler(weights, num_samples=len(train_items), replacement=True)
    dl = DataLoader(TrainDS(), batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
                    pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4)
    vdl = DataLoader(ValDS(), batch_size=128, num_workers=args.num_workers, pin_memory=True)

    model = build(args.backbone, hf_repo=args.hf_repo).cuda()
    if args.init_ckpt:
        sd0 = torch.load(args.init_ckpt, map_location="cpu")
        model.load_state_dict(sd0.get("model_state_dict", sd0))
        print(f"init-ckpt: {args.init_ckpt} (step {sd0.get('step')}, val {sd0.get('val')})")
    if args.freeze_blocks:
        nfrozen = freeze_blocks(model, args.freeze_blocks)
        print(f"backbone {args.backbone}: froze {nfrozen} param tensors (first {args.freeze_blocks} blocks + embeddings)")
    ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad); ntot = sum(p.numel() for p in model.parameters())
    print(f"params: {ntrain/1e6:.1f}M trainable / {ntot/1e6:.1f}M total; input {C}px")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.05)
    total_steps = args.max_steps or (len(dl) * args.epochs)
    warmup = min(300, total_steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup
        else 0.5 * (1 + np.cos(np.pi * (s - warmup) / max(1, total_steps - warmup))))
    crit = nn.BCEWithLogitsLoss()
    ckdir = f"{DATA}/ckpts/{args.run_name}"
    os.makedirs(ckdir, exist_ok=True)

    def validate():
        model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for xb, yb in vdl:
                keep = yb >= 0
                if keep.sum() == 0:
                    continue
                with torch.autocast("cuda", torch.bfloat16):
                    z = model(xb[keep].cuda()).squeeze(-1)
                scores.append(torch.sigmoid(z.float()).cpu().numpy())
                labels.append(yb[keep].numpy())
        model.train()
        s, y = np.concatenate(scores), np.concatenate(labels)
        tpr = float((s[y == 1] >= 0.65).mean())
        tnr = float((s[y == 0] < 0.65).mean())
        best = max(((s[y == 1] >= t).mean() + (s[y == 0] < t).mean()) / 2
                   for t in np.linspace(0.02, 0.98, 97))
        return {"bal@0.65": round((tpr + tnr) / 2, 4), "tpr": round(tpr, 4),
                "tnr": round(tnr, 4), "best_bal": round(float(best), 4)}

    print("initial val:", validate())
    best_bal, step = 0.0, 0
    model.train()
    done = False
    for ep in range(args.epochs):
        if done:
            break
        for xb, yb in dl:
            keep = yb >= 0
            if keep.sum() < 2:
                continue
            with torch.autocast("cuda", torch.bfloat16):
                z = model(xb[keep].cuda()).squeeze(-1)
                loss = crit(z, yb[keep].cuda().to(z.dtype))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"ep{ep} step{step}/{total_steps} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e}")
            if step % args.val_every == 0 or step == total_steps:
                m = validate()
                print(f"  VAL step{step}: {json.dumps(m)}")
                torch.save({"model_state_dict": model.state_dict(), "step": step, "val": m, "backbone": args.backbone, "input_size": C}, f"{ckdir}/last.pt")
                if m["best_bal"] > best_bal:
                    best_bal = m["best_bal"]
                    torch.save({"model_state_dict": model.state_dict(), "step": step, "val": m, "backbone": args.backbone, "input_size": C}, f"{ckdir}/best.pt")
                    print(f"  new best ({best_bal})")
            if step >= total_steps:
                done = True
                break

    print("final val:", validate())
    print(json.dumps({"best_val_bal": best_bal, "ckpt": f"{ckdir}/best.pt"}))


if __name__ == "__main__":
    main()
