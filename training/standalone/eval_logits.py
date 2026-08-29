#!/usr/bin/env python3
"""Standalone benchmark logit dump: scores a checkpoint over the node's eval
benchmark under clean/web/hard degradation, one CSV of raw logits per view.
Calibration fitting and comparisons happen offline from the CSVs.

Usage: DATA=~/data python3 -u eval_logits.py --ckpt ~/data/ckpts/ft2/best.pt --tag ft2
"""
import argparse
import csv
import io
import os
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--input-size", type=int, default=384)
    ap.add_argument("--views", default="clean,web,hard")
    ap.add_argument("--manifests", default="synthbuster,coco,openfake_test,openfake_reddit")
    ap.add_argument("--num-workers", type=int, default=12)
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    from backbones import build, sizes

    Image.MAX_IMAGE_PIXELS = None
    sd = torch.load(args.ckpt, map_location="cpu")
    backbone = sd.get("backbone", "cf") if isinstance(sd, dict) else "cf"
    C = args.input_size if backbone == "cf" else sizes(backbone)[0]
    resize_size = 440 if C == 384 else 256
    print(f"backbone {backbone}, input {C}px")
    post = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(C),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    items = []
    for n in args.manifests.split(","):
        with open(f"{DATA}/manifests/{n}.csv") as f:
            items += list(csv.DictReader(f))
    print(f"eval set: {len(items)}")

    model = build(backbone, pretrained=False)
    model.load_state_dict(sd.get("model_state_dict", sd))
    model.eval().cuda()

    class DS(Dataset):
        def __init__(self, view):
            self.view = view

        def __len__(self):
            return len(items)

        def __getitem__(self, i):
            try:
                img = Image.open(items[i]["path"]).convert("RGB")
                return post(_degrade(img, self.view)), i
            except Exception:
                return torch.zeros(3, C, C), -1

    os.makedirs(f"{DATA}/results", exist_ok=True)
    for view in args.views.split(","):
        dl = DataLoader(DS(view), batch_size=128, num_workers=args.num_workers,
                        pin_memory=True, prefetch_factor=4)
        z = np.full(len(items), np.nan)
        with torch.no_grad():
            for xb, idx in dl:
                out = model(xb.cuda()).squeeze(-1).float().cpu().numpy()
                for j, i in enumerate(idx.tolist()):
                    if i >= 0:
                        z[i] = out[j]
        outp = f"{DATA}/results/{args.tag}_{view}_logits.csv"
        with open(outp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "source", "z"])
            for it, zz in zip(items, z):
                w.writerow([it["path"], it["label"], it["source"], zz])
        print(f"{view}: saved {outp}")


if __name__ == "__main__":
    main()
