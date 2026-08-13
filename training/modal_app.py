"""Modal pipeline for the AI image detector.

Stages:
  1. Dataset download/materialization onto a persistent Volume
     - SynthBuster (Zenodo): 9 commercial/OSS generators, eval-only
     - COCO val2017: real photos, eval-only
     - OpenFake core/test + reddit/test subsets: modern generators, eval-only
  2. Zero-shot baseline eval of Community Forensics (balanced acc @ 0.65)

Run:
  modal run training/modal_app.py::download_all
  modal run training/modal_app.py::run_eval
"""

import io
import os
import csv
import json
import random

import modal

app = modal.App("ai-image-detector")

data_vol = modal.Volume.from_name("ai-detector-data", create_if_missing=True)
DATA = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("unzip")
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "timm==1.0.15",
        "datasets==3.6.0",
        "huggingface_hub[hf_transfer]",
        "hf_transfer",
        "pillow",
        "scikit-learn",
        "pandas",
        "requests",
        "onnx",
        "onnxruntime",
        "onnxconverter-common",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(os.path.join(os.path.dirname(__file__), "vendor"), remote_path="/root/vendor")
)

hf_secret = modal.Secret.from_name("huggingface-secret")


def _hf_token():
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]
    return None


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


@app.function(image=image, volumes={DATA: data_vol}, timeout=7200, cpu=4)
def download_synthbuster():
    import requests

    marker = f"{DATA}/synthbuster/.done"
    if os.path.exists(marker):
        print("synthbuster already downloaded")
        return
    os.makedirs(f"{DATA}/downloads", exist_ok=True)
    zpath = f"{DATA}/downloads/synthbuster.zip"
    url = "https://zenodo.org/api/records/10066460/files/synthbuster.zip/content"
    print("downloading synthbuster (12.4GB)...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(zpath, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 22):
                f.write(chunk)
    print("unzipping...")
    os.makedirs(f"{DATA}/synthbuster", exist_ok=True)
    os.system(f"unzip -q -o {zpath} -d {DATA}/synthbuster")
    os.remove(zpath)

    rows = []
    for root, _, files in os.walk(f"{DATA}/synthbuster"):
        for fn in files:
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")):
                p = os.path.join(root, fn)
                gen = os.path.basename(os.path.dirname(p))
                rows.append([p, 1, f"sb_{gen}", "synthbuster"])
    _write_manifest(f"{DATA}/manifests/synthbuster.csv", rows)
    open(marker, "w").close()
    print(f"synthbuster: {len(rows)} fake images")
    data_vol.commit()


@app.function(image=image, volumes={DATA: data_vol}, timeout=3600, cpu=4)
def download_coco():
    import requests

    marker = f"{DATA}/real/coco_val2017/.done"
    if os.path.exists(marker):
        print("coco already downloaded")
        return
    os.makedirs(f"{DATA}/downloads", exist_ok=True)
    zpath = f"{DATA}/downloads/val2017.zip"
    print("downloading COCO val2017 (778MB)...")
    with requests.get("http://images.cocodataset.org/zips/val2017.zip", stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(zpath, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 22):
                f.write(chunk)
    os.makedirs(f"{DATA}/real/coco_val2017", exist_ok=True)
    os.system(f"unzip -q -o {zpath} -d {DATA}/real/coco_val2017")
    os.remove(zpath)

    all_imgs = []
    for root, _, files in os.walk(f"{DATA}/real/coco_val2017"):
        for fn in files:
            if fn.lower().endswith(".jpg"):
                all_imgs.append(os.path.join(root, fn))
    random.Random(0).shuffle(all_imgs)
    rows = [[p, 0, "coco_val2017", "coco"] for p in all_imgs[:3000]]
    _write_manifest(f"{DATA}/manifests/coco.csv", rows)
    open(marker, "w").close()
    print(f"coco: {len(rows)} real images (of {len(all_imgs)} total)")
    data_vol.commit()


@app.function(image=image, volumes={DATA: data_vol}, timeout=14400, cpu=8, memory=16384, secrets=[hf_secret])
def materialize_openfake(config: str, fake_cap_per_model: int = 500, real_cap: int = 4000):
    """Stream an OpenFake split and save a capped subset as original-encoded files."""
    import datasets as hfds

    subset = f"openfake_{config}"
    marker = f"{DATA}/{subset}/.done"
    if os.path.exists(marker):
        print(f"{subset} already materialized")
        return

    ds = hfds.load_dataset(
        "ComplexDataLab/OpenFake", config, split="test", streaming=True, token=_hf_token()
    ).cast_column("image", hfds.Image(decode=False))

    per_model, real_count, rows = {}, 0, []
    outdir = f"{DATA}/{subset}"
    os.makedirs(outdir, exist_ok=True)
    n_seen = 0
    for ex in ds:
        n_seen += 1
        if n_seen % 20000 == 0:
            print(f"  scanned {n_seen} rows, kept {len(rows)}")
        label = ex["label"]
        model = (ex["model"] or "unknown").replace("/", "_")
        if label == "fake":
            k = per_model.get(model, 0)
            if k >= fake_cap_per_model:
                continue
            per_model[model] = k + 1
            lab = 1
            src = f"of_{model}"
        else:
            if real_count >= real_cap:
                continue
            real_count += 1
            lab = 0
            src = f"of_real_{model}"
        b = ex["image"]["bytes"]
        if b is None:
            continue
        fn = f"{model}_{per_model.get(model, real_count):06d}{_sniff_ext(b)}"
        sub = "fake" if lab else "real"
        os.makedirs(f"{outdir}/{sub}", exist_ok=True)
        p = f"{outdir}/{sub}/{fn}"
        with open(p, "wb") as f:
            f.write(b)
        rows.append([p, lab, src, subset])

    _write_manifest(f"{DATA}/manifests/{subset}.csv", rows)
    open(marker, "w").close()
    n_fake = sum(1 for r in rows if r[1] == 1)
    print(f"{subset}: kept {n_fake} fake / {len(rows) - n_fake} real (scanned {n_seen})")
    print("per-model:", json.dumps(per_model, indent=0))
    data_vol.commit()


# ---------------------------------------------------------------- evaluation

DEGRADATIONS = ["clean", "web", "hard"]


def _degrade(pil_img, view: str):
    """Simulate web-realistic re-encoding. Applied before the test transform."""
    from PIL import Image

    if view == "clean":
        return pil_img
    if view == "web":
        target, q = 768, 60
    elif view == "hard":
        target, q = 512, 40
    else:
        raise ValueError(view)
    w, h = pil_img.size
    m = max(w, h)
    if m > target:
        s = target / m
        pil_img = pil_img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


@app.function(image=image, volumes={DATA: data_vol}, gpu="L4", timeout=7200, cpu=8, memory=16384, secrets=[hf_secret])
def eval_zeroshot(hf_repo: str = "OwensLab/commfor-model-384", input_size: int = 384, views: str = "clean,web,hard"):
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    from vendor.cf_models import ViTClassifier

    Image.MAX_IMAGE_PIXELS = None

    manifests = [f"{DATA}/manifests/{n}.csv" for n in
                 ("synthbuster", "coco", "openfake_core", "openfake_reddit")]
    items = []
    for m in manifests:
        if not os.path.exists(m):
            print(f"WARNING: missing manifest {m}")
            continue
        with open(m) as f:
            items += [r for r in csv.DictReader(f)]
    print(f"eval set: {len(items)} images")

    resize_size = 440 if input_size == 384 else 256
    post = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class DS(Dataset):
        def __init__(self, items, view):
            self.items, self.view = items, view

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            it = self.items[i]
            try:
                img = Image.open(it["path"]).convert("RGB")
                img = _degrade(img, self.view)
                return post(img), i
            except Exception:
                return torch.zeros(3, input_size, input_size), -1

    model = ViTClassifier.from_pretrained(hf_repo)
    model.eval().cuda()

    os.makedirs(f"{DATA}/results", exist_ok=True)
    tag = hf_repo.split("/")[-1]
    summary = {}
    for view in views.split(","):
        dl = DataLoader(DS(items, view), batch_size=128, num_workers=8, pin_memory=True)
        scores = np.full(len(items), np.nan, dtype=np.float64)
        with torch.no_grad():
            for xb, idx in dl:
                logits = model(xb.cuda()).squeeze(-1)
                probs = torch.sigmoid(logits).float().cpu().numpy()
                for j, i in enumerate(idx.tolist()):
                    if i >= 0:
                        scores[i] = probs[j]

        df = pd.DataFrame(items)
        df["score"] = scores
        df["label"] = df["label"].astype(int)
        df = df.dropna(subset=["score"])
        df.to_csv(f"{DATA}/results/zeroshot_{tag}_{view}.csv", index=False)

        real, fake = df[df.label == 0], df[df.label == 1]
        thr = 0.65
        tnr = float((real.score < thr).mean())
        tpr = float((fake.score >= thr).mean())
        bal = (tpr + tnr) / 2
        auc = roc_auc_score(df.label, df.score)
        # best achievable balanced acc over thresholds (calibration headroom)
        ths = np.linspace(0.01, 0.99, 197)
        bals = [((fake.score.values >= t).mean() + (real.score.values < t).mean()) / 2 for t in ths]
        best_i = int(np.argmax(bals))
        summary[view] = dict(
            balanced_acc_at_065=round(bal, 4), tpr_fake=round(tpr, 4), tnr_real=round(tnr, 4),
            auc=round(float(auc), 4), best_balanced_acc=round(float(bals[best_i]), 4),
            best_threshold=round(float(ths[best_i]), 3), n=len(df),
        )
        print(f"\n=== view={view}: {json.dumps(summary[view])}")
        per_gen = df[df.label == 1].groupby("source").score.apply(lambda s: (s >= thr).mean())
        print("fake TPR@0.65 per generator:")
        print(per_gen.sort_values().to_string())
        per_real = df[df.label == 0].groupby("source").score.apply(lambda s: (s < thr).mean())
        print("real TNR@0.65 per source:")
        print(per_real.sort_values().to_string())

    with open(f"{DATA}/results/zeroshot_{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    data_vol.commit()
    return summary


@app.function(image=image, volumes={DATA: data_vol}, timeout=3600, cpu=4)
def download_coco_train():
    import requests

    marker = f"{DATA}/real/coco_train2017/.done"
    if os.path.exists(marker):
        print("coco train already downloaded")
        return
    os.makedirs(f"{DATA}/downloads", exist_ok=True)
    zpath = f"{DATA}/downloads/train2017.zip"
    print("downloading COCO train2017 (19GB)...")
    with requests.get("http://images.cocodataset.org/zips/train2017.zip", stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(zpath, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 22):
                f.write(chunk)
    os.makedirs(f"{DATA}/real/coco_train2017", exist_ok=True)
    os.system(f"unzip -q -o {zpath} -d {DATA}/real/coco_train2017")
    os.remove(zpath)
    rows = []
    for root, _, files in os.walk(f"{DATA}/real/coco_train2017"):
        for fn in files:
            if fn.lower().endswith(".jpg"):
                rows.append([os.path.join(root, fn), 0, "coco_train2017", "coco_train"])
    _write_manifest(f"{DATA}/manifests/coco_train.csv", rows)
    open(marker, "w").close()
    print(f"coco train: {len(rows)} real images")
    data_vol.commit()


@app.function(image=image, volumes={DATA: data_vol}, timeout=43200, cpu=8, memory=32768, secrets=[hf_secret])
def materialize_openfake_split(split: str, fake_cap_per_model: int = 2500, real_cap: int = 150000):
    """Materialize a capped subset of OpenFake core train/validation for training."""
    import datasets as hfds

    subset = f"openfake_{split}"
    marker = f"{DATA}/{subset}/.done"
    if os.path.exists(marker):
        print(f"{subset} already materialized")
        return
    ds = hfds.load_dataset(
        "ComplexDataLab/OpenFake", "core", split=split, streaming=True, token=_hf_token()
    ).cast_column("image", hfds.Image(decode=False))

    per_model, real_count, rows, n_seen = {}, 0, [], 0
    outdir = f"{DATA}/{subset}"
    os.makedirs(f"{outdir}/fake", exist_ok=True)
    os.makedirs(f"{outdir}/real", exist_ok=True)
    for ex in ds:
        n_seen += 1
        if n_seen % 50000 == 0:
            print(f"  scanned {n_seen}, kept {len(rows)} ({real_count} real)")
        label = ex["label"]
        model = (ex["model"] or "unknown").replace("/", "_")
        if label == "fake":
            k = per_model.get(model, 0)
            if k >= fake_cap_per_model:
                continue
            per_model[model] = k + 1
            lab, sub, idx = 1, "fake", per_model[model]
        else:
            if real_count >= real_cap:
                continue
            real_count += 1
            lab, sub, idx = 0, "real", real_count
        b = ex["image"]["bytes"]
        if b is None:
            continue
        p = f"{outdir}/{sub}/{model}_{idx:07d}{_sniff_ext(b)}"
        with open(p, "wb") as f:
            f.write(b)
        rows.append([p, lab, f"of_{model}" if lab else f"of_real_{model}", subset])

    _write_manifest(f"{DATA}/manifests/{subset}.csv", rows)
    open(marker, "w").close()
    n_fake = sum(1 for r in rows if r[1] == 1)
    print(f"{subset}: kept {n_fake} fake / {len(rows) - n_fake} real (scanned {n_seen})")
    print("per-model:", json.dumps(per_model))
    data_vol.commit()


# ---------------------------------------------------------------- fine-tuning

def _train_augment(pil_img, rng, input_size, resize_size):
    """Web-realistic degradation + CF-style crop augmentation."""
    from PIL import Image

    interps = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
    # random rescale (simulates CDN thumbnailing / upscaling)
    if rng.random() < 0.7:
        s = rng.uniform(0.4, 1.3)
        w, h = pil_img.size
        nw, nh = max(64, round(w * s)), max(64, round(h * s))
        pil_img = pil_img.resize((nw, nh), rng.choice(interps))
    # JPEG recompression, sometimes twice
    for _ in range(2):
        if rng.random() < (0.85 if _ == 0 else 0.3):
            buf = io.BytesIO()
            pil_img.convert("RGB").save(buf, format="JPEG", quality=rng.randint(30, 95))
            buf.seek(0)
            pil_img = Image.open(buf).convert("RGB")
    # CF protocol: resize shorter side, random crop
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


@app.function(image=image, volumes={DATA: data_vol}, gpu="H100", timeout=43200, cpu=16, memory=65536, secrets=[hf_secret])
def finetune(
    run_name: str = "ft1",
    epochs: int = 2,
    lr: float = 2e-5,
    batch_size: int = 96,
    input_size: int = 384,
    hf_repo: str = "OwensLab/commfor-model-384",
    train_manifests: str = "openfake_train,coco_train",
    val_manifest: str = "openfake_validation",
    max_steps: int = 0,
    val_every: int = 1500,
    real_fake_balance: bool = True,
):
    import random as pyrandom

    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from torchvision import transforms

    from vendor.cf_models import ViTClassifier

    Image.MAX_IMAGE_PIXELS = None
    torch.backends.cuda.matmul.allow_tf32 = True

    resize_size = 440 if input_size == 384 else 256
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def load_items(names):
        items = []
        for n in names.split(","):
            p = f"{DATA}/manifests/{n}.csv"
            with open(p) as f:
                items += list(csv.DictReader(f))
        return items

    train_items = load_items(train_manifests)
    val_items = load_items(val_manifest)
    n_fake = sum(1 for it in train_items if int(it["label"]) == 1)
    n_real = len(train_items) - n_fake
    print(f"train: {len(train_items)} ({n_fake} fake / {n_real} real); val: {len(val_items)}")

    class TrainDS(Dataset):
        def __len__(self):
            return len(train_items)

        def __getitem__(self, i):
            it = train_items[i]
            rng = pyrandom.Random((hash(it["path"]) ^ pyrandom.getrandbits(32)) & 0xFFFFFFFF)
            try:
                img = Image.open(it["path"]).convert("RGB")
                img = _train_augment(img, rng, input_size, resize_size)
                x = norm(transforms.functional.to_tensor(img))
                return x, float(it["label"])
            except Exception:
                return torch.zeros(3, input_size, input_size), -1.0

    class ValDS(Dataset):
        def __len__(self):
            return len(val_items)

        def __getitem__(self, i):
            it = val_items[i]
            try:
                img = Image.open(it["path"]).convert("RGB")
                img = _degrade(img, "web")  # validate under web-realistic conditions
                w, h = img.size
                s = resize_size / min(w, h)
                img = img.resize((max(input_size, round(w * s)), max(input_size, round(h * s))), Image.BILINEAR)
                w, h = img.size
                x0, y0 = (w - input_size) // 2, (h - input_size) // 2
                img = img.crop((x0, y0, x0 + input_size, y0 + input_size))
                x = norm(transforms.functional.to_tensor(img))
                return x, float(it["label"])
            except Exception:
                return torch.zeros(3, input_size, input_size), -1.0

    if real_fake_balance:
        w_fake, w_real = 0.5 / max(n_fake, 1), 0.5 / max(n_real, 1)
        weights = [w_fake if int(it["label"]) == 1 else w_real for it in train_items]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_items), replacement=True)
        dl = DataLoader(TrainDS(), batch_size=batch_size, sampler=sampler, num_workers=14,
                        pin_memory=True, drop_last=True, persistent_workers=True)
    else:
        dl = DataLoader(TrainDS(), batch_size=batch_size, shuffle=True, num_workers=14,
                        pin_memory=True, drop_last=True, persistent_workers=True)
    vdl = DataLoader(ValDS(), batch_size=128, num_workers=14, pin_memory=True)

    model = ViTClassifier.from_pretrained(hf_repo, device="cpu").cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    total_steps = max_steps or (len(dl) * epochs)
    warmup = min(300, total_steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup
        else 0.5 * (1 + np.cos(np.pi * (s - warmup) / max(1, total_steps - warmup))))
    crit = nn.BCEWithLogitsLoss()

    ckdir = f"{DATA}/ckpts/{run_name}"
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
        ths = np.linspace(0.02, 0.98, 97)
        best = max(((s[y == 1] >= t).mean() + (s[y == 0] < t).mean()) / 2 for t in ths)
        return {"bal@0.65": round((tpr + tnr) / 2, 4), "tpr": round(tpr, 4),
                "tnr": round(tnr, 4), "best_bal": round(float(best), 4)}

    print("initial val:", validate())
    best_bal, step = 0.0, 0
    model.train()
    for ep in range(epochs):
        for xb, yb in dl:
            keep = yb >= 0
            if keep.sum() < 2:
                continue
            with torch.autocast("cuda", torch.bfloat16):
                z = model(xb[keep].cuda()).squeeze(-1)
                loss = crit(z, yb[keep].cuda().to(z.dtype))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"ep{ep} step{step}/{total_steps} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e}")
            if step % val_every == 0 or step == total_steps:
                m = validate()
                print(f"  VAL step{step}: {json.dumps(m)}")
                torch.save({"model_state_dict": model.state_dict(), "step": step, "val": m},
                           f"{ckdir}/last.pt")
                if m["best_bal"] > best_bal:
                    best_bal = m["best_bal"]
                    torch.save({"model_state_dict": model.state_dict(), "step": step, "val": m},
                               f"{ckdir}/best.pt")
                    print(f"  new best ({best_bal})")
                data_vol.commit()
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break

    print("final val:", validate())
    data_vol.commit()
    return {"best_val_bal": best_bal, "ckpt": f"{ckdir}/best.pt"}


@app.function(image=image, volumes={DATA: data_vol}, timeout=1800, cpu=8, memory=16384, secrets=[hf_secret])
def export_onnx(hf_repo: str = "OwensLab/commfor-model-384", input_size: int = 384, ckpt_path: str = ""):
    """Export to ONNX (fp32 + fp16). Output: raw logit; sigmoid+calibration happen in JS."""
    import numpy as np
    import torch

    from vendor.cf_models import ViTClassifier

    if ckpt_path:
        model = ViTClassifier(model_size="small", input_size=input_size, patch_size=16, device="cpu")
        sd = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(sd.get("model_state_dict", sd))
        tag = os.path.basename(ckpt_path).replace(".pt", "")
    else:
        model = ViTClassifier.from_pretrained(hf_repo, device="cpu")
        tag = hf_repo.split("/")[-1]
    model.eval()

    os.makedirs(f"{DATA}/models", exist_ok=True)
    fp32_path = f"{DATA}/models/{tag}.onnx"
    dummy = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(
        model, dummy, fp32_path,
        input_names=["input"], output_names=["logit"],
        dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
    )

    # parity check fp32
    import onnxruntime as onnxrt

    sess = onnxrt.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    x = torch.randn(4, 3, input_size, input_size)
    with torch.no_grad():
        ref = model(x).squeeze(-1).numpy()
    out = sess.run(None, {"input": x.numpy()})[0].squeeze(-1)
    err = float(np.abs(ref - out).max())
    print(f"fp32 parity max logit err: {err:.2e}")
    assert err < 1e-3

    # fp16 conversion (best-effort; fp32 is the fallback deliverable)
    import onnx
    from onnxconverter_common import float16

    fp16_path, err16 = None, None
    for attempt, kwargs in enumerate([
        dict(keep_io_types=True, op_block_list=float16.DEFAULT_OP_BLOCK_LIST + ["Cast"]),
        dict(keep_io_types=True),
        dict(keep_io_types=True, disable_shape_infer=True),
    ]):
        try:
            m = onnx.load(fp32_path)
            m16 = float16.convert_float_to_float16(m, **kwargs)
            cand = f"{DATA}/models/{tag}_fp16.onnx"
            onnx.save(m16, cand)
            sess16 = onnxrt.InferenceSession(cand, providers=["CPUExecutionProvider"])
            out16 = sess16.run(None, {"input": x.numpy()})[0].squeeze(-1)
            err16 = float(np.abs(ref - out16).max())
            print(f"fp16 attempt {attempt} parity max logit err: {err16:.2e}")
            if err16 < 0.05:
                fp16_path = cand
                break
        except Exception as e:
            print(f"fp16 attempt {attempt} failed: {e}")

    for p in filter(None, (fp32_path, fp16_path)):
        print(p, f"{os.path.getsize(p)/1e6:.1f}MB")
    data_vol.commit()
    return {"fp32": fp32_path, "fp16": fp16_path, "fp32_err": err, "fp16_err": err16}


@app.local_entrypoint()
def download_all():
    calls = [
        download_synthbuster.spawn(),
        download_coco.spawn(),
        materialize_openfake.spawn("core"),
        materialize_openfake.spawn("reddit", 300, 1500),
    ]
    for c in calls:
        c.get()
    print("all downloads complete")


@app.local_entrypoint()
def run_eval(hf_repo: str = "OwensLab/commfor-model-384", input_size: int = 384):
    summary = eval_zeroshot.remote(hf_repo, input_size)
    print(json.dumps(summary, indent=2))
