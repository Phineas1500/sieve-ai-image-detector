# Model training & evaluation pipeline

Everything runs on [Modal](https://modal.com) (`pip install modal`, `modal token new`).
A Hugging Face token is optional (all datasets used are public/ungated) but can be
attached as a Modal secret named `huggingface-secret`.

## Model lineage

- Base: [OwensLab/commfor-model-384](https://huggingface.co/OwensLab/commfor-model-384)
  (Community Forensics, Park & Owens CVPR 2025, MIT) — timm ViT-S/16 @ 384, binary head.
- Fine-tuned (run `ft1`): 4,500 steps, batch 96, AdamW lr 2e-5 (300-step warmup,
  cosine), bf16, balanced real/fake sampling, on:
  - **Fakes (108k)**: OpenFake train subset — ~80 generators through 2026
    (GPT Image 2/1.5, Midjourney v7, Nano Banana Pro, Seedream v5, Z-Image Turbo,
    Flux.2, Grok Aurora, Sora 2/Veo 3 frames, …), capped ~1.5k/generator.
  - **Reals (263k)**: OpenFake reals (ImageNet/DOCCI/web), COCO train2017,
    WikiArt paintings (hard negatives against "stylized = AI").
  - **Degradation augmentation** on every sample: random rescale (0.4–1.3×, random
    interpolation), one or two rounds of JPEG re-encoding (q30–95), CF-style
    resize-440 → random-crop-384, horizontal flip.
- Calibration: a single logit bias fit on the OpenFake validation split under a
  pooled clean/web/hard degradation mixture, placing the balanced-accuracy-optimal
  operating point exactly at the bounty's 65% confidence threshold.
- Export: ONNX opset 17, fp16 weights with fp32 I/O boundaries (42 MB).

## Commands (in order)

```bash
# 1. Eval + train data onto the Modal volume (parallel shard workers)
modal run training/modal_app.py::download_synthbuster
modal run training/modal_app.py::download_coco          # val2017 (eval reals)
modal run training/modal_app.py::download_coco_train    # train2017 (train reals)
modal run training/modal_app.py::materialize_wikiart
modal run --detach training/modal_app.py::materialize_all_v2

# 2. Zero-shot baseline (optional, reproduces the stock-model numbers)
modal run training/modal_app.py::run_eval

# 3. Fine-tune (H100, ~40 min)
modal run --detach training/modal_app.py::finetune --run-name ft1 --epochs 2 \
  --max-steps 4500 --val-every 750 \
  --train-manifests "openfake_train,coco_train,wikiart" \
  --val-manifest openfake_validation

# 4. Calibrate the 0.65 operating point
modal run training/modal_app.py::calibrate --ckpt-path /data/ckpts/ft1/best.pt

# 5. Final proxy benchmark with the calibrated model (bias from step 4)
modal run training/modal_app.py::eval_zeroshot --ckpt-path /data/ckpts/ft1/best.pt \
  --bias=<bias from calibration.json>

# 6. Export the shipping artifact
modal run training/modal_app.py::export_onnx_fp16 --ckpt-path /data/ckpts/ft1/best.pt
modal volume get ai-detector-data /models/ft1_best_fp16.onnx .
```

Determinism note: dataset materialization caps per-generator counts with fixed
seeds; training uses a fixed seed for sampling but GPU nondeterminism means
re-trained checkpoints differ slightly. The released checkpoint is the exact
artifact evaluated in the README numbers, with its SHA-256 pinned in
`extension/model_manifest.json`.

## Proxy benchmark

36,384 images, disjoint from training:
- SynthBuster (9 commercial 2023 generators; never trained on)
- OpenFake test split (frontier 2026 generators)
- OpenFake reddit split (in-the-wild social media; never trained on)
- COCO val2017 + OpenFake real test images
Each scored under three conditions: clean, web (≤768px + JPEG q60), hard
(≤512px + JPEG q40), at the fixed 0.65 threshold.

Datasets are used under their respective licenses (OpenFake CC-BY-NC-4.0 for
research/evaluation use; CommunityForensics CC-BY-4.0; SynthBuster
CC-BY-NC-SA-4.0; COCO/WikiArt terms) and are not redistributed here.
