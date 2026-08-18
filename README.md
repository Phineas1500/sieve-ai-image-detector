<img src="assets/sieve.jpg" width="110" align="right" alt="Sieve logo">

# Sieve — Local AI Image Detector for Chrome

A Manifest V3 Chrome extension that detects AI-generated images **entirely on-device**.
No cloud inference, no external APIs, no local server. Images never leave your browser.

Built for the [poidh local-AI challenge](https://poidh.xyz/arbitrum/bounty/323). MIT licensed.

## Results

Held-out proxy benchmark: 36,384 images — SynthBuster (9 commercial 2023 generators,
never trained on), OpenFake test (2026 frontier: GPT Image 2, Midjourney v7,
Nano Banana Pro, Flux.2, …), OpenFake reddit (in-the-wild, never trained on),
COCO val2017 + OpenFake reals. Scored at the fixed **0.65** confidence threshold.

| condition | balanced accuracy | AI recall | real-photo accuracy |
|---|---|---|---|
| clean | **91.3%** | 85.6% | 97.1% |
| web (≤768px, JPEG q60) | **87.3%** | 78.8% | 95.7% |
| hard (≤512px, JPEG q40) | **84.6%** | 74.1% | 95.0% |

For reference, the stock Community Forensics base model scores 75.8 / 65.9 / 56.5
under the same protocol (its detection rate on GPT Image 2 is 7%; ours is >90%).
Inference: ~91ms/image end-to-end (WebGPU, Apple Silicon), ~30ms model time.

## How it works

- A content script finds images on the page as they approach the viewport
  (IntersectionObserver + MutationObserver for dynamic content).
- The extension service worker orchestrates a queue with caching; an offscreen
  document fetches image bytes (host permissions bypass CORS, usually a browser
  cache hit), decodes, and preprocesses them.
- Inference runs in ONNX Runtime Web on **WebGPU**, falling back to **WASM (SIMD)**
  on machines without WebGPU. The model is a ViT-S/16 initialized from
  [Community Forensics](https://github.com/JeongsooP/Community-Forensics)
  (Park & Owens, CVPR 2025, MIT) and fine-tuned on modern generator outputs
  (GPT Image, Midjourney v7, Flux, Nano Banana, …) with heavy web-realistic
  degradation augmentation (JPEG recompression, rescaling).
- Every analyzed image gets a confidence badge; images at or above the
  threshold (default **65%**) are flagged as AI and optionally blurred.
  Click a badge to toggle the blur.
- Model weights (~45 MB) are downloaded **once** during setup from a pinned,
  checksummed URL and stored in OPFS. After that the extension runs fully
  offline and never downloads anything again.

## Install (from source)

Requirements: Node.js ≥ 20, Chrome ≥ 124.

```bash
cd extension
npm install       # pulls onnxruntime-web (pinned)
npm run build     # copies the ORT runtime into extension/vendor/ort/
```

Then:

1. Open `chrome://extensions`, enable **Developer mode**.
2. **Load unpacked** → select the `extension/` directory.
3. The setup tab opens automatically (or click the extension icon → *Model setup*).
   Press **Download model** and wait for "Sieve is active".
4. Browse. Badges appear on analyzed images; the popup shows per-page stats
   and lets you adjust the threshold (bounty evaluation: 65%).

## Reproducing the model

Training runs on [Modal](https://modal.com) (`training/modal_app.py`):

```bash
pip install modal
modal run training/modal_app.py::download_all   # eval + training datasets
modal run training/modal_app.py::run_eval       # zero-shot baseline
# fine-tuning + calibration + ONNX export: see training/README (in progress)
```

Evaluation data: SynthBuster (Zenodo), COCO val2017, and capped subsets of
OpenFake test/reddit splits. Training data: OpenFake train subset + real photo
datasets. Datasets are used under their respective licenses and are not
redistributed with this repository.

## Repository layout

```
extension/            the Chrome extension (load this directory unpacked)
  src/                service worker, offscreen inference, content script, UI
  vendor/ort/         ONNX Runtime Web (copied by npm run build, not committed)
  model_manifest.json pinned model URL + sha256 + calibration constants
training/             Modal pipeline: datasets, eval, fine-tune, ONNX export
  vendor/             Community Forensics model code (MIT, J. Park)
```

## License

MIT. Vendored Community Forensics code is MIT (Copyright Jeongsoo Park).
ONNX Runtime Web is MIT (Microsoft).
