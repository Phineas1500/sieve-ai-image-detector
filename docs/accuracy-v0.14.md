# v0.14.0 accuracy (model w5810)

Model `w5810` = weight average of 2× the v0.12 model (ft58s) and 1× the ft10
round (twin 5k-step continued fine-tunes from ft58s on 620,566 images: FLUX.1-dev
sets, CelebA-HQ / Open Images / FiveK-retouched / Pexels reals, with the
2021–22 cohort and faces run through the benchmark's web/hard delivery path at
train time — `train.py --degrade-view`). Calibration bias **0.20** (fit by
`tools/calibrate_bias.py`: worst-tier balanced accuracy at 0.65 s.t. per-tier
real-photo accuracy ≥ 95%). sha256 `8392faaf0191c339ed76cf68861b1f4fac15da67e8669fa222e3181c59b2d748`.

## Benchmark (36,355 held-out images, fixed 0.65 threshold)

| condition | balanced accuracy | AI recall | real-photo accuracy |
|---|---|---|---|
| clean | 91.3% | 85.8% | 96.8% |
| web | 87.5% | 79.1% | 95.9% |
| hard | 86.2% | 77.4% | 95.0% |

v0.12 (ft58s): 91.6 / 87.5 / 86.2. The −0.3 at clean buys the category
movement below.

## Audit slices, v0.12 → v0.14 (accuracy at 0.65; clean and hard delivery)

| slice (n) | clean | hard |
|---|---|---|
| photo_pexels — lifestyle reals (800) | 74.9 → **97.4** | 73.6 → **95.6** |
| photo_pexels_small (330) | 66.7 → **95.5** | 76.1 → **97.6** |
| press_photo (900) | 94.8 → 96.9 | 91.2 → 94.0 |
| lowlight_phone (560) | 96.1 → 97.7 | 98.6 → 99.6 |
| photo_openimages (500) | 95.4 → 95.4 | 91.4 → 92.4 |
| photo_retouched — FiveK expert edits (150) | 100.0 → 100.0 | 98.7 → 98.7 |
| face_ffhq256 reals (600) | 100.0 → 100.0 | 96.7 → 98.8 |
| lowres_real — Caltech/COCO 256px (1400) | 98.6 → 98.6 | 95.1 → 95.6 |
| ai_flux_dev (800) | 95.6 → 97.0 | 83.4 → 88.6 |
| ai_flux_dev_small (495) | 85.1 → 89.3 | 75.8 → 83.4 |
| ai_flux_face (500) | 98.0 → 98.4 | 95.6 → 97.4 |
| ai_flux_small — native 256px (600) | 91.7 → 94.5 | 96.7 → 97.5 |
| ai_biggan_2022 (240) | 89.6 → 95.4 | 28.7 → 55.0 |
| ai_glide_2022 (240) | 90.0 → 93.3 | 50.0 → 62.5 |
| ai_vqdm_2022 (240) | 92.9 → 94.6 | 40.8 → 57.1 |
| ai_adm_2022 (240) | 64.6 → 72.5 | 19.2 → **25.8 — still broken** |
| ai_stylegan_face (450) | 100.0 → 100.0 | 89.3 → 92.7 |
| ai_nanobanana_pro_wild (1500) | 81.9 → 84.9 | 79.1 → 80.2 |
| ai_gptimage2_wild (1500) | 91.9 → 93.4 | 90.1 → 91.3 |
| yt_thumbnail reals (800) | 91.8 → 92.2 | 87.4 → 89.4 |
| product_catalog (1000) | 98.0 → 98.4 | 95.8 → 96.6 |
| celeb_portrait (400) | 99.8 → 99.8 | 98.8 → 99.0 |
| cartoon_southpark (500) | 100.0 → 100.0 | 100.0 → 100.0 |
| cgi_render (400) | 100.0 → 100.0 | 100.0 → 100.0 |

Honest negatives: ADM at hard delivery stays near-invisible (25.8%) even with
in-distribution training — it is the one 2022 generator this backbone cannot
hold under q40; and the confident smartphone-portrait false positives from the
reported set (#27 #28 #32 #50 #52) remain ≥ 0.92 in every candidate this round
produced. Both stay on the roadmap (#53).

## Regression sets

- Face set (`agentatwork/sieve-corpus-test@07cd96d`, in-extension): StyleGAN
  **24/24** at clean/web/hard; FFHQ reals 9/10 · 10/10 · 8/10 — same counts as
  v0.12, with the false alarms pulled from 0.83/0.91 down to 0.75/0.82.
- Reported-image set, in-extension: **26/38** (v0.12: 25/38) — #39 (press
  false positive) 0.86 → 0.28, nothing regressed; plus the newer reports:
  #40's second image correctly 0.92 AI, #50/#52 still wrong.

The full per-source table (every held-out slice, like docs/accuracy-v0.12.md)
follows once the round's logit CSVs are exported; the numbers above are from
the same eval run (`eval_logits.py`, w5810cat, clean+hard views).
