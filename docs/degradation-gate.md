# Degraded-input gate — calibration (v0.13.0)

The extension refuses to show a red "AI" verdict on inputs delivered in a
regime where the model's answer is not evidence-backed. Two cheap statistics
on the decoded luma (sampled rows, `offscreen.js degradationStats()`):

| statistic | definition | trips when | targets |
|---|---|---|---|
| `block` | mean \|Δ\| across 8-px JPEG grid boundaries / mean \|Δ\| everywhere, averaged over x and y | ≥ 1.8 | heavy recompression ("deep-fried" reals, audit #41). Native 0.97–1.07, q40 1.33–1.60, deep-fry 2.3–2.8 |
| `d12` | mean \|1-px luma step\| / mean \|2-px luma step\| | < 0.528 | content upscaled far beyond its true resolution (audit #42). ≈0.50 for anything interpolated up; 0.55–0.9 for native images |

A degraded input that clears the 65% threshold is badged amber **unsure n%**
(dashed marker, tooltip explains) instead of red, and is never blurred or
hidden. Sub-threshold scores are unaffected. Python reference:
`tools/degradation_scan.py stats()`; JS and PIL agree to 4 decimals on the
e2e samples.

## Why d12 replaced the Laplacian-energy ratio

The v0.13 release candidate used `hf = Laplacian variance / global variance
< 0.01`. Joining the seeded false-trigger scan (`tools/degradation_scan.py`,
500 images per manifest × clean/web/hard views) with the ft58s logits
(`tools/degradation_join.py`) showed what the gate actually changed at 0.65:

| manifest / view | AI scored red | demoted to amber (hf<0.01) | real FPs | rescued to amber |
|---|---|---|---|---|
| openfake_reddit clean | 224 | **42 (18.8%)** | 22 | 3 |
| openfake_reddit web | 217 | 11 (5.1%) | 26 | 1 |
| openfake_reddit hard | 228 | 10 (4.4%, block gate) | 36 | 1 |
| openfake_test clean | 318 | 13 (4.1%) | 2 | 1 |
| openfake_test web | 304 | 4 (1.3%) | 3 | 1 |
| synthbuster clean/web/hard | 429/373/344 | 2/2/3 | 0 | – |
| coco clean/web/hard | – | – | 2/8/12 | 0 |

Mean high-frequency energy cannot tell an upscaled thumbnail from a smooth
native AI render: as-posted reddit AI images have p10 `hf` = 0.005, the same
as a 64-px image blown up to full size. Lowering the threshold traded catch
rate for demotions almost 1:1.

`tools/upscale_metric_scan.py` compared five statistics on 1,050 images
(150 per manifest; negatives = native clean + web views, 2,100; positives =
the same images downscaled to side *s* and bicubic-upscaled back, raw and
JPEG q75). At the threshold that trips 1% of negatives:

| metric | reddit-AI demoted clean / web | catch 64px raw / q75 | 96px raw / q75 | 128px raw / q75 |
|---|---|---|---|---|
| hf (lapVar / globalVar) | 8.5% / 3.2% | 82% / 71% | 63% / 54% | 51% / 44% |
| lg (lapVar / gradVar) | 3.2% / 3.2% | 96% / 82% | 93% / 79% | 78% / 63% |
| **d12 (mean\|d1\| / mean\|d2\|)** | **2.1% / 1.1%** | **99% / 95.5%** | **93% / 79%** | **79% / 62%** |
| p99 \|lap\| / sd | 9.6% / 3.2% | 83% / 76% | 63% / 57% | 51% / 45% |
| p99.9 \|lap\| / sd | 11.7% / 2.1% | 92% / 89% | 75% / 69% | 61% / 57% |

d12 at 0.528 (the 1% point) is what ships. Native trip rates at that
threshold: coco 0.0%, small_neg 0.3%, small_pos2 0.0%, synthbuster 1.7%,
openfake_test 1.1% real / 3.3% AI, openfake_reddit 0.9% real / 1.6% AI,
cartoons 1.5% real / 0.6% AI.

The block gate is unchanged: on the benchmark it trips 1.6–5.8% of hard-tier
(≤512 px, q40) images and almost nothing at clean/web; q40 is below its
deep-fry target and the demotions there are the price of catching real
deep-fries.

## Small-image TTA A/B (v0.13.0)

For in-band images (score 0.25–0.85) with a shorter side below `min_side`
(384), the extension averages the standard view with a fit view (shorter
side resampled to the model's input size). Measured through the real
extension (`eval/e2e/score-dir.mjs`, with and without `--no-small-tta`) on
the local small held-out sets, accuracy at 0.65:

| slice | no small TTA | small TTA | Δ | TTA fired on | accuracy on that subset |
|---|---|---|---|---|---|
| small_pos2 held-out (AI, 800) | 84.1% | 84.8% | +0.6 | 163 | 55.8% → 58.9% |
| small_neg held-out (real, 798) | 95.2% | 95.1% | -0.1 | 134 | 82.1% → 81.3% |
| small_pos held-out (AI, 800) | 86.2% | 86.0% | -0.3 | 149 | 51.7% → 50.3% |
| all | 88.5% | 88.6% | +0.1 | – | – |

Net effect is marginal; it is kept because it never hurts materially and the
cost is one extra forward pass on the ~20% of small images that land in the
band. `tta.small_view: false` in `model_manifest.json` disables it.

## Reproduce

```
# on the training node ($DATA = benchmark + category manifests)
python tools/degradation_scan.py --manifests coco,openfake_test,openfake_reddit,synthbuster,cartoons_heldout,small_pos2_heldout,small_neg_heldout --per 500
BIAS=0.30 python tools/degradation_join.py
PER=150 python tools/upscale_metric_scan.py
# locally
cd eval/e2e && node score-dir.mjs --model ../../dev_model/ft58s_best_fp16.onnx --bias 0.30 --dir <images> --out out.json [--no-small-tta 1]
python tools/small_tta_ab.py dev_model/small_heldout
```
