// Offscreen document: fetches image bytes (extension host permissions bypass
// CORS), decodes, preprocesses to the model's input, and runs ONNX Runtime Web.
// All computation is local; nothing is ever sent anywhere.

import { sniffMetadata } from "./forensics.js";

let sessionPromise = null;
let modelMeta = null;
let ep = "none";

const MODEL_FILE = "model.onnx";
const META_FILE = "model_meta.json";

async function opfsRead(name) {
  const root = await navigator.storage.getDirectory();
  try {
    const fh = await root.getFileHandle(name);
    return await fh.getFile();
  } catch {
    return null;
  }
}

async function loadMeta() {
  const f = await opfsRead(META_FILE);
  if (f) return JSON.parse(await f.text());
  // fall back to bundled manifest defaults
  const resp = await fetch(chrome.runtime.getURL("model_manifest.json"));
  return resp.json();
}

async function opfsWrite(name, data) {
  const root = await navigator.storage.getDirectory();
  const fh = await root.getFileHandle(name, { create: true });
  const w = await fh.createWritable();
  await w.write(data);
  await w.close();
}

// Deterministic dense test input (xorshift32) in normalized-image range:
// exercises the whole network so per-platform fp16 divergence shows up.
function selftestInput(C) {
  const n = 3 * C * C;
  const out = new Float32Array(n);
  let s = 0x5eed1234 >>> 0;
  for (let i = 0; i < n; i++) {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    out[i] = (s / 4294967296 - 0.45) / 0.225;
  }
  return new ort.Tensor("float32", out, [1, 3, C, C]);
}

async function sessionLogit(session, tensor) {
  const res = await session.run({ input: tensor });
  return Number(res[session.outputNames[0]].data[0]);
}

let selftest = "n/a";

async function createSession() {
  const modelFile = await opfsRead(MODEL_FILE);
  if (!modelFile) throw new Error("no-model");
  modelMeta = await loadMeta();
  const buf = await modelFile.arrayBuffer();

  ort.env.wasm.wasmPaths = chrome.runtime.getURL("vendor/ort/");
  ort.env.wasm.numThreads = 1; // no SharedArrayBuffer in this context

  // Debug/test setting: force the WASM path (also exercised in CI so the
  // trimmed vendor set provably serves machines without WebGPU). Offscreen
  // documents lack chrome.storage — ask the service worker.
  let forceWasm = false;
  try {
    const s = await chrome.runtime.sendMessage({ kind: "aid:get-settings" });
    forceWasm = !!(s && s.forceWasm);
  } catch {}

  // Only use WebGPU on a real hardware adapter — software (SwiftShader)
  // WebGPU is far slower than WASM SIMD.
  let hardwareGpu = false;
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    hardwareGpu = !forceWasm && !!adapter && !adapter.isFallbackAdapter;
  } catch {}

  const wasmSession = () => ort.InferenceSession.create(buf, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });

  try {
    if (!hardwareGpu) throw new Error("no hardware WebGPU adapter");
    const s = await ort.InferenceSession.create(buf, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "all",
    });
    // Numerical self-test: WebGPU stacks (drivers, Chromium forks) can
    // silently compute this fp16 graph WRONG while running fine — verified
    // in the field (stable 5-logit divergence on one browser). Compare a
    // fixed input against the deterministic WASM reference; GPU speed is
    // not worth wrong verdicts.
    const x = selftestInput(modelMeta.input_size || 384);
    const zGpu = await sessionLogit(s, x);
    let ref = modelMeta.wasm_ref_logit;
    if (typeof ref !== "number") {
      ref = await sessionLogit(await wasmSession(), x);
      modelMeta.wasm_ref_logit = ref;
      opfsWrite(META_FILE, JSON.stringify(modelMeta)).catch(() => {});
    }
    if (Math.abs(zGpu - ref) > 0.5) {
      console.warn(`WebGPU self-test FAILED: logit ${zGpu.toFixed(3)} vs WASM ref ${ref.toFixed(3)} — using WASM`);
      selftest = `fallback (gpu drift ${(zGpu - ref).toFixed(2)})`;
      ep = "wasm";
      return wasmSession();
    }
    selftest = "ok";
    ep = "webgpu";
    return s;
  } catch (e) {
    console.warn("WebGPU EP unavailable, falling back to WASM:", e);
    const s = await wasmSession();
    ep = "wasm";
    return s;
  }
}

function getSession() {
  if (!sessionPromise) sessionPromise = createSession();
  return sessionPromise;
}

async function fetchImage(url) {
  if (url.startsWith("data:")) {
    const resp = await fetch(url);
    return resp.blob();
  }
  const resp = await fetch(url, { credentials: "omit", cache: "force-cache" });
  if (!resp.ok) throw new Error(`fetch ${resp.status}`);
  return resp.blob();
}

function imageDataToTensor(data, C, meta) {
  const [mr, mg, mb] = meta.norm_mean;
  const [sr, sg, sb] = meta.norm_std;
  const n = C * C;
  const out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    const j = i * 4;
    out[i] = (data[j] / 255 - mr) / sr;
    out[n + i] = (data[j + 1] / 255 - mg) / sg;
    out[2 * n + i] = (data[j + 2] / 255 - mb) / sb;
  }
  return new ort.Tensor("float32", out, [1, 3, C, C]);
}

// Decode to raw RGBA at native size (resampling happens in our own code, so
// browser resize quirks never touch the pixels). Preferred path: canvas 1:1
// draw + getImageData — matches libjpeg/libpng decoding faithfully. But
// canvas readback is a fingerprinting surface: privacy browsers (Brave,
// Helium, …) can inject deterministic ±1 pixel noise into it — invisible to
// people, but exactly the kind of high-frequency pattern a forensic model
// reads as a generator artifact (field-verified: a real photo pinned at 95%
// by perturbed pixels). At startup we decode a known image and verify the
// readback byte-exactly; if the canvas lies, pixels come from WebCodecs
// ImageDecoder instead (not a fingerprinting surface; its JPEG YUV→RGB
// conversion differs slightly from libjpeg, which is far better than noise).
const CANVAS_PROBE_PNG = Uint8Array.from(atob(
  "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"
), (c) => c.charCodeAt(0));
// the probe's true pixels (32x32 RGB) — random noise, because fingerprint
// defenses target high-entropy content and must not be able to pass on a
// trivially solid image
const CANVAS_PROBE_RGB = Uint8Array.from(atob(
  "lwwUQ73gfo9cg58o6CRsNqdMM0fZVQeLp8qVdVpZsghyFy1SUnBCPFRktYRIuJSXCbEOQ4ZRxFW2ueD5Wk0+RRe8mbuYOrVOgZyCHRsTaM1d7OJiBptbGBZ05aBywfZcwssDIGlevL1o18n7y2TyKEEvd3E8eNbc134gypOSNsruWkvLUSFeAW+je1xb2qx2s8mFajm3jWHD8JGVlIE/hQiemsJILrVePTYUMEdOw3l6UsVJWwcT3Yl2k6vEcq9n3/Lof+G3xHBMrbYthgMPJMSgw2n0X/fRcnX+MvWlzOWucx2I9xas7RQ/FtUSppqeGptsL8OmGfVpfhX1oj6J4ar552zEokbQ+5oGgQiiN7FOOa6C5p9nsYPoCgRY33LWXQpQu4gu4p0vPUn+R8NDqoRXsinjpix1EiNrIL1sd2oJIvVf8cmNoVzUdrFNeVzOMh++AEw0mK2wy4VeLREZ2bObBQ+4GYvC8u0RyqGS2fWteYpZAH+NDTGY2LNJzK2GVDDBff8RnK0tKQ754j7Cv7D9AdbwT/iJm4M4nPyF20PHUISQPfiD1hTlVP92Qie81Zk1QSaSS9xsRK4Ds1rATUCoyxYQ5gmc9Clh8NL83EaFojT56fIk4tm70X2wQ7ZJcpRHV13YPoJyMg2515qNJVMhygKGuhiwikqMwu2Az+OKDeGreu7PibMJwTBtJ0jEem2dqpCl0XVbUT/pmxHWqgOHgClppgGODKkb6e3hV8qPHglTAAzv38zBRrMHOGzcs430Q8sCYU5BedqEJkkR+6GKbi0oPzu588DEd/wol3W9Vu9y3CEHrIl9ZGXEn5viB6C7yiHJrXv2WE9+h43paV+mVLkKmOAOuo4Dywff77DWe/1Zw4wfA7sN3+iLzaBehct/tuquPpdPBbLlnjT8ev7FkmupcPszlCKvdxHWuKnAxWfn17q1+Nc9MKJCqJF3BINniITY7gRpgc08RhLMQ7EUTvLm4GGHH9A2j908xSUyyuHBDI4vgU5GdDz6mh5NF6sEbok3w2D1oORkegyr/dzZvYIAk3rLJWkaIFGZJzjS4wOvvba3j1usFgcFkA6oIiHdO5HSqLV6doe87mMIhrNfKO9reXlUUSUVnb5wURRBflBoUy85XgdCmtoEKksDdzkq/gdQXKG7YZ1LbKqpsJYE4mMsShbABtXusnq16zjpoGVb7AixOF4+LAhkENVj9ldhWeSySX36Jxl8gngrM68iWQzyHLpWxho/6gFhud74u765DSEDed4IwMXQdJ4kOvKwGFhzBPOQ5YcJSEXEuZ0Ug0ITJNhKmkYhIG3ingBbXfc8Zr1IVjLvoq5BdaHY/SQxe9MFulSDA7/vJcxbdGmK1pRixHU5p7ykL/wh09PfhtovbFJL91Zzq1aVdo4I/9XJbuaMeDjBN6cTzrz9RSZ30aLe27wRge3zDhtsyHXlFEO5b7LHWLRx/Ki8XftTAO9BAg7XAM7R2l/Dwjj62aCVsFZkn9W7TeG1xumCwpDFwfZVQSlky6FonpRv184qV64GLA8RmVqluKHYRAQm40TmWUV3AaCZFq2gvp+OcMm5RTyQ/kLinvAobOZs3S3mH9ZejF6mtjL1qW8Gqpt81CvNpBQ77uKQRYOf9MlnKJiie750MldL1QGMuCWrBfEXmD7KrNw+ZbJ6nqifvCc7aHVQ2owwCk7mcEHf52o3aKqJOvjlhhoGvKzOLEMkYv0KzagauIeIAsZ9jzJ8NL0pvIcR6EeZHn6tdWbrDlW86WyHtn40oSWNhf2Z2aO3casKKZ01STdq7OalMEpxWUal9Uvlj1gSQVNFScXuf0dpV/b4zrKns4qMcTrbDPNYNNQj3Y13CO3b3oLbO2yOq2GrtOx6C6SsccNR4paPKA4ENEUXePi5UpSF6Gex5mCW67uZzib1gh3vG3k4mweUikdWsBJSZNjc9BR+oBGrszNTJdhakxKh4Evcm5vRJA2gwo9jOcHAC+OMMvC8h22H1gRIRg9+dchtpHNtP+n8s6fB3b0xnVUyYxEd7sQKkUV672xMt6YaNxfkN6DJbIloth56CKQrwKcFbPqd2K4zReMVEbfvN8tH2b689ZwSIsijaasofk0k2lB32aBfxySMAQ3umU+OX7l2SEHcCRXkbhOOitYffw7SubBAm7JCB9N0Yafbow88BEK1w4b9hai73DNoCyo9nbdBlnIsT0AhuGj/cLOSGDzJ2EjadKXJ6+8Dq3EM/F59qFVO/0J93P8vu7dZsjk6LrDgAdq05EWK/VNKi54QLRTqTAPxGAgEdt345VgDsLv0C9RTh7FxvAECUDHpXeer3794GvfPqkXJuxdF1+tNv1e1/BwFCZqQNiiqq4MwAApZHiDDCd6HGhApcoVy51x0u9LLxCNsBBBaZ+PnfcCkgtstvG1SMzwep6EYHggpOl7CCWX4WEj+JBcYLJedE/mTlxyR1n+VFPt/sEwEDUyTIwWen0+1V/+95JKNSsPCg+Z703QZxG6Swqm+77k/AjrQBTqnBIKsAd9/FhWeoE0KVz2A/hHQXxIbTlB1zUtkYeiZRO2jPVY0yWqC8d2qQYTvnbPZnzEEMDU96CG9m6UoySt0zyWtl4RbgIgw8NqEIIMOzzPyFmagQqRj8eZQ/8gNNL5KkBrfZhcj/AoA8GqoNfM1PEABaesAE5SBD32yyfKAuGS7Fn+DNDEiKwiQ6ERYIdCjBFIf+5yzvDSo7nxP4pC+mvnfo6c78tfWD5ik1TXwvyTU49tXpbm9KdlIoo/qU6Q6exLOcvPEowFTwQ6SIe7TGIWyOUkWg9U+Ubmw0KGzOO55rM7XVoSpaT3WhW1b4S249HenhOZQ0hrXDuilCCYIPdaFdGJexAjoxJI/9Ff5BNwGU5SSwkSl3jVrl2uTGoj8aos9nEAOB8ItXI2n8cQG5xoWaD4ZaL82qq6kHY2qv76Jt6HWaz47VB4QT0DU1shvY60NtPkf2EplCVmn45nUAUIWU4Y5G44Tnwk1GhQasTfMNLtxBKfFev3SeKZZHr0lgRehxbA1aJUMUOQqkNyr5MY4jBLNabIIH/SVQgSo0AQVdVO5nG1Z1lwzCgLzEoZ37fM2opwZMQW3vS/TOaBAlpDUim9HnYr6lR9fogQvTbrIjo5WQCClH6nxbAbXhAwrnyC8mzIAHAwP3whoI/fBi03nX3XLXnEzvicPnoY++Myjaa92dl4fRbcTsj1Umlu842rTJTdq7Jpp1z7Y5vvVs/rtomL7aPqQ+LQtZ/MOMjlOWVfa3/5b3Yw1RCXXLaXq9W4E6hfuRleggkvWjAAMP1SDq9A/4ZRIp39NHL8JMMxBnXfOpBHK7m4MYr2YXmHmQPkBJwqTU22GNEHm0LumNUnO076i8Z//0Pyt8G3DT/KyCa6SFXi/YsldMkp9N2fBkv3xIcPUwzaYQxKy3VEL0LBFQMLI6+hBCyvutw54pbaHwlRhrd9zVgyPTjK/pAXcaNUGIzV1NcC+LkF6pgCd0u3/4/oUle22NsqLfQ+wmsYVkChpwbqPVdpJgvojEWqZulJWR8/71aAm9dEIU86UvgV8IXpUn76Lyhk2uQ/qPmOd5I636kkdN3FaUx8VYv2RRpaLY29XCL1r7QuKHLM4URayRVr2T2pb2+Mp6LxAPU7vjJQf5fZD9banCzxbRo4erSIvJMYlnDKybtvxPVSb9kjkwa3b7sJLG2w5s6IcN2KrN3x7IcHk0Ih+AhAwDqkoMcj4wemUC8GS+gg5Yc2YOpnIfOQInGBfq1njNzIkWKUMDr0f/+EyqOfuhFi2Jbd6St2I4Cgw+m53ikAdlyFHoN2664R2zqfITeWOAcnubwYk4rHdW+JusyGd+kYjEhvqVDqu4lViQP/OR4IrG0Dep1WjNQvvqR3mRgzrMPoJUYsWHyTuawK3/TIFhja/ImFPe7rt+3w8RcukEF2ORWjyJxeMXGNh9ugwVARNO406nCkwIWHpvwT/okABe5PJkfy1yYgdDrHB4CoNzQhhuGCApgEc1x14yM8iecCBjkjAy7NOG19WQ3Yt8kpbkg5hf5v4STULr4sOGSaksWuEBJQ6wbpXgKiAqyR3f2ja"
), (c) => c.charCodeAt(0));
let canvasTrustworthy = null;

async function canvasDecode(blob) {
  const bitmap = await createImageBitmap(blob);
  try {
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0);
    const { data } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
    return { data, width: bitmap.width, height: bitmap.height };
  } finally {
    bitmap.close();
  }
}

async function webcodecsDecode(blob) {
  const dec = new ImageDecoder({ data: await blob.arrayBuffer(), type: blob.type || "image/*" });
  const { image: frame } = await dec.decode();
  try {
    const rect = frame.visibleRect || { x: 0, y: 0, width: frame.codedWidth, height: frame.codedHeight };
    const opts = { format: "RGBA", colorSpace: "srgb", rect };
    const buf = new Uint8ClampedArray(frame.allocationSize(opts));
    const [layout] = await frame.copyTo(buf, opts);
    const w = rect.width, h = rect.height;
    let data = buf;
    if (layout && (layout.stride !== w * 4 || layout.offset !== 0)) {
      data = new Uint8ClampedArray(w * h * 4);
      for (let y = 0; y < h; y++) {
        data.set(buf.subarray(layout.offset + y * layout.stride,
                              layout.offset + y * layout.stride + w * 4), y * w * 4);
      }
    }
    return { data, width: w, height: h };
  } finally {
    frame.close();
    dec.close();
  }
}

async function probeCanvasIntegrity() {
  try {
    const { data } = await canvasDecode(new Blob([CANVAS_PROBE_PNG], { type: "image/png" }));
    for (let i = 0, p = 0; i < data.length; i += 4, p += 3) {
      if (data[i] !== CANVAS_PROBE_RGB[p] ||
          data[i + 1] !== CANVAS_PROBE_RGB[p + 1] ||
          data[i + 2] !== CANVAS_PROBE_RGB[p + 2]) return false;
    }
    return true;
  } catch {
    return false;
  }
}

async function decodeRGBA(blob) {
  if (canvasTrustworthy === null) {
    canvasTrustworthy = await probeCanvasIntegrity();
    if (!canvasTrustworthy) console.warn("canvas readback is noisy (fingerprint protection?) — decoding via WebCodecs");
  }
  if (!canvasTrustworthy && typeof ImageDecoder !== "undefined") {
    try {
      return await webcodecsDecode(blob);
    } catch {
      // unsupported container — noisy canvas beats no decode
    }
  }
  return canvasDecode(blob);
}

// Pillow-style separable triangle (bilinear, antialiased) resampling — the
// same math as the training/eval pipeline (PIL Image.resize BILINEAR), so
// scores match the Python benchmark instead of whatever Skia version the
// browser ships. Deterministic across every browser.
function resamplePIL(src, dw, dh) {
  const pass = (data, sw, sh, dsize, horizontal) => {
    const srcSize = horizontal ? sw : sh;
    const scale = srcSize / dsize;
    const filterscale = Math.max(scale, 1.0);
    const support = filterscale; // triangle support = 1.0 * filterscale
    const ow = horizontal ? dsize : sw;
    const oh = horizontal ? sh : dsize;
    const out = new Float32Array(ow * oh * 4);
    const bounds = [];
    for (let i = 0; i < dsize; i++) {
      const center = (i + 0.5) * scale;
      const lo = Math.max(0, Math.floor(center - support));
      const hi = Math.min(srcSize, Math.ceil(center + support));
      const w = new Float32Array(hi - lo);
      let sum = 0;
      for (let k = lo; k < hi; k++) {
        const t = Math.abs((k + 0.5 - center) / filterscale);
        const v = t < 1 ? 1 - t : 0;
        w[k - lo] = v;
        sum += v;
      }
      if (sum > 0) for (let k = 0; k < w.length; k++) w[k] /= sum;
      bounds.push([lo, w]);
    }
    for (let y = 0; y < oh; y++) {
      for (let x = 0; x < ow; x++) {
        const [lo, w] = bounds[horizontal ? x : y];
        let r = 0, g = 0, b = 0;
        for (let k = 0; k < w.length; k++) {
          const j = horizontal ? (y * sw + lo + k) * 4 : ((lo + k) * sw + x) * 4;
          r += data[j] * w[k];
          g += data[j + 1] * w[k];
          b += data[j + 2] * w[k];
        }
        const o = (y * ow + x) * 4;
        out[o] = r; out[o + 1] = g; out[o + 2] = b; out[o + 3] = 255;
      }
    }
    return out;
  };
  let d = pass(src.data, src.width, src.height, dw, true);
  d = pass(d, dw, src.height, dh, false);
  const clamped = new Uint8ClampedArray(dw * dh * 4);
  for (let i = 0; i < d.length; i++) clamped[i] = Math.round(d[i]);
  return { data: clamped, width: dw, height: dh };
}

function cropRGBA(src, x0, y0, C) {
  const out = new Uint8ClampedArray(C * C * 4);
  for (let y = 0; y < C; y++) {
    const s = ((y0 + y) * src.width + x0) * 4;
    out.set(src.data.subarray(s, s + C * 4), y * C * 4);
  }
  return out;
}

// torchvision test protocol: resize shorter side -> S, center crop C, ImageNet norm.
function preprocess(img, meta) {
  const S = meta.resize_shorter_side, C = meta.input_size;
  const scale = S / Math.min(img.width, img.height);
  const rw = Math.max(C, Math.round(img.width * scale));
  const rh = Math.max(C, Math.round(img.height * scale));
  const resized = resamplePIL(img, rw, rh);
  const data = cropRGBA(resized, Math.floor((rw - C) / 2), Math.floor((rh - C) / 2), C);
  return imageDataToTensor(data, C, meta);
}

// TTA view: native-resolution center crop — no resampling, preserves the
// pixel-grid artifacts the standard resized view destroys.
// Second view for images below the native-crop size: scale the shorter side
// straight to the crop size (a 440/384 scale jitter vs. the standard view)
// instead of skipping TTA entirely in the regime where the model is weakest
// (audit #48: profile pictures / thumbnails).
function preprocessFit(img, meta) {
  const C = meta.input_size;
  const scale = C / Math.min(img.width, img.height);
  const rw = Math.max(C, Math.round(img.width * scale));
  const rh = Math.max(C, Math.round(img.height * scale));
  const resized = resamplePIL(img, rw, rh);
  const data = cropRGBA(resized, Math.floor((rw - C) / 2), Math.floor((rh - C) / 2), C);
  return imageDataToTensor(data, C, meta);
}

// Delivery-degradation estimate on the decoded pixels (audit #41/#42): heavy
// recompression shows as energy piling up on the 8px JPEG block grid
// (blockiness ~1.0 for clean, ~1.4 at q40, >2 when "deep-fried"); content
// upscaled far beyond its true resolution has no pixel-scale texture left,
// which shows as the mean 1px luma step falling to ~half the mean 2px step
// (d12 ~0.50 for anything interpolated up, 0.55-0.9 for native images).
// d12 replaced a Laplacian-energy ratio before v0.13.0 shipped: mean
// high-frequency energy could not tell an upscaled thumbnail from a smooth
// native AI render (19% of as-posted reddit AI images tripped it), while d12
// at 0.528 trips ~1% of native images and catches 99% of 64px and ~80% of
// 128px bicubic upscales (tools/upscale_metric_scan.py). Either regime is
// one where a verdict is not evidence-backed, so the badge shows "unsure"
// instead of "AI".
function degradationStats(img) {
  const { data, width, height } = img;
  if (width < 24 || height < 24) return { block: 1, d12: 1 };
  const L = (i) => 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  const sy = Math.max(1, Math.floor(height / 192));
  let dxAll = 0, dxN = 0, dxB = 0, dxBN = 0, dyAll = 0, dyN = 0, dyB = 0, dyBN = 0, d2All = 0;
  for (let y = 1; y < height - 1; y += sy) {
    const row = y * width * 4, dn = (y + 1) * width * 4;
    let prev = L(row);
    for (let x = 1; x < width - 1; x++) {
      const i = row + x * 4;
      const c = L(i), r = L(i + 4), d = L(dn + x * 4);
      const dx = Math.abs(r - c), dy = Math.abs(d - c);
      dxAll += dx; dxN++; dyAll += dy; dyN++;
      d2All += Math.abs(r - prev);
      if (x % 8 === 7) { dxB += dx; dxBN++; }
      if (y % 8 === 7) { dyB += dy; dyBN++; }
      prev = c;
    }
  }
  const bx = dxBN ? (dxB / dxBN) / (dxAll / dxN + 1e-6) : 1;
  const by = dyBN ? (dyB / dyBN) / (dyAll / dyN + 1e-6) : 1;
  return { block: (bx + by) / 2, d12: (dxAll / dxN) / (d2All / dxN + 1e-6) };
}

function preprocessNative(img, meta) {
  const C = meta.input_size;
  const data = cropRGBA(img, Math.floor((img.width - C) / 2), Math.floor((img.height - C) / 2), C);
  return imageDataToTensor(data, C, meta);
}

function calibrate(logit, meta) {
  const c = meta.calibration || { temperature: 1.0, bias: 0.0 };
  const z = logit / (c.temperature || 1.0) + (c.bias || 0.0);
  return 1 / (1 + Math.exp(-z));
}

// Degenerate inputs (solid fills, pure noise) sit far outside the training
// manifold and score confident nonsense (solid black -> 0.91, RGB noise ->
// 0.99). Two cheap statistics separate them from photographs: luma variance
// is ~0 for flat fills, and adjacent-pixel correlation is ~0 for noise,
// while any real image is strongly locally correlated. No verdict is badged
// for these — there is nothing meaningful to say.
function degenerateReason(img) {
  const { data, width, height } = img;
  const sy = Math.max(1, height >> 7); // sample ~128 rows
  let n = 0, sum = 0, sum2 = 0;
  let m = 0, sa = 0, sb = 0, sab = 0, sa2 = 0, sb2 = 0;
  for (let y = 0; y < height; y += sy) {
    const row = y * width * 4;
    let prev = -1;
    for (let x = 0; x < width; x++) {
      const i = row + x * 4;
      const l = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      n++; sum += l; sum2 += l * l;
      if (prev >= 0) { m++; sa += prev; sb += l; sab += prev * l; sa2 += prev * prev; sb2 += l * l; }
      prev = l;
    }
  }
  const varL = sum2 / n - (sum / n) ** 2;
  if (varL < 4) return "flat"; // luma std < 2 of 255
  const cov = sab / m - (sa / m) * (sb / m);
  const denom = Math.sqrt(Math.max((sa2 / m - (sa / m) ** 2) * (sb2 / m - (sb / m) ** 2), 1e-9));
  if (cov / denom < 0.15) return "noise";
  return null;
}

// serialize inference; decode may overlap
let chain = Promise.resolve();

async function infer(url) {
  const t0 = performance.now();
  const [session, blob] = await Promise.all([getSession(), fetchImage(url)]);

  // High-precision metadata forensics first: a structural marker of AI
  // generation short-circuits inference (score can only go up, never down).
  const meta = sniffMetadata(new Uint8Array(await blob.arrayBuffer()));
  if (meta.hit) {
    return { ok: true, score: 0.99, ms: Math.round(performance.now() - t0), ep: "metadata", reason: meta.reason };
  }

  const img = await decodeRGBA(blob);
  {
    if (img.width < 32 || img.height < 32) throw new Error("too-small");
    const degen = degenerateReason(img);
    if (degen) throw new Error(`degenerate-${degen}`);
    const dq = degradationStats(img);
    const degraded = dq.block >= 1.8 || dq.d12 < 0.528;
    const input = preprocess(img, modelMeta);

    const runLogit = (tensor) =>
      session.run({ [session.inputNames[0]]: tensor }).then((out) =>
        Number(out[session.outputNames[0]].data[0]));

    // Selective TTA: when the standard view lands in the uncertainty band and
    // the image is large enough for a native crop, average raw logits with a
    // second, resampling-free view (measured: +0.9/+0.5/+0.3 balanced acc).
    let tta = false;
    let msModel = 0;
    const task = async () => {
      const t1 = performance.now();
      const zStd = await runLogit(input);
      const cfg = modelMeta.tta || {};
      const C = modelMeta.input_size;
      let z = zStd;
      if (cfg.enabled) {
        const s = calibrate(zStd, modelMeta);
        const large = Math.min(img.width, img.height) >= (cfg.min_side || C);
        // small_view:false restores the pre-v0.13 behaviour (no second view
        // below min_side) — kept as a manifest switch for A/B measurement
        if (s >= cfg.band_lo && s <= cfg.band_hi && (large || cfg.small_view !== false)) {
          const z2 = await runLogit(large ? preprocessNative(img, modelMeta) : preprocessFit(img, modelMeta));
          z = (zStd + z2) / 2;
          tta = true;
        }
      }
      msModel = Math.round(performance.now() - t1);
      return calibrate(z, modelMeta);
    };
    const p = chain.then(task, task);
    chain = p.then(() => {}, () => {});
    const score = await p;
    // ms = model time (both views when TTA fires); msTotal includes fetch,
    // decode and time spent queued behind other images
    return { ok: true, score, ms: msModel, msTotal: Math.round(performance.now() - t0), ep, tta, degraded,
             quality: { block: +dq.block.toFixed(3), d12: +dq.d12.toFixed(4) } };
  }
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.kind === "aid:infer") {
    infer(msg.url)
      .then(sendResponse)
      .catch((e) => sendResponse({ ok: false, error: String(e?.message || e) }));
    return true;
  }
  if (msg?.kind === "aid:status") {
    (async () => {
      try {
        await getSession();
        sendResponse({ ok: true, ready: true, ep, selftest, version: (modelMeta || {}).version });
      } catch (e) {
        sessionPromise = null;
        sendResponse({ ok: true, ready: false, error: String(e?.message || e) });
      }
    })();
    return true;
  }
  if (msg?.kind === "aid:reload-model") {
    sessionPromise = null;
    getSession()
      .then(() => sendResponse({ ok: true }))
      .catch((e) => sendResponse({ ok: false, error: String(e?.message || e) }));
    return true;
  }
  return false;
});
