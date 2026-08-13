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

async function createSession() {
  const modelFile = await opfsRead(MODEL_FILE);
  if (!modelFile) throw new Error("no-model");
  modelMeta = await loadMeta();
  const buf = await modelFile.arrayBuffer();

  ort.env.wasm.wasmPaths = chrome.runtime.getURL("vendor/ort/");
  ort.env.wasm.numThreads = 1; // no SharedArrayBuffer in this context

  // Only use WebGPU on a real hardware adapter — software (SwiftShader)
  // WebGPU is far slower than WASM SIMD.
  let hardwareGpu = false;
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    hardwareGpu = !!adapter && !adapter.isFallbackAdapter;
  } catch {}

  try {
    if (!hardwareGpu) throw new Error("no hardware WebGPU adapter");
    const s = await ort.InferenceSession.create(buf, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "all",
    });
    ep = "webgpu";
    return s;
  } catch (e) {
    console.warn("WebGPU EP unavailable, falling back to WASM:", e);
    const s = await ort.InferenceSession.create(buf, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
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

// torchvision test protocol: resize shorter side -> S, center crop C, ImageNet norm.
function preprocess(bitmap, meta) {
  const S = meta.resize_shorter_side, C = meta.input_size;
  const { width: w, height: h } = bitmap;
  const scale = S / Math.min(w, h);
  const rw = Math.max(C, Math.round(w * scale));
  const rh = Math.max(C, Math.round(h * scale));
  const canvas = new OffscreenCanvas(rw, rh);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, rw, rh);
  const x0 = Math.floor((rw - C) / 2);
  const y0 = Math.floor((rh - C) / 2);
  const { data } = ctx.getImageData(x0, y0, C, C);

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

function calibrate(logit, meta) {
  const c = meta.calibration || { temperature: 1.0, bias: 0.0 };
  const z = logit / (c.temperature || 1.0) + (c.bias || 0.0);
  return 1 / (1 + Math.exp(-z));
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

  const bitmap = await createImageBitmap(blob);
  try {
    if (bitmap.width < 32 || bitmap.height < 32) throw new Error("too-small");
    const input = preprocess(bitmap, modelMeta);
    const run = () =>
      session.run({ [session.inputNames[0]]: input }).then((out) => {
        const logit = out[session.outputNames[0]].data[0];
        return calibrate(Number(logit), modelMeta);
      });
    const p = chain.then(run, run);
    chain = p.then(() => {}, () => {});
    const score = await p;
    return { ok: true, score, ms: Math.round(performance.now() - t0), ep };
  } finally {
    bitmap.close();
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
        sendResponse({ ok: true, ready: true, ep, version: (modelMeta || {}).version });
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
