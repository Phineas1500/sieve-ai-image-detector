// Service worker: orchestrates analysis requests from content scripts.
// Inference itself runs in the offscreen document (WebGPU/WASM live there).

const OFFSCREEN_URL = "src/offscreen.html";

const cache = new Map(); // url -> {score, ms}
const CACHE_MAX = 3000;
const inflight = new Map(); // url -> Promise

let offscreenReady = null;

async function ensureOffscreen() {
  if (!offscreenReady) {
    offscreenReady = (async () => {
      const has = await chrome.offscreen.hasDocument();
      if (!has) {
        await chrome.offscreen.createDocument({
          url: OFFSCREEN_URL,
          reasons: ["WORKERS"],
          justification: "Runs local ML inference (ONNX Runtime Web, WebGPU/WASM) on images.",
        });
      }
    })();
  }
  return offscreenReady;
}

function cachePut(url, result) {
  if (cache.size >= CACHE_MAX) {
    const first = cache.keys().next().value;
    cache.delete(first);
  }
  cache.set(url, result);
}

async function analyze(url) {
  if (cache.has(url)) return cache.get(url);
  if (inflight.has(url)) return inflight.get(url);
  const p = (async () => {
    await ensureOffscreen();
    const resp = await chrome.runtime.sendMessage({ kind: "aid:infer", url });
    if (resp && resp.ok) {
      const result = { score: resp.score, ms: resp.ms };
      cachePut(url, result);
      return result;
    }
    return { error: resp ? resp.error : "no response" };
  })().finally(() => inflight.delete(url));
  inflight.set(url, p);
  return p;
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.kind === "aid:analyze") {
    analyze(msg.url).then(sendResponse);
    return true; // async
  }
  if (msg?.kind === "aid:model-status") {
    (async () => {
      await ensureOffscreen();
      const resp = await chrome.runtime.sendMessage({ kind: "aid:status" });
      sendResponse(resp);
    })();
    return true;
  }
  if (msg?.kind === "aid:model-updated") {
    // Setup page stored a (new) model; offscreen must reload its session.
    (async () => {
      await ensureOffscreen();
      await chrome.runtime.sendMessage({ kind: "aid:reload-model" });
      cache.clear();
      sendResponse({ ok: true });
    })();
    return true;
  }
  return false;
});

chrome.runtime.onInstalled.addListener(async (details) => {
  if (details.reason === "install") {
    chrome.tabs.create({ url: chrome.runtime.getURL("src/setup.html") });
  }
});
