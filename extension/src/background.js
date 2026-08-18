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
      const result = { score: resp.score, ms: resp.ms, tta: !!resp.tta };
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
  if (msg?.kind === "aid:get-settings") {
    // offscreen documents lack chrome.storage; serve settings from here
    chrome.storage.sync.get({ forceWasm: false }, sendResponse);
    return true;
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
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: "aid-check-image",
      title: "Sieve: check this image",
      contexts: ["image"],
    });
    chrome.contextMenus.create({
      id: "aid-report",
      title: "Sieve: report misclassification…",
      contexts: ["image"],
    });
  });
  if (details.reason === "install") {
    chrome.tabs.create({ url: chrome.runtime.getURL("src/setup.html") });
  }
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "aid-check-image" && tab?.id && info.srcUrl) {
    chrome.tabs.sendMessage(tab.id, { kind: "aid:check-image", srcUrl: info.srcUrl });
  }
  if (info.menuItemId === "aid-report" && info.srcUrl) {
    // Privacy by construction: reporting opens a public GitHub issue form the
    // user can read and edit BEFORE submitting. The extension itself sends
    // nothing anywhere.
    const cached = cache.get(info.srcUrl);
    const said = !cached ? "Not analyzed"
      : cached.score >= 0.65 ? "Flagged as AI (red badge, ≥65%)"
      : cached.score >= 0.5 ? "Unsure (amber badge, 50–65%)"
      : "Low score (gray badge, <50%)";
    const params = new URLSearchParams({
      template: "misclassification.yml",
      title: "[misclassification] ",
      "image-url": info.srcUrl,
      "sieve-said": said,
      "sieve-verdict": cached ? `score ${(cached.score * 100).toFixed(1)}%${cached.tta ? " (TTA)" : ""}` : "not analyzed / unknown",
      "model-version": chrome.runtime.getManifest().version,
    });
    chrome.tabs.create({
      url: `https://github.com/Phineas1500/sieve-ai-image-detector/issues/new?${params}`,
    });
  }
});
