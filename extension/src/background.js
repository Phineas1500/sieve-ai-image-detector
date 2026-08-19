// Service worker: orchestrates analysis requests from content scripts.
// Inference itself runs in the offscreen document (WebGPU/WASM live there).

const OFFSCREEN_URL = "src/offscreen.html";

const cache = new Map(); // url -> {score, ms, tta}
const CACHE_MAX = 3000;
const inflight = new Map(); // url -> Promise

// The installed weights' version (OPFS meta): cache entries are only valid
// for the model that produced them.
async function installedModelVersion() {
  try {
    const root = await navigator.storage.getDirectory();
    const fh = await root.getFileHandle("model_meta.json");
    return JSON.parse(await (await fh.getFile()).text()).version || null;
  } catch {
    return null;
  }
}

// The in-memory cache dies with every MV3 service-worker restart; mirror it in
// session storage (survives restarts, cleared when the browser closes) so
// features like report-prefill still know recent verdicts. The mirror is
// stamped with the model version that produced it: restoring scores from an
// older model would resurrect stale verdicts across SW restarts (the bug
// behind immortal wrong badges — a cache.clear() alone doesn't reach the
// mirror).
async function restoreCacheMirror() {
  try {
    const [d, ver] = await Promise.all([
      chrome.storage.session.get(["aidCache", "aidCacheVer"]),
      installedModelVersion(),
    ]);
    if (d.aidCacheVer && ver && d.aidCacheVer !== ver) {
      await chrome.storage.session.remove(["aidCache", "aidCacheVer"]).catch(() => {});
      return;
    }
    for (const [k, v] of d.aidCache || []) if (!cache.has(k)) cache.set(k, v);
  } catch {}
}
restoreCacheMirror();
// e2e hooks
globalThis.__aidRestoreCacheMirror = restoreCacheMirror;
globalThis.__aidCacheDump = () => [...cache.entries()];
let persistTimer = null;
function schedulePersist() {
  clearTimeout(persistTimer);
  persistTimer = setTimeout(async () => {
    const ver = await installedModelVersion();
    chrome.storage.session.set({
      aidCache: [...cache.entries()].slice(-1000),
      aidCacheVer: ver,
    }).catch(() => {});
  }, 2000);
}

// Report-path lookup: exact URL, else same asset ignoring the query string
// (CDNs serve size variants like twitter's ?name=small/large).
function cacheLookup(url) {
  if (cache.has(url)) return cache.get(url);
  const base = url.split("?")[0];
  for (const [k, v] of cache) {
    if (k.split("?")[0] === base) return v;
  }
  return null;
}

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
  schedulePersist();
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
      // reach the session-storage mirror too, or the next SW restart
      // resurrects scores produced by the previous model
      await chrome.storage.session.remove(["aidCache", "aidCacheVer"]).catch(() => {});
      sendResponse({ ok: true });
    })();
    return true;
  }
  return false;
});

// The model in OPFS may predate the manifest bundled with this build:
// extension updates ship a new pinned model URL, but the download itself
// stays user-initiated (setup page), so detect the mismatch and prompt.
async function modelOutdated() {
  try {
    const bundled = await (await fetch(chrome.runtime.getURL("model_manifest.json"))).json();
    const root = await navigator.storage.getDirectory();
    const fh = await root.getFileHandle("model_meta.json");
    const meta = JSON.parse(await (await fh.getFile()).text());
    return !!bundled.version && meta.version !== bundled.version;
  } catch {
    return false; // nothing stored yet — the install flow covers that
  }
}
globalThis.__aidModelOutdated = modelOutdated; // e2e hook

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
  } else if (details.reason === "update" && (await modelOutdated())) {
    chrome.tabs.create({ url: chrome.runtime.getURL("src/setup.html?update=1") });
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
    const cached = cacheLookup(info.srcUrl);
    // GitHub issue forms only prefill input/textarea fields (not dropdowns),
    // and want %20 rather than "+" for spaces — encode manually.
    const band = !cached ? "not analyzed"
      : cached.score >= 0.65 ? "flagged as AI"
      : cached.score >= 0.5 ? "unsure"
      : "low score";
    const fields = {
      template: "misclassification.yml",
      title: "[misclassification] ",
      "image-url": info.srcUrl,
      "sieve-verdict": cached
        ? `${band} — score ${(cached.score * 100).toFixed(1)}%${cached.tta ? " (TTA)" : ""}`
        : "not analyzed",
      "model-version": chrome.runtime.getManifest().version,
    };
    const q = Object.entries(fields).map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join("&");
    chrome.tabs.create({
      url: `https://github.com/Phineas1500/sieve-ai-image-detector/issues/new?${q}`,
    });
  }
});
