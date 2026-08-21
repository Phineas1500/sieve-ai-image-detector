const $ = (id) => document.getElementById(id);

const defaults = { enabled: true, threshold: 0.65, blur: true, flaggedAction: "blur" };

chrome.storage.sync.get(defaults, (s) => {
  $("enabled").checked = s.enabled;
  $("blur").checked = s.flaggedAction === "blur" && s.blur !== false;
  $("threshold").value = s.threshold;
  $("thval").textContent = `${Math.round(s.threshold * 100)}%`;
});

$("enabled").addEventListener("change", (e) => chrome.storage.sync.set({ enabled: e.target.checked }));
$("blur").addEventListener("change", (e) => chrome.storage.sync.set({
  blur: e.target.checked,
  flaggedAction: e.target.checked ? "blur" : "badge",
}));
$("threshold").addEventListener("input", (e) => {
  $("thval").textContent = `${Math.round(e.target.value * 100)}%`;
  chrome.storage.sync.set({ threshold: Number(e.target.value) });
});

$("setup").addEventListener("click", () => chrome.tabs.create({ url: chrome.runtime.getURL("src/setup.html") }));
$("options").addEventListener("click", () => chrome.runtime.openOptionsPage());

chrome.runtime.sendMessage({ kind: "aid:model-status" }, async (st) => {
  $("ep").textContent = st && st.ready
    ? (st.selftest || "").startsWith("fallback")
      ? "ready (wasm — gpu numerics fallback)"
      : `ready (${st.ep})`
    : "not set up";
  try {
    const manifest = await (await fetch(chrome.runtime.getURL("model_manifest.json"))).json();
    if (st && st.ready && manifest.version && st.version !== manifest.version) {
      $("ep").textContent = "model update available — open Model setup";
    }
  } catch {}
});

// Which weights are actually installed (OPFS meta) — the ground truth that
// separates "stale model" reports from real regressions.
(async () => {
  try {
    const root = await navigator.storage.getDirectory();
    const fh = await root.getFileHandle("model_meta.json");
    const meta = JSON.parse(await (await fh.getFile()).text());
    $("modelver").textContent = meta.version || "unknown";
  } catch {
    $("modelver").textContent = "none";
  }
})();

chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
  if (!tab?.id) return;
  chrome.tabs.sendMessage(tab.id, { kind: "aid:page-stats" }, (r) => {
    if (chrome.runtime.lastError || !r) return;
    $("analyzed").textContent = r.analyzed;
    $("flagged").textContent = r.flagged;
  });
});
