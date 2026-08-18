const $ = (id) => document.getElementById(id);

const defaults = { enabled: true, threshold: 0.65, blur: true };

chrome.storage.sync.get(defaults, (s) => {
  $("enabled").checked = s.enabled;
  $("blur").checked = s.blur;
  $("threshold").value = s.threshold;
  $("thval").textContent = `${Math.round(s.threshold * 100)}%`;
});

$("enabled").addEventListener("change", (e) => chrome.storage.sync.set({ enabled: e.target.checked }));
$("blur").addEventListener("change", (e) => chrome.storage.sync.set({ blur: e.target.checked }));
$("threshold").addEventListener("input", (e) => {
  $("thval").textContent = `${Math.round(e.target.value * 100)}%`;
  chrome.storage.sync.set({ threshold: Number(e.target.value) });
});

$("setup").addEventListener("click", () => chrome.tabs.create({ url: chrome.runtime.getURL("src/setup.html") }));
$("options").addEventListener("click", () => chrome.runtime.openOptionsPage());

chrome.runtime.sendMessage({ kind: "aid:model-status" }, async (st) => {
  $("ep").textContent = st && st.ready ? `ready (${st.ep})` : "not set up";
  try {
    const manifest = await (await fetch(chrome.runtime.getURL("model_manifest.json"))).json();
    if (st && st.ready && manifest.version && st.version !== manifest.version) {
      $("ep").textContent = "model update available — open Model setup";
    }
  } catch {}
});

chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
  if (!tab?.id) return;
  chrome.tabs.sendMessage(tab.id, { kind: "aid:page-stats" }, (r) => {
    if (chrome.runtime.lastError || !r) return;
    $("analyzed").textContent = r.analyzed;
    $("flagged").textContent = r.flagged;
  });
});
