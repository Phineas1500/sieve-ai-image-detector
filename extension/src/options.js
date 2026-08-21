const $ = (id) => document.getElementById(id);
const defaults = { enabled: true, threshold: 0.65, blur: true, minSize: 96, minDisplaySize: 100,
  scanMode: "auto", flaggedAction: "blur", badgeDisplay: "all" };

chrome.storage.sync.get(defaults, (s) => {
  $("enabled").checked = s.enabled;
  const action = ["badge", "blur", "hide"].includes(s.flaggedAction)
    ? s.flaggedAction
    : s.blur ? "blur" : "badge";
  $("scanMode").value = s.scanMode;
  $("flaggedAction").value = action;
  $("badgeDisplay").value = s.badgeDisplay;
  $("threshold").value = s.threshold;
  $("thval").textContent = `${Math.round(s.threshold * 100)}%`;
  $("minSize").value = s.minSize;
  $("minDisplaySize").value = s.minDisplaySize;
});

$("enabled").addEventListener("change", (e) => chrome.storage.sync.set({ enabled: e.target.checked }));
$("threshold").addEventListener("input", (e) => {
  $("thval").textContent = `${Math.round(e.target.value * 100)}%`;
  chrome.storage.sync.set({ threshold: Number(e.target.value) });
});
$("minSize").addEventListener("change", (e) => chrome.storage.sync.set({ minSize: Number(e.target.value) }));
$("minDisplaySize").addEventListener("change", (e) => chrome.storage.sync.set({ minDisplaySize: Number(e.target.value) }));
$("scanMode").addEventListener("change", (e) => chrome.storage.sync.set({ scanMode: e.target.value }));
$("flaggedAction").addEventListener("change", (e) => chrome.storage.sync.set({
  flaggedAction: e.target.value,
  blur: e.target.value === "blur",
}));
$("badgeDisplay").addEventListener("change", (e) => chrome.storage.sync.set({ badgeDisplay: e.target.value }));
