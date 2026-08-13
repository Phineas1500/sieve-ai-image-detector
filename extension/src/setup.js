// One-time model download into OPFS. The URL and hash are pinned at build time
// in model_manifest.json; nothing else is ever downloaded after this.

const statusEl = document.getElementById("status");
const progressEl = document.getElementById("progress");
const btn = document.getElementById("download");

async function loadManifest() {
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

async function sha256hex(buf) {
  const d = await crypto.subtle.digest("SHA-256", buf);
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function finishInstall(buf, manifest) {
  statusEl.textContent = "Storing model…";
  await opfsWrite("model.onnx", buf);
  await opfsWrite("model_meta.json", JSON.stringify(manifest));
  statusEl.textContent = "Verifying with a test inference…";
  const resp = await chrome.runtime.sendMessage({ kind: "aid:model-updated" });
  const st = await chrome.runtime.sendMessage({ kind: "aid:model-status" });
  if (st && st.ready) {
    statusEl.textContent = `Done. Inference backend: ${st.ep}. You can close this tab — Sieve is active.`;
  } else {
    statusEl.textContent = `Model stored, but inference init failed: ${st && st.error}`;
  }
}

btn.addEventListener("click", async () => {
  btn.disabled = true;
  try {
    const manifest = await loadManifest();
    if (!manifest.model_url) {
      statusEl.textContent = "No model URL pinned in this build (dev build). Use the local file option below.";
      btn.disabled = false;
      return;
    }
    statusEl.textContent = "Downloading model…";
    const resp = await fetch(manifest.model_url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const total = Number(resp.headers.get("content-length")) || 0;
    const reader = resp.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      if (total) progressEl.style.width = `${(received / total) * 100}%`;
    }
    const buf = await new Blob(chunks).arrayBuffer();
    statusEl.textContent = "Verifying checksum…";
    const hash = await sha256hex(buf);
    if (manifest.sha256 && hash !== manifest.sha256) {
      throw new Error(`checksum mismatch: got ${hash.slice(0, 12)}…`);
    }
    await finishInstall(buf, manifest);
  } catch (e) {
    statusEl.textContent = `Failed: ${e.message}`;
    btn.disabled = false;
  }
});

document.getElementById("localfile").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const manifest = await loadManifest();
  statusEl.textContent = `Loading local ${f.name}…`;
  await finishInstall(await f.arrayBuffer(), manifest);
});
