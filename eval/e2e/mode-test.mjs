// Regression test for v0.5.0 modes: manual scanning, hide (slopblocker), and
// flags-only badge display — asserted through the real extension.
import http from "node:http";
import { readdirSync, readFileSync } from "node:fs";
import { extname, join, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(
  process.argv.slice(2).map((a, i, arr) =>
    a.startsWith("--") ? [a.slice(2), arr[i + 1]?.startsWith("--") || arr[i + 1] === undefined ? true : arr[i + 1]] : null
  ).filter(Boolean)
);
const MODEL = resolve(here, args.model || "../../dev_model/ft2_best_fp16.onnx");
const IMAGES = resolve(here, "./sample_images");
const EXT = resolve(here, "../../extension");

const MIME = { ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp" };
const files = readdirSync(IMAGES).filter((f) => MIME[extname(f).toLowerCase()]);

const server = http.createServer((req, res) => {
  const url = decodeURIComponent(req.url.split("?")[0]);
  if (url === "/") {
    res.writeHead(200, { "content-type": "text/html" });
    res.end(`<!doctype html><meta charset="utf-8"><title>modes</title>` +
      files.map((f) => `<img src="/img/${encodeURIComponent(f)}" style="display:block;max-width:480px;margin:16px">`).join(""));
  } else if (url.startsWith("/img/")) {
    try {
      res.writeHead(200, { "content-type": MIME[extname(url).toLowerCase()] || "application/octet-stream" });
      res.end(readFileSync(join(IMAGES, url.slice(5))));
    } catch { res.writeHead(404).end(); }
  } else res.writeHead(404).end();
});
await new Promise((r) => server.listen(0, r));
const port = server.address().port;

const browser = await puppeteer.launch({
  headless: true,
  protocolTimeout: 600000,
  args: [`--disable-extensions-except=${EXT}`, `--load-extension=${EXT}`, "--no-first-run", "--enable-unsafe-webgpu"],
});

let failed = false;
const check = (name, cond, detail = "") => {
  console.log(`${cond ? "PASS" : "FAIL"}  ${name}${detail ? ` (${detail})` : ""}`);
  if (!cond) failed = true;
};

try {
  const swTarget = await browser.waitForTarget((t) => t.type() === "service_worker" && t.url().includes("chrome-extension://"), { timeout: 15000 });
  const extId = new URL(swTarget.url()).host;
  const setup = await browser.newPage();
  await setup.goto(`chrome-extension://${extId}/src/setup.html`);
  await (await setup.$("#localfile")).uploadFile(MODEL);
  await setup.waitForFunction(
    () => /Sieve is active|failed/i.test(document.getElementById("status").textContent),
    { timeout: 120000, polling: 500 }
  );
  const setSettings = (obj) => setup.evaluate((o) => new Promise((r) => chrome.storage.sync.set(o, r)), obj);

  const visit = async () => {
    const page = await browser.newPage();
    await page.goto(`http://localhost:${port}/`, { waitUntil: "networkidle2" });
    await page.evaluate(async () => {
      for (let y = 0; y < document.body.scrollHeight; y += 500) { window.scrollTo(0, y); await new Promise((r) => setTimeout(r, 60)); }
      window.scrollTo(0, 0);
    });
    return page;
  };

  // --- default auto mode: everything scored (baseline sanity)
  await setSettings({ scanMode: "auto", flaggedAction: "blur", badgeDisplay: "all" });
  let page = await visit();
  await page.waitForFunction((n) => document.querySelectorAll("img[data-aid-score]").length >= n,
    { timeout: 300000, polling: 500 }, files.length);
  check("auto mode scores all", true, `${files.length}/${files.length}`);
  await page.close();

  // --- manual mode: nothing scored automatically
  await setSettings({ scanMode: "manual" });
  page = await visit();
  await new Promise((r) => setTimeout(r, 6000));
  const manualScored = await page.evaluate(() => document.querySelectorAll("img[data-aid-score]").length);
  check("manual mode: no auto-scan", manualScored === 0, `${manualScored} scored`);
  await page.close();

  // --- hide mode: flagged images collapsed, others visible
  await setSettings({ scanMode: "auto", flaggedAction: "hide" });
  page = await visit();
  await page.waitForFunction((n) => document.querySelectorAll("img[data-aid-score]").length >= n,
    { timeout: 300000, polling: 500 }, files.length);
  const hideAudit = await page.evaluate(() => {
    const imgs = [...document.querySelectorAll("img[data-aid-score]")];
    return {
      flaggedHidden: imgs.filter((i) => +i.dataset.aidScore >= 0.65 && getComputedStyle(i).display === "none").length,
      flagged: imgs.filter((i) => +i.dataset.aidScore >= 0.65).length,
      cleanVisible: imgs.filter((i) => +i.dataset.aidScore < 0.65 && getComputedStyle(i).display !== "none").length,
      clean: imgs.filter((i) => +i.dataset.aidScore < 0.65).length,
    };
  });
  check("hide mode: flagged collapsed", hideAudit.flaggedHidden === hideAudit.flagged && hideAudit.flagged > 0,
    `${hideAudit.flaggedHidden}/${hideAudit.flagged} hidden`);
  check("hide mode: clean images untouched", hideAudit.cleanVisible === hideAudit.clean,
    `${hideAudit.cleanVisible}/${hideAudit.clean} visible`);
  await page.close();

  // --- flags-only badges
  await setSettings({ flaggedAction: "blur", badgeDisplay: "flags" });
  page = await visit();
  await page.waitForFunction((n) => document.querySelectorAll("img[data-aid-score]").length >= n,
    { timeout: 300000, polling: 500 }, files.length);
  const badgeAudit = await page.evaluate(() => ({
    quiet: document.querySelectorAll(".aid-badge.aid-quiet").length,
    flaggedBadges: [...document.querySelectorAll(".aid-badge")].filter((b) => b.classList.contains("aid-flagged")).length,
    flagged: [...document.querySelectorAll("img[data-aid-score]")].filter((i) => +i.dataset.aidScore >= 0.65).length,
    clean: [...document.querySelectorAll("img[data-aid-score]")].filter((i) => +i.dataset.aidScore < 0.65).length,
  }));
  check("flags-only: sub-threshold badges quieted", badgeAudit.quiet === badgeAudit.clean,
    `${badgeAudit.quiet}/${badgeAudit.clean} quiet`);
  check("flags-only: flagged badges still shown", badgeAudit.flaggedBadges === badgeAudit.flagged,
    `${badgeAudit.flaggedBadges}/${badgeAudit.flagged}`);
  await page.close();

  // --- forced WASM: the fallback path must work from the trimmed vendor set
  await setSettings({ forceWasm: true, badgeDisplay: "all" });
  await setup.evaluate(() => chrome.runtime.sendMessage({ kind: "aid:reload-model" }));
  await new Promise((r) => setTimeout(r, 3000));
  const st = await setup.evaluate(() => chrome.runtime.sendMessage({ kind: "aid:model-status" }));
  check("forced WASM: session initializes", !!st && st.ready === true, `ep=${st && st.ep}`);
  check("forced WASM: uses wasm EP", !!st && st.ep === "wasm", `ep=${st && st.ep}`);
  page = await visit();
  await page.waitForFunction((n) => document.querySelectorAll("img[data-aid-score]").length >= n,
    { timeout: 600000, polling: 500 }, files.length);
  const wasmScores = await page.evaluate(() =>
    [...document.querySelectorAll("img[data-aid-score]")].map((i) => +i.dataset.aidScore));
  check("forced WASM: all images scored", wasmScores.length === files.length, `${wasmScores.length}/${files.length}`);
  await setSettings({ forceWasm: false });
  await setup.evaluate(() => chrome.runtime.sendMessage({ kind: "aid:reload-model" }));
  await new Promise((r) => setTimeout(r, 3000));

  // --- WebGPU numerical self-test: healthy GPU passes; a session whose GPU
  // output disagrees with the WASM reference must auto-fall back to WASM
  // (spoof the stored reference to simulate a numerically broken GPU stack)
  const stOk = await setup.evaluate(() => chrome.runtime.sendMessage({ kind: "aid:model-status" }));
  check("self-test: healthy WebGPU passes", !!stOk && (stOk.ep !== "webgpu" || stOk.selftest === "ok"),
    `ep=${stOk && stOk.ep} selftest=${stOk && stOk.selftest}`);
  if (stOk && stOk.ep === "webgpu") {
    await setup.evaluate(async () => {
      const root = await navigator.storage.getDirectory();
      const fh = await root.getFileHandle("model_meta.json");
      const meta = JSON.parse(await (await fh.getFile()).text());
      meta.wasm_ref_logit = 999; // no real GPU output can be within 0.5 of this
      const w = await (await root.getFileHandle("model_meta.json", { create: true })).createWritable();
      await w.write(JSON.stringify(meta));
      await w.close();
    });
    await setup.evaluate(() => chrome.runtime.sendMessage({ kind: "aid:reload-model" }));
    await new Promise((r) => setTimeout(r, 4000));
    const stBad = await setup.evaluate(() => chrome.runtime.sendMessage({ kind: "aid:model-status" }));
    check("self-test: divergent GPU falls back to WASM",
      !!stBad && stBad.ep === "wasm" && (stBad.selftest || "").startsWith("fallback"),
      `ep=${stBad && stBad.ep} selftest=${stBad && stBad.selftest}`);
    // restore: drop the spoofed reference so the next session re-derives it
    await setup.evaluate(async () => {
      const root = await navigator.storage.getDirectory();
      const fh = await root.getFileHandle("model_meta.json");
      const meta = JSON.parse(await (await fh.getFile()).text());
      delete meta.wasm_ref_logit;
      const w = await (await root.getFileHandle("model_meta.json", { create: true })).createWritable();
      await w.write(JSON.stringify(meta));
      await w.close();
    });
    await setup.evaluate(() => chrome.runtime.sendMessage({ kind: "aid:reload-model" }));
  }
} finally {
  await browser.close().catch(() => {});
  server.close();
}
process.exit(failed ? 1 : 0);
