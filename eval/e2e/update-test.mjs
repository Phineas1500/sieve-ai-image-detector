#!/usr/bin/env node
// Regression test for the stale-model-after-update bug (v0.6.1): an extension
// update ships a new pinned model manifest, but the model in OPFS is only
// replaced when the user downloads it — so the service worker must detect the
// version mismatch (modelOutdated) to prompt, and the setup page must announce
// the pending update. Usage: node update-test.mjs [--headless]
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

const here = dirname(fileURLToPath(import.meta.url));
const EXT = resolve(here, "../../extension");
const args = Object.fromEntries(process.argv.slice(2).map((a) => [a.replace(/^--/, ""), true]));

const browser = await puppeteer.launch({
  headless: !!args.headless,
  protocolTimeout: 120000,
  args: [`--disable-extensions-except=${EXT}`, `--load-extension=${EXT}`, "--no-first-run"],
});

const swTarget = await browser.waitForTarget(
  (t) => t.type() === "service_worker" && t.url().includes("chrome-extension://"),
  { timeout: 15000 });
const extId = new URL(swTarget.url()).host;
const sw = await swTarget.worker();

let failures = 0;
const check = (name, cond, extra = "") => {
  console.log(`${cond ? "PASS" : "FAIL"}  ${name}${extra ? ` (${extra})` : ""}`);
  if (!cond) failures++;
};

const outdated = () => sw.evaluate(() => globalThis.__aidModelOutdated());

// 1. fresh profile, nothing stored — not "outdated" (the install flow owns this)
check("no stored meta -> not outdated", (await outdated()) === false);

// 2. stored meta older than the bundled manifest -> outdated (written from a
//    page context, read from the SW: proves both share one OPFS bucket)
const page = await browser.newPage();
await page.goto(`chrome-extension://${extId}/src/setup.html`);
const writeMeta = (version) => page.evaluate(async (v) => {
  const root = await navigator.storage.getDirectory();
  const fh = await root.getFileHandle("model_meta.json", { create: true });
  const w = await fh.createWritable();
  await w.write(JSON.stringify({ version: v }));
  await w.close();
}, version);
await writeMeta("ft0-test-old");
check("stale meta -> outdated", (await outdated()) === true);

// 3. setup page announces the pending update
await page.goto(`chrome-extension://${extId}/src/setup.html?update=1`);
await page.waitForFunction(
  () => /Model update ready/.test(document.getElementById("status").textContent),
  { timeout: 5000 }).catch(() => {});
const status = await page.$eval("#status", (el) => el.textContent);
check("setup page announces update", /Model update ready: ft0-test-old →/.test(status), status);

// 4. meta matching the bundled manifest -> not outdated
const bundledVersion = await page.evaluate(async () =>
  (await (await fetch(chrome.runtime.getURL("model_manifest.json"))).json()).version);
await writeMeta(bundledVersion);
check("current meta -> not outdated", (await outdated()) === false);

// 5. session-storage cache mirror stamped with an older model's version must
//    NOT be restored (the immortal-stale-badge bug): scores survive SW
//    restarts via the mirror, so a model change has to invalidate it
const seed = (ver) => page.evaluate((v) => chrome.storage.session.set({
  aidCache: [["http://test/poisoned.jpg", { score: 0.955, ms: 1, tta: false }]],
  aidCacheVer: v,
}), ver);
await seed("ft-OLD-model");
await sw.evaluate(() => globalThis.__aidRestoreCacheMirror());
const afterStale = await sw.evaluate(() => globalThis.__aidCacheDump().length);
check("stale-version cache mirror dropped", afterStale === 0, `${afterStale} entries restored`);
const mirrorGone = await page.evaluate(async () =>
  (await chrome.storage.session.get("aidCache")).aidCache === undefined);
check("stale mirror purged from session storage", mirrorGone);

// 6. matching-version mirror restores normally
await seed(bundledVersion);
await sw.evaluate(() => globalThis.__aidRestoreCacheMirror());
const afterMatch = await sw.evaluate(() => globalThis.__aidCacheDump().length);
check("current-version cache mirror restored", afterMatch === 1, `${afterMatch} entries`);

await browser.close();
console.log(failures ? `${failures} FAILURES` : "ALL CHECKS PASSED");
process.exit(failures ? 1 : 0);
