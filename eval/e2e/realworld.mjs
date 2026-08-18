// Real-world A/B harness: loads the extension with a given model + calibration
// into a throwaway profile, visits real sites (the historical trouble spots),
// and records per-site score distributions from the extension's own verdicts.
//
// Usage:
//   node realworld.mjs --model ../../dev_model/ft1_best_fp16.onnx --bias 0.88 --out rw_ft1.json
//   node realworld.mjs --model ../../dev_model/ft2_best_fp16.onnx --bias 1.04 --out rw_ft2.json
import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(
  process.argv.slice(2).map((a, i, arr) =>
    a.startsWith("--") ? [a.slice(2), arr[i + 1]?.startsWith("--") || arr[i + 1] === undefined ? true : arr[i + 1]] : null
  ).filter(Boolean)
);
const MODEL = resolve(here, args.model);
const BIAS = Number(args.bias);
const OUT = resolve(here, args.out || "rw.json");
const EXT = resolve(here, "../../extension");
const MANIFEST = join(EXT, "model_manifest.json");

const SITES = [
  { name: "gimg-mj7-art", url: "https://www.google.com/search?q=midjourney+v7+art&udm=2", kind: "ai-lean" },
  { name: "gimg-ai-art", url: "https://www.google.com/search?q=ai+generated+art&udm=2", kind: "ai-lean" },
  { name: "gimg-recraft", url: "https://www.google.com/search?q=recraft+v3+design+examples&udm=2", kind: "ai-lean" },
  { name: "gimg-posters", url: "https://www.google.com/search?q=vintage+movie+poster+scan&udm=2", kind: "real-lean" },
  { name: "gimg-nfl", url: "https://www.google.com/search?q=nfl&udm=2", kind: "real-lean" },
  { name: "reddit-mj", url: "https://www.reddit.com/r/midjourney/", kind: "ai-lean" },
  { name: "reddit-pics", url: "https://www.reddit.com/r/pics/", kind: "real-lean" },
  { name: "wiki-potd", url: "https://commons.wikimedia.org/wiki/Commons:Picture_of_the_day", kind: "real-lean" },
];

// patch manifest calibration for this run; restore on exit
const origManifest = readFileSync(MANIFEST, "utf8");
const patched = JSON.parse(origManifest);
patched.calibration.bias = BIAS;
writeFileSync(MANIFEST, JSON.stringify(patched, null, 2));

const browser = await puppeteer.launch({
  headless: true,
  protocolTimeout: 600000,
  args: [`--disable-extensions-except=${EXT}`, `--load-extension=${EXT}`, "--no-first-run", "--enable-unsafe-webgpu", "--lang=en-US"],
});

const results = {};
try {
  const swTarget = await browser.waitForTarget((t) => t.type() === "service_worker" && t.url().includes("chrome-extension://"), { timeout: 15000 });
  const extId = new URL(swTarget.url()).host;
  const setup = await browser.newPage();
  await setup.goto(`chrome-extension://${extId}/src/setup.html`);
  await (await setup.$("#localfile")).uploadFile(MODEL);
  await setup.waitForFunction(
    () => /Sieve is active|failed|Failed/i.test(document.getElementById("status").textContent),
    { timeout: 120000, polling: 500 }
  );
  console.log("model installed:", await setup.$eval("#status", (el) => el.textContent));

  for (const site of SITES) {
    const page = await browser.newPage();
    try {
      await page.goto(site.url, { waitUntil: "domcontentloaded", timeout: 45000 });
      await new Promise((r) => setTimeout(r, 5000));
      await page.evaluate(async () => {
        for (let i = 0; i < 8; i++) {
          window.scrollBy(0, 700);
          await new Promise((r) => setTimeout(r, 900));
        }
      });
      await new Promise((r) => setTimeout(r, 7000));
      const data = await page.evaluate(() => {
        const seen = new Set();
        const out = [];
        for (const img of document.querySelectorAll("img[data-aid-score]")) {
          const r = img.getBoundingClientRect();
          if (seen.has(img.currentSrc)) continue;
          seen.add(img.currentSrc);
          out.push({
            s: +(+img.dataset.aidScore).toFixed(3),
            tta: img.dataset.aidTta === "true",
            nat: Math.min(img.naturalWidth, img.naturalHeight),
          });
        }
        return out;
      });
      const scores = data.map((d) => d.s);
      results[site.name] = {
        kind: site.kind,
        n: data.length,
        flagged: scores.filter((s) => s >= 0.65).length,
        unsure: scores.filter((s) => s >= 0.5 && s < 0.65).length,
        ttaFired: data.filter((d) => d.tta).length,
        scores: scores.sort((a, b) => b - a),
      };
      console.log(`${site.name}: n=${data.length} flagged=${results[site.name].flagged} unsure=${results[site.name].unsure} tta=${results[site.name].ttaFired}`);
    } catch (e) {
      results[site.name] = { kind: site.kind, error: String(e.message).slice(0, 80) };
      console.log(`${site.name}: ERROR ${e.message.slice(0, 80)}`);
    } finally {
      await page.close().catch(() => {});
    }
  }
} finally {
  writeFileSync(MANIFEST, origManifest);
  await browser.close().catch(() => {});
}
writeFileSync(OUT, JSON.stringify(results, null, 2));
console.log(`wrote ${OUT} (manifest restored)`);
