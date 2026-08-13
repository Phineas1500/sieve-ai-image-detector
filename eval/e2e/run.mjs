// End-to-end harness: loads the extension into a real Chrome via Puppeteer,
// installs a model through the setup page, serves a page of test images, and
// collects per-image scores from the extension's DOM annotations.
//
// Usage:
//   node run.mjs --model ../../dev_model/commfor-model-384.onnx \
//                --images ./sample_images [--out scores.json] [--headless]
//
// Output: JSON array of {file, score, ms} — diffable against the Python
// pipeline's scores for the same files to verify preprocessing parity.

import http from "node:http";
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { extname, join, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(
  process.argv.slice(2).map((a, i, arr) =>
    a.startsWith("--") ? [a.slice(2), arr[i + 1]?.startsWith("--") || arr[i + 1] === undefined ? true : arr[i + 1]] : null
  ).filter(Boolean)
);

const MODEL = resolve(here, args.model || "../../dev_model/commfor-model-384.onnx");
const IMAGES = resolve(here, args.images || "./sample_images");
const OUT = resolve(here, args.out || "scores.json");
const EXT = resolve(here, "../../extension");

const MIME = { ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp" };
const files = readdirSync(IMAGES).filter((f) => MIME[extname(f).toLowerCase()]);
if (!files.length) throw new Error(`no images found in ${IMAGES}`);
console.log(`serving ${files.length} images from ${IMAGES}`);

const server = http.createServer((req, res) => {
  const url = decodeURIComponent(req.url.split("?")[0]);
  if (url === "/") {
    const body = files
      .map((f) => `<figure><img src="/img/${encodeURIComponent(f)}" data-file="${f}" loading="eager"><figcaption>${f}</figcaption></figure>`)
      .join("\n");
    res.writeHead(200, { "content-type": "text/html" });
    res.end(`<!doctype html><meta charset="utf-8"><title>Sieve e2e</title>
      <style>figure{margin:20px}img{max-width:480px;display:block}</style>${body}`);
  } else if (url.startsWith("/img/")) {
    try {
      const data = readFileSync(join(IMAGES, url.slice(5)));
      res.writeHead(200, { "content-type": MIME[extname(url).toLowerCase()] || "application/octet-stream" });
      res.end(data);
    } catch {
      res.writeHead(404).end();
    }
  } else res.writeHead(404).end();
});
await new Promise((r) => server.listen(0, r));
const port = server.address().port;

const browser = await puppeteer.launch({
  headless: !!args.headless, // new headless supports extensions; headful for WebGPU on macOS
  args: [
    `--disable-extensions-except=${EXT}`,
    `--load-extension=${EXT}`,
    "--no-first-run",
    "--enable-unsafe-webgpu",
  ],
});

// find the extension id via the service worker target
const swTarget = await browser.waitForTarget((t) => t.type() === "service_worker" && t.url().includes("chrome-extension://"), { timeout: 15000 });
const extId = new URL(swTarget.url()).host;
console.log(`extension loaded: ${extId}`);

// install the model through the setup page's local-file input
const setup = await browser.newPage();
await setup.goto(`chrome-extension://${extId}/src/setup.html`);
const fileInput = await setup.$("#localfile");
await fileInput.uploadFile(MODEL);
await setup.waitForFunction(
  () => /Sieve is active|inference init failed|Failed/.test(document.getElementById("status").textContent),
  { timeout: 120000 }
);
const setupStatus = await setup.$eval("#status", (el) => el.textContent);
console.log(`setup: ${setupStatus}`);
if (!/Sieve is active/.test(setupStatus)) {
  await browser.close();
  server.close();
  throw new Error("model setup failed");
}

// open the test page and let the extension analyze everything
const page = await browser.newPage();
await page.goto(`http://localhost:${port}/`, { waitUntil: "networkidle2" });

// scroll through to trigger IntersectionObserver on all images
await page.evaluate(async () => {
  for (let y = 0; y < document.body.scrollHeight; y += 600) {
    window.scrollTo(0, y);
    await new Promise((r) => setTimeout(r, 60));
  }
  window.scrollTo(0, 0);
});

const t0 = Date.now();
await page.waitForFunction(
  (n) => document.querySelectorAll("img[data-aid-score]").length >= n,
  { timeout: 600000, polling: 500 },
  files.length
);
const elapsed = Date.now() - t0;

const results = await page.$$eval("img[data-aid-score]", (imgs) =>
  imgs.map((img) => ({ file: img.dataset.file, score: Number(img.dataset.aidScore) }))
);
console.log(`analyzed ${results.length} images in ${(elapsed / 1000).toFixed(1)}s (${(elapsed / results.length).toFixed(0)}ms/img incl. queueing)`);
for (const r of results) console.log(`  ${r.score.toFixed(4)}  ${r.file}`);

writeFileSync(OUT, JSON.stringify(results, null, 2));
console.log(`wrote ${OUT}`);

await browser.close();
server.close();
