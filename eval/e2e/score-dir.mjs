// Score every image in a directory through the REAL extension (decode,
// preprocess, WebGPU/WASM inference, selective TTA) and dump {file: {score,
// tta}} as JSON. This is the in-extension verdict — what a user sees — as
// opposed to tools/pil_score.py, which is the single-view reference pipeline.
//
//   node score-dir.mjs --model ../../dev_model/ft5s_best_fp16.onnx --bias 0.42 --dir ../../dev_model/ft5_results/reported --out scores_ft5s.json
//   add --size 224 --resize 256 for a 224px backbone (patches the manifest for the run, restored on exit)
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
const MODEL = resolve(here, args.model);
const BIAS = Number(args.bias);
const DIR = resolve(here, args.dir);
const OUT = resolve(here, args.out || "scores.json");
const EXT = resolve(here, "../../extension");
const MANIFEST = join(EXT, "model_manifest.json");

const MIME = { ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".avif": "image/avif", ".gif": "image/gif" };
const files = readdirSync(DIR).filter((f) => MIME[extname(f).toLowerCase()] && !f.startsWith("view_"));

const server = http.createServer((req, res) => {
  const url = decodeURIComponent(req.url.split("?")[0]);
  if (url === "/") {
    res.writeHead(200, { "content-type": "text/html" });
    res.end(`<!doctype html><meta charset="utf-8"><title>score-dir</title>` +
      files.map((f) => `<img src="/img/${encodeURIComponent(f)}" data-file="${f}" style="display:block;max-width:480px;margin:16px">`).join(""));
  } else if (url.startsWith("/img/")) {
    try {
      res.writeHead(200, { "content-type": MIME[extname(url).toLowerCase()] });
      res.end(readFileSync(join(DIR, url.slice(5))));
    } catch { res.writeHead(404).end(); }
  } else res.writeHead(404).end();
});
await new Promise((r) => server.listen(0, r));
const port = server.address().port;

const origManifest = readFileSync(MANIFEST, "utf8");
const patched = JSON.parse(origManifest);
patched.calibration.bias = BIAS;
if (args.size) { patched.input_size = Number(args.size); patched.resize_shorter_side = Number(args.resize || Math.round(Number(args.size) * 440 / 384)); }
writeFileSync(MANIFEST, JSON.stringify(patched, null, 2));

const browser = await puppeteer.launch({
  headless: true, protocolTimeout: 600000,
  args: [`--disable-extensions-except=${EXT}`, `--load-extension=${EXT}`, "--no-first-run", "--enable-unsafe-webgpu"],
});
const results = {};
try {
  const swTarget = await browser.waitForTarget((t) => t.type() === "service_worker" && t.url().includes("chrome-extension://"), { timeout: 15000 });
  const extId = new URL(swTarget.url()).host;
  const setup = await browser.newPage();
  await setup.goto(`chrome-extension://${extId}/src/setup.html`);
  await (await setup.$("#localfile")).uploadFile(MODEL);
  await setup.waitForFunction(() => /Sieve is active|failed/i.test(document.getElementById("status").textContent), { timeout: 120000, polling: 500 });
  const page = await browser.newPage();
  await page.goto(`http://localhost:${port}/`, { waitUntil: "networkidle2" });
  await page.evaluate(async () => {
    for (let y = 0; y < document.body.scrollHeight; y += 500) { window.scrollTo(0, y); await new Promise((r) => setTimeout(r, 80)); }
    window.scrollTo(0, 0);
  });
  try {
    await page.waitForFunction((n) => document.querySelectorAll("img[data-aid-score]").length >= n, { timeout: 240000, polling: 500 }, files.length);
  } catch { console.error("timeout: not every image scored"); }
  await new Promise((r) => setTimeout(r, 1500));
  Object.assign(results, await page.evaluate(() => Object.fromEntries(
    [...document.querySelectorAll("img[data-file]")].map((i) => [i.dataset.file,
      i.dataset.aidScore === undefined ? null : { score: +i.dataset.aidScore, tta: i.dataset.aidTta === "true", ms: i.dataset.aidMs === undefined ? null : +i.dataset.aidMs }]))));
} finally {
  writeFileSync(MANIFEST, origManifest);
  await browser.close();
  server.close();
}
writeFileSync(OUT, JSON.stringify({ model: args.model, bias: BIAS, results }, null, 1));
const scored = Object.values(results).filter(Boolean);
const ms = scored.map((r) => r.ms).filter((m) => typeof m === "number").sort((a, b) => a - b);
const med = ms.length ? ms[Math.floor(ms.length / 2)] : null;
console.log(`${scored.length}/${files.length} scored, tta fired on ${scored.filter((r) => r.tta).length}, median inference ${med === null ? "n/a" : med + "ms"} -> ${OUT}`);
