// Regression test for badge cleanup on dynamic feeds (the "infinite scroll
// leak"): images removed from the DOM must have their badges reclaimed, and a
// re-attached image must get re-badged (self-healing via analysis cache).
//
// Usage: node leak-test.mjs [--model ../../dev_model/ft1_best_fp16.onnx]

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
const MODEL = resolve(here, args.model || "../../dev_model/ft1_best_fp16.onnx");
const IMAGES = resolve(here, "./sample_images");
const EXT = resolve(here, "../../extension");

const MIME = { ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp" };
const files = readdirSync(IMAGES).filter((f) => MIME[extname(f).toLowerCase()]);

const server = http.createServer((req, res) => {
  const url = decodeURIComponent(req.url.split("?")[0]);
  if (url === "/") {
    const body = files
      .map((f, i) => `<img id="im${i}" src="/img/${encodeURIComponent(f)}" style="display:block;max-width:480px;margin:16px">`)
      .join("\n");
    res.writeHead(200, { "content-type": "text/html" });
    res.end(`<!doctype html><meta charset="utf-8"><title>leak test</title>${body}`);
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
  headless: true,
  protocolTimeout: 300000,
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
    () => /Sieve is active|failed|Failed/i.test(document.getElementById("status").textContent),
    { timeout: 120000, polling: 500 }
  );

  const page = await browser.newPage();
  await page.goto(`http://localhost:${port}/`, { waitUntil: "networkidle2" });
  await page.evaluate(async () => {
    for (let y = 0; y < document.body.scrollHeight; y += 500) {
      window.scrollTo(0, y);
      await new Promise((r) => setTimeout(r, 80));
    }
    window.scrollTo(0, 0);
  });
  await page.waitForFunction(
    (n) => document.querySelectorAll("img[data-aid-score]").length >= n,
    { timeout: 300000, polling: 500 },
    files.length
  );

  const before = await page.evaluate(() => document.querySelectorAll(".aid-badge").length);
  check("all images badged", before === files.length, `${before}/${files.length}`);

  // simulate feed virtualization: remove 3 images, keep a handle to one
  await page.evaluate(() => {
    window.__removed = document.getElementById("im1");
    document.getElementById("im1").remove();
    document.getElementById("im2").remove();
    document.getElementById("im3").remove();
  });
  await new Promise((r) => setTimeout(r, 3500)); // > the 1.5s sweep interval

  const afterRemove = await page.evaluate(() => document.querySelectorAll(".aid-badge").length);
  check("badges reclaimed after image removal", afterRemove === files.length - 3, `${afterRemove} badges for ${files.length - 3} images`);

  // re-attach the SAME node: must re-badge via cached analysis
  await page.evaluate(() => document.body.appendChild(window.__removed));
  await page.evaluate(() => window.__removed.scrollIntoView({ block: "center" }));
  await page.waitForFunction(
    (n) => document.querySelectorAll(".aid-badge").length === n,
    { timeout: 30000, polling: 500 },
    files.length - 2
  ).catch(() => {});
  const afterReattach = await page.evaluate(() => document.querySelectorAll(".aid-badge").length);
  check("re-attached image re-badged (self-healing)", afterReattach === files.length - 2, `${afterReattach} badges for ${files.length - 2} images`);

  // no unbounded growth after repeated add/remove cycles
  await page.evaluate(async () => {
    for (let k = 0; k < 5; k++) {
      const img = document.createElement("img");
      img.src = document.querySelector("img").src + `?v=${k}`;
      img.style.cssText = "display:block;max-width:480px;margin:16px";
      document.body.appendChild(img);
      img.scrollIntoView();
      await new Promise((r) => setTimeout(r, 1200));
      img.remove();
    }
  });
  await new Promise((r) => setTimeout(r, 3500));
  const afterChurn = await page.evaluate(() => document.querySelectorAll(".aid-badge").length);
  check("no badge accumulation after churn", afterChurn === files.length - 2, `${afterChurn} badges remain`);

  // stacked-animation case (etched.com): several imgs layered in one slot,
  // all but one inside opacity:0 wrappers — only the revealed frame may show
  // a badge, and a frame swap moves the badge, never multiplies it
  await page.evaluate((src) => {
    const stack = document.createElement("div");
    stack.id = "stack";
    stack.style.cssText = "position:relative;width:320px;height:220px;margin:24px";
    for (let k = 0; k < 3; k++) {
      const wrap = document.createElement("div");
      wrap.id = `wrap${k}`;
      wrap.style.cssText = `position:absolute;inset:0;opacity:${k === 0 ? 1 : 0}`;
      const img = document.createElement("img");
      img.src = src;
      img.style.cssText = "width:300px;height:200px";
      wrap.appendChild(img);
      stack.appendChild(wrap);
    }
    document.body.appendChild(stack);
    stack.scrollIntoView({ block: "center" });
  }, await page.evaluate(() => document.querySelector("img").src));
  await page.waitForFunction(
    () => [...document.getElementById("stack").querySelectorAll("img")].every((im) => im.dataset.aidScore),
    { timeout: 60000, polling: 500 }
  );
  await new Promise((r) => setTimeout(r, 2500)); // let the sweep settle visibility
  const visibleInStack = () => page.evaluate(() => {
    const s = document.getElementById("stack").getBoundingClientRect();
    return [...document.querySelectorAll(".aid-badge")].filter((b) => {
      if (b.style.display === "none") return false;
      const r = b.getBoundingClientRect();
      return r.left >= s.left - 4 && r.left <= s.right && r.top >= s.top - 4 && r.top <= s.bottom;
    }).length;
  });
  check("hidden stack frames unbadged", (await visibleInStack()) === 1, `${await visibleInStack()} visible badges for 3 stacked imgs`);

  await page.evaluate(() => {
    document.getElementById("wrap0").style.opacity = "0";
    document.getElementById("wrap1").style.opacity = "1";
  });
  await new Promise((r) => setTimeout(r, 2500));
  check("frame swap moves badge, no pileup", (await visibleInStack()) === 1, `${await visibleInStack()} visible after swap`);
} finally {
  await browser.close();
  server.close();
}
process.exit(failed ? 1 : 0);
