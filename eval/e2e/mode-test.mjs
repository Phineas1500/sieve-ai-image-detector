// Regression test for v0.5.0 modes: manual scanning, hide (slopblocker), and
// flags-only badge display — asserted through the real extension.
import http from "node:http";
import { deflateSync } from "node:zlib";
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

// Degenerate inputs (issue-#41-adjacent): solid fills and pure noise must get
// NO verdict — the model scores them confident nonsense. Generated here as
// minimal PNGs so the sample corpus stays untouched.
const CRC_TABLE = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
const crc32 = (buf) => {
  let c = 0xffffffff;
  for (const b of buf) c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
};
const pngChunk = (type, data) => {
  const out = Buffer.alloc(12 + data.length);
  out.writeUInt32BE(data.length, 0);
  out.write(type, 4, "latin1");
  data.copy(out, 8);
  out.writeUInt32BE(crc32(Buffer.concat([Buffer.from(type, "latin1"), data])), 8 + data.length);
  return out;
};
function makePng(size, pixel) {
  const raw = Buffer.alloc(size * (1 + size * 3));
  for (let y = 0; y < size; y++) {
    const row = y * (1 + size * 3) + 1;
    for (let x = 0; x < size; x++) {
      const [r, g, b] = pixel(x, y);
      raw[row + x * 3] = r; raw[row + x * 3 + 1] = g; raw[row + x * 3 + 2] = b;
    }
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0); ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8; ihdr[9] = 2;
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngChunk("IHDR", ihdr), pngChunk("IDAT", deflateSync(raw)), pngChunk("IEND", Buffer.alloc(0)),
  ]);
}
let noiseSeed = 0x9e3779b9;
const xorshift = () => {
  noiseSeed ^= noiseSeed << 13; noiseSeed ^= noiseSeed >>> 17; noiseSeed ^= noiseSeed << 5;
  return (noiseSeed >>> 0) & 0xff;
};
const DEGEN = {
  "black.png": makePng(256, () => [0, 0, 0]),
  "noise.png": makePng(256, () => [xorshift(), xorshift(), xorshift()]),
};

const server = http.createServer((req, res) => {
  const url = decodeURIComponent(req.url.split("?")[0]);
  if (url === "/") {
    const proxy = new URL(req.url, "http://localhost").searchParams.get("proxy") === "1";
    res.writeHead(200, { "content-type": "text/html" });
    const degraded = new URL(req.url, "http://localhost").searchParams.get("degraded") === "1"
      ? ["deepfry_fake.jpg", "up64_real.jpg"].map((f) =>
          `<img class="degraded" src="/dg/${f}" style="display:block;max-width:480px;margin:16px">`).join("")
      : "";
    const degen = new URL(req.url, "http://localhost").searchParams.get("degen") === "1"
      ? Object.keys(DEGEN).map((f) =>
          `<img class="degen" src="/degen/${f}" style="display:block;width:256px;height:256px;margin:16px">`).join("")
      : "";
    const images = degraded + degen + files.map((f) => proxy
      ? `<div class="aid-proxy" style="position:relative;width:480px;height:360px;margin:16px;overflow:hidden">` +
        `<div data-aid-background style="position:absolute;inset:0;background:center/cover no-repeat url('/img/${encodeURIComponent(f)}')"></div>` +
        `<img src="/img/${encodeURIComponent(f)}" style="position:absolute;inset:0;width:100%;height:100%;opacity:0">` +
        `</div>`
      : `<img src="/img/${encodeURIComponent(f)}" style="display:block;max-width:480px;margin:16px">`).join("");
    res.end(`<!doctype html><meta charset="utf-8"><title>modes</title>${images}`);
  } else if (url.startsWith("/dg/")) {
    try {
      res.writeHead(200, { "content-type": "image/jpeg" });
      res.end(readFileSync(join(here, "degraded_samples", url.slice(4))));
    } catch { res.writeHead(404).end(); }
  } else if (url.startsWith("/degen/")) {
    const buf = DEGEN[url.slice(7)];
    if (buf) { res.writeHead(200, { "content-type": "image/png" }); res.end(buf); }
    else res.writeHead(404).end();
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

  const visit = async (proxy = false, extra = "") => {
    const page = await browser.newPage();
    await page.goto(`http://localhost:${port}/?proxy=${proxy ? "1" : "0"}${extra}`, { waitUntil: "networkidle2" });
    await page.evaluate(async () => {
      for (let y = 0; y < document.body.scrollHeight; y += 500) { window.scrollTo(0, y); await new Promise((r) => setTimeout(r, 60)); }
      window.scrollTo(0, 0);
    });
    return page;
  };

  // --- default auto mode: everything scored (baseline sanity)
  await setSettings({ scanMode: "auto", blur: true, flaggedAction: "blur", badgeDisplay: "all" });
  let page = await visit();
  await page.waitForFunction((n) => document.querySelectorAll("img[data-aid-score]").length >= n,
    { timeout: 300000, polling: 500 }, files.length);
  check("auto mode scores all", true, `${files.length}/${files.length}`);
  const blurAudit = await page.evaluate(() => {
    const flagged = [...document.querySelectorAll("img[data-aid-score]")].filter((i) => +i.dataset.aidScore >= 0.65);
    return {
      flagged: flagged.length,
      blurred: flagged.filter((i) => i.classList.contains("aid-blurred") && getComputedStyle(i).filter.includes("blur(")).length,
    };
  });
  check("blur mode: flagged images are blurred", blurAudit.blurred === blurAudit.flagged && blurAudit.flagged > 0,
    `${blurAudit.blurred}/${blurAudit.flagged} blurred`);

  // --- badge click toggles the blur on a single image (issue #44: the
  // reveal boolean shipped inverted in v0.10.1, making clicks a no-op)
  const clickBadge = () => page.evaluate(async () => {
    const img = [...document.querySelectorAll("img[data-aid-score]")].find((i) => +i.dataset.aidScore >= 0.65);
    img.scrollIntoView({ block: "center" });
    await new Promise((r) => setTimeout(r, 400)); // let the scroll handler reposition badges
    const r = img.getBoundingClientRect();
    const badge = [...document.querySelectorAll(".aid-badge")].find((b) =>
      b.style.display !== "none" &&
      Math.abs(parseFloat(b.style.top) - (scrollY + r.top + 6)) < 2 &&
      Math.abs(parseFloat(b.style.left) - (scrollX + r.left + 6)) < 2);
    if (!badge) return "no-badge";
    badge.click();
    return img.classList.contains("aid-blurred");
  });
  const afterFirstClick = await clickBadge();
  const afterSecondClick = await clickBadge();
  check("badge click reveals a blurred image", afterFirstClick === false, `blurred=${afterFirstClick}`);
  check("second badge click re-blurs it", afterSecondClick === true, `blurred=${afterSecondClick}`);

  await setSettings({ blur: false, flaggedAction: "badge" });
  await page.waitForFunction(() => [...document.querySelectorAll("img[data-aid-score]")]
    .filter((i) => +i.dataset.aidScore >= 0.65)
    .every((i) => !i.classList.contains("aid-blurred")), { timeout: 10000, polling: 100 });
  const unblurAudit = await page.evaluate(() => {
    const flagged = [...document.querySelectorAll("img[data-aid-score]")].filter((i) => +i.dataset.aidScore >= 0.65);
    return { flagged: flagged.length, blurred: flagged.filter((i) => i.classList.contains("aid-blurred")).length };
  });
  check("blur toggle off: flagged images are revealed", unblurAudit.blurred === 0,
    `${unblurAudit.blurred}/${unblurAudit.flagged} still blurred`);
  await page.close();

  // --- background-image twin: X/Twitter-style renderers must be blurred too
  await setSettings({ blur: true, flaggedAction: "blur" });
  page = await visit(true);
  await page.waitForFunction((n) => document.querySelectorAll("img[data-aid-score]").length >= n,
    { timeout: 300000, polling: 500 }, files.length);
  const proxyAudit = await page.evaluate(() => {
    const flagged = [...document.querySelectorAll(".aid-proxy img[data-aid-score]")]
      .filter((i) => +i.dataset.aidScore >= 0.65);
    return {
      flagged: flagged.length,
      blurred: flagged.filter((img) => {
        const bg = img.parentElement.querySelector("[data-aid-background]");
        return bg?.classList.contains("aid-blurred") && getComputedStyle(bg).filter.includes("blur(");
      }).length,
    };
  });
  check("background-image twins: flagged visuals are blurred", proxyAudit.blurred === proxyAudit.flagged && proxyAudit.flagged > 0,
    `${proxyAudit.blurred}/${proxyAudit.flagged} blurred`);
  await page.close();

  // --- degenerate inputs: solid fills and pure noise get NO verdict
  page = await visit(false, "&degen=1");
  await page.waitForFunction((n) => document.querySelectorAll("img[data-aid-score]").length >= n,
    { timeout: 300000, polling: 500 }, files.length);
  await new Promise((r) => setTimeout(r, 4000)); // give degen images time to (wrongly) score
  const degenAudit = await page.evaluate(() => ({
    total: document.querySelectorAll("img.degen").length,
    scored: document.querySelectorAll("img.degen[data-aid-score]").length,
    na: document.querySelectorAll("img.degen[data-aid-na]").length,
    chips: [...document.querySelectorAll(".aid-badge.aid-na")].filter((b) => b.textContent === "not analysed").length,
  }));
  check("degenerate inputs: no verdict badged", degenAudit.total === 2 && degenAudit.scored === 0,
    `${degenAudit.scored}/${degenAudit.total} scored`);
  check("degenerate inputs: 'not analysed' chip shown (#49)", degenAudit.na === 2 && degenAudit.chips >= 2,
    `${degenAudit.na} marked, ${degenAudit.chips} chips`);
  await page.close();

  // --- degraded delivery (deep-fried / upscaled-thumbnail input) is never
  // flagged red: shown as unsure with the degraded marker (#41/#42)
  page = await visit(false, "&degraded=1");
  await page.waitForFunction(() => document.querySelectorAll("img.degraded[data-aid-score]").length >= 2,
    { timeout: 120000, polling: 500 });
  const dgAudit = await page.evaluate(() => [...document.querySelectorAll("img.degraded")].map((i) => ({
    src: i.getAttribute("src").split("/").pop(), score: +i.dataset.aidScore, degraded: i.dataset.aidDegraded,
  })));
  const flaggedDegraded = await page.evaluate(() => [...document.querySelectorAll("img.degraded")].map((i) => {
    const r = i.getBoundingClientRect();
    const b = [...document.querySelectorAll(".aid-badge")].find((b) =>
      Math.abs(parseFloat(b.style.top) - (scrollY + r.top + 6)) < 2 && Math.abs(parseFloat(b.style.left) - (scrollX + r.left + 6)) < 2);
    return b ? { text: b.textContent, flagged: b.classList.contains("aid-flagged"), degraded: b.classList.contains("aid-degraded") } : null;
  }));
  check("degraded inputs: detected as degraded", dgAudit.every((d) => d.degraded === "true"),
    dgAudit.map((d) => `${d.src}:${d.score.toFixed(2)}/${d.degraded}`).join(" "));
  check("degraded inputs: never flagged red", flaggedDegraded.every((b) => b && !b.flagged),
    flaggedDegraded.map((b) => b && b.text).join(" | "));
  await page.close();

  // --- manual mode: nothing scored automatically
  await setSettings({ scanMode: "manual", blur: true, flaggedAction: "blur" });
  page = await visit();
  await new Promise((r) => setTimeout(r, 6000));
  const manualScored = await page.evaluate(() => document.querySelectorAll("img[data-aid-score]").length);
  check("manual mode: no auto-scan", manualScored === 0, `${manualScored} scored`);
  await page.close();

  // --- hide mode: flagged images collapsed, others visible
  await setSettings({ scanMode: "auto", blur: false, flaggedAction: "hide" });
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
  await setSettings({ blur: true, flaggedAction: "blur", badgeDisplay: "flags" });
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
