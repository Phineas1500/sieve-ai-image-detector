#!/usr/bin/env node
// Synthesizes authentic chat/UI screenshots: real HTML rendered by headless
// Chrome — genuine renderer pixels, so they are true members of the
// "human-made screenshot" class. Used as hard-real training data (ft3) and
// for FP measurement. Usage: node chat-screenshot-synth.mjs --n 40 --out dir
import { mkdirSync, readdirSync, readFileSync } from "node:fs";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

// Real photos for image-message bubbles (compound case: screenshot containing
// a photo). Only real COCO samples — never the fake_* files.
const here = dirname(fileURLToPath(import.meta.url));
const PHOTOS = readdirSync(join(here, "sample_images"))
  .filter((f) => /^0\d+\.jpg$/.test(f))
  .slice(0, 8)
  .map((f) => `data:image/jpeg;base64,${readFileSync(join(here, "sample_images", f)).toString("base64")}`);

const STICKERS = [
  `<svg width="90" height="90" viewBox="0 0 90 90"><circle cx="45" cy="45" r="40" fill="#ffd93b"/><circle cx="32" cy="38" r="5" fill="#333"/><circle cx="58" cy="38" r="5" fill="#333"/><path d="M28 55 Q45 72 62 55" stroke="#333" stroke-width="4" fill="none" stroke-linecap="round"/></svg>`,
  `<svg width="90" height="90" viewBox="0 0 90 90"><path d="M45 78 C20 60 8 42 14 28 C19 16 34 14 45 26 C56 14 71 16 76 28 C82 42 70 60 45 78Z" fill="#ff5d73"/></svg>`,
  `<svg width="100" height="70" viewBox="0 0 100 70"><rect x="5" y="10" width="90" height="50" rx="10" fill="#7bd88f"/><text x="50" y="42" font-size="20" font-family="sans-serif" text-anchor="middle" fill="#1a4726" font-weight="bold">nice!</text></svg>`,
];

const args = Object.fromEntries(process.argv.slice(2).map((a, i, arr) =>
  a.startsWith("--") ? [a.slice(2), arr[i + 1]] : null).filter(Boolean));
const N = Number(args.n || 40);
const OUT = resolve(args.out || "chat_shots");
mkdirSync(OUT, { recursive: true });

const NAMES = ["Alex", "Sam", "Jordan", "Maya", "Chris", "Priya", "Diego", "Lena", "Marcus", "Aisha", "Tom", "Nina"];
const MSGS = [
  "hey are you coming tonight?", "lol no way 😂", "did you see the game", "omg", "brb",
  "can you send me the file", "sounds good!", "I'm 5 min away", "what time works for you?",
  "haha exactly", "let's do thursday instead", "ok ok fine", "check your email",
  "that meeting could've been an email", "🔥🔥🔥", "no thoughts just vibes", "wait what",
  "I'll call you later", "did you eat yet", "the wifi here is terrible", "same tbh",
  "look at this", "we still on for tomorrow?", "yesss", "ugh mondays", "typing...",
  "she said yes!!", "landing at 8:40", "can't talk now, in class", "send pics",
  "my battery is at 2%", "this app keeps crashing", "just pushed the fix", "deploy went fine",
];
const rand = (a) => a[Math.floor(Math.random() * a.length)];
const rint = (lo, hi) => lo + Math.floor(Math.random() * (hi - lo + 1));

const THEMES = {
  imessage: { bg: "#000", header: "#1c1c1e", mine: "#0a84ff", theirs: "#26252a", text: "#fff", theirText: "#fff", font: "-apple-system, 'SF Pro Text', 'Helvetica Neue', sans-serif" },
  imessageLight: { bg: "#fff", header: "#f6f6f6", mine: "#0a84ff", theirs: "#e9e9eb", text: "#fff", theirText: "#000", font: "-apple-system, 'SF Pro Text', 'Helvetica Neue', sans-serif" },
  whatsapp: { bg: "#0b141a", header: "#202c33", mine: "#005c4b", theirs: "#202c33", text: "#e9edef", theirText: "#e9edef", font: "'Segoe UI', 'Helvetica Neue', sans-serif" },
  whatsappLight: { bg: "#efeae2", header: "#f0f2f5", mine: "#d9fdd3", theirs: "#fff", text: "#111b21", theirText: "#111b21", font: "'Segoe UI', 'Helvetica Neue', sans-serif" },
  slack: { bg: "#1a1d21", header: "#350d36", mine: "transparent", theirs: "transparent", text: "#d1d2d3", theirText: "#d1d2d3", font: "'Lato', 'Helvetica Neue', sans-serif" },
};
const DEVICES = [[390, 844], [430, 932], [360, 800], [393, 873], [412, 915]];

function chatHTML(themeName, t) {
  const other = rand(NAMES);
  const slack = themeName === "slack";
  const n = rint(6, 12);
  let bubbles = "";
  for (let i = 0; i < n; i++) {
    const mine = !slack && Math.random() < 0.5;
    const msg = rand(MSGS);
    const time = `${rint(1, 12)}:${String(rint(0, 59)).padStart(2, "0")} ${rand(["AM", "PM"])}`;
    const roll = Math.random();
    if (roll < 0.14 && PHOTOS.length) {
      // photo-message bubble (real photo inside the screenshot)
      bubbles += `<div style="display:flex;justify-content:${mine ? "flex-end" : "flex-start"};padding:2px 12px"><img src="${rand(PHOTOS)}" style="max-width:65%;border-radius:14px"></div>`;
    } else if (roll < 0.24) {
      // sticker/doodle (vector graphics inside the screenshot)
      bubbles += `<div style="display:flex;justify-content:${mine ? "flex-end" : "flex-start"};padding:2px 16px">${rand(STICKERS)}</div>`;
    } else if (slack) {
      bubbles += `<div style="display:flex;gap:8px;padding:6px 16px"><div style="width:36px;height:36px;border-radius:4px;background:hsl(${rint(0, 360)},50%,45%)"></div><div><b style="color:#fff;font-size:15px">${rand(NAMES)}</b> <span style="color:#9a9b9e;font-size:12px">${time}</span><div style="font-size:15px">${msg}</div></div></div>`;
    } else {
      bubbles += `<div style="display:flex;justify-content:${mine ? "flex-end" : "flex-start"};padding:2px 12px"><div style="max-width:70%;padding:8px 12px;border-radius:18px;background:${mine ? t.mine : t.theirs};color:${mine ? t.text : t.theirText};font-size:16px;line-height:1.3">${msg}</div></div>`;
    }
  }
  const battery = rint(15, 100);
  return `<!doctype html><html><body style="margin:0;background:${t.bg};font-family:${t.font};-webkit-font-smoothing:antialiased">
  <div style="display:flex;justify-content:space-between;align-items:center;padding:6px 18px;background:${t.header};color:${t.theirText};font-size:13px;font-weight:600">
    <span>${rint(1, 12)}:${String(rint(0, 59)).padStart(2, "0")}</span><span>${rand(["", "5G", "LTE", "WiFi"])} ▮ ${battery}%</span>
  </div>
  <div style="text-align:center;padding:10px;background:${t.header};color:${slack ? "#fff" : t.theirText};font-weight:700;font-size:16px">${slack ? "#" + rand(["general", "random", "eng", "design", "memes"]) : other}</div>
  <div style="padding-top:8px">${bubbles}</div>
  <div style="position:fixed;bottom:0;left:0;right:0;padding:10px 14px;background:${t.header}"><div style="border:1px solid #4444;border-radius:18px;padding:8px 14px;color:#888;font-size:15px">${slack ? "Message #general" : "iMessage"}</div></div>
  </body></html>`;
}

const browser = await puppeteer.launch({ headless: true });
const page = await browser.newPage();
const themeNames = Object.keys(THEMES);
for (let i = 0; i < N; i++) {
  const themeName = themeNames[i % themeNames.length];
  const [w, h] = rand(DEVICES);
  await page.setViewport({ width: w, height: h, deviceScaleFactor: rand([2, 3]) });
  await page.setContent(chatHTML(themeName, THEMES[themeName]), { waitUntil: "load" });
  const type = Math.random() < 0.5 ? "png" : "jpeg";
  await page.screenshot({ path: `${OUT}/chat_${String(i).padStart(3, "0")}_${themeName}.${type === "png" ? "png" : "jpg"}`, type, ...(type === "jpeg" ? { quality: rint(70, 92) } : {}) });
}
await browser.close();
console.log(`wrote ${N} chat screenshots -> ${OUT}`);
