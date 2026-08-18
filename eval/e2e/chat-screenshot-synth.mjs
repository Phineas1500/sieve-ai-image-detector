#!/usr/bin/env node
// Synthesizes authentic chat/UI screenshots: real HTML rendered by headless
// Chrome — genuine renderer pixels, so they are true members of the
// "human-made screenshot" class. Used as hard-real training data (ft3) and
// for FP measurement. Usage: node chat-screenshot-synth.mjs --n 40 --out dir
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import puppeteer from "puppeteer";

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
    if (slack) {
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
