// Unit test for structural metadata forensics (issue #43): markers must only
// be honored inside their valid containers — a raw byte-searched payload
// appended to a file must NOT flip the verdict. Pure node, no browser.
import { deflateSync } from "node:zlib";
import { readFileSync, readdirSync } from "node:fs";
import { join, resolve, dirname, extname } from "node:path";
import { fileURLToPath } from "node:url";
import { sniffMetadata } from "../../extension/src/forensics.js";

const here = dirname(fileURLToPath(import.meta.url));

let failed = false;
const check = (name, cond, detail = "") => {
  console.log(`${cond ? "PASS" : "FAIL"}  ${name}${detail ? ` (${detail})` : ""}`);
  if (!cond) failed = true;
};

// --- PNG construction ------------------------------------------------------
const CRC_TABLE = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
function crc32(buf) {
  let c = 0xffffffff;
  for (const b of buf) c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function chunk(type, data) {
  const out = Buffer.alloc(12 + data.length);
  out.writeUInt32BE(data.length, 0);
  out.write(type, 4, "latin1");
  Buffer.from(data).copy(out, 8);
  out.writeUInt32BE(crc32(Buffer.concat([Buffer.from(type, "latin1"), Buffer.from(data)])), 8 + data.length);
  return out;
}
function png(extraChunks = [], trailing = Buffer.alloc(0)) {
  const sig = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(1, 0); ihdr.writeUInt32BE(1, 4);
  ihdr[8] = 8; ihdr[9] = 2; // 8-bit RGB
  const idat = deflateSync(Buffer.from([0, 0, 0, 0])); // 1 black pixel
  return new Uint8Array(Buffer.concat([
    sig, chunk("IHDR", ihdr), ...extraChunks, chunk("IDAT", idat), chunk("IEND", Buffer.alloc(0)), trailing,
  ]));
}
const text = (kw, body) => Buffer.concat([Buffer.from(kw, "latin1"), Buffer.from([0]), Buffer.from(body, "latin1")]);

// --- JPEG construction -----------------------------------------------------
function jpegSegment(marker, data) {
  const out = Buffer.alloc(4 + data.length);
  out[0] = 0xff; out[1] = marker;
  out.writeUInt16BE(data.length + 2, 2);
  Buffer.from(data).copy(out, 4);
  return out;
}
function jpeg(segments = [], trailing = Buffer.alloc(0)) {
  return new Uint8Array(Buffer.concat([
    Buffer.from([0xff, 0xd8]), ...segments,
    Buffer.from([0xff, 0xda, 0x00, 0x02]), Buffer.from([0x12, 0x34]), // SOS + fake scan
    Buffer.from([0xff, 0xd9]), trailing,
  ]));
}
const C2PA_URI = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia";

// --- the #43 spoof: raw payload appended to a real JPEG --------------------
const spoof = Buffer.concat([Buffer.from("tEXtprompt", "latin1"), Buffer.from([0]), Buffer.from('{"', "latin1")]);
check("spoof payload is 13 bytes (as reported)", spoof.length === 13, `${spoof.length}`);
check("JPEG + appended comfyui payload: NO hit", !sniffMetadata(jpeg([], spoof)).hit);
check("JPEG + payload inside COM segment: NO hit", !sniffMetadata(jpeg([jpegSegment(0xfe, spoof)])).hit);
check("JPEG + C2PA URI appended raw: NO hit", !sniffMetadata(jpeg([], Buffer.from(C2PA_URI))).hit);
check("JPEG + C2PA URI in COM segment: NO hit", !sniffMetadata(jpeg([jpegSegment(0xfe, Buffer.from(C2PA_URI))])).hit);
check("PNG + payload after IEND: NO hit", !sniffMetadata(png([], spoof)).hit);

// --- genuine provenance still short-circuits -------------------------------
let r = sniffMetadata(png([chunk("tEXt", text("parameters", "masterpiece, Steps: 20"))]));
check("PNG tEXt parameters (SD WebUI): hit", r.hit && r.reason === "png:sd-parameters", r.reason);
r = sniffMetadata(png([chunk("tEXt", text("prompt", '{"1":{"class_type":"KSampler"}}'))]));
check("PNG tEXt prompt JSON (ComfyUI): hit", r.hit && r.reason === "png:comfyui-prompt", r.reason);
r = sniffMetadata(png([chunk("iTXt", text("workflow", '\0\0\0\0{"nodes":[]}'))]));
check("PNG iTXt workflow (ComfyUI): hit", r.hit && r.reason === "png:comfyui-workflow", r.reason);
r = sniffMetadata(jpeg([jpegSegment(0xeb, Buffer.concat([Buffer.from("JP\x00 jumb"), Buffer.from(C2PA_URI)]))]));
check("JPEG APP11 C2PA URI: hit", r.hit && r.reason === "c2pa:trainedAlgorithmicMedia", r.reason);
r = sniffMetadata(jpeg([jpegSegment(0xe1, Buffer.from(`http://ns.adobe.com/xap/1.0/\0<x:xmpmeta><rdf:Description Iptc4xmpExt:DigitalSourceType="${C2PA_URI}"/></x:xmpmeta>`))]));
check("JPEG APP1 XMP digitalSourceType: hit", r.hit && r.reason === "c2pa:trainedAlgorithmicMedia", r.reason);
r = sniffMetadata(png([chunk("iTXt", text("XML:com.adobe.xmp", `\0\0\0\0<x:xmpmeta>${C2PA_URI}</x:xmpmeta>`))]));
check("PNG iTXt XMP digitalSourceType: hit", r.hit && r.reason === "c2pa:trainedAlgorithmicMedia", r.reason);

// --- benign metadata must not hit ------------------------------------------
check("PNG tEXt Comment: NO hit", !sniffMetadata(png([chunk("tEXt", text("Comment", "hello world"))])).hit);
check("PNG tEXt prompt non-JSON: NO hit", !sniffMetadata(png([chunk("tEXt", text("prompt", "say cheese"))])).hit);

// --- every shipped sample image must stay metadata-clean -------------------
const IMAGES = resolve(here, "./sample_images");
const bad = [];
for (const f of readdirSync(IMAGES)) {
  if (![".jpg", ".jpeg", ".png", ".webp"].includes(extname(f).toLowerCase())) continue;
  const res = sniffMetadata(new Uint8Array(readFileSync(join(IMAGES, f))));
  if (res.hit) bad.push(`${f}:${res.reason}`);
}
check("sample corpus: no metadata hits", bad.length === 0, bad.join(", "));

process.exit(failed ? 1 : 0);
