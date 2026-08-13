// Unit tests for the metadata forensics module. Run: node --test extension/test/
import { test } from "node:test";
import assert from "node:assert";
import { sniffMetadata } from "../src/forensics.js";

function ascii(s) {
  return Uint8Array.from(s, (c) => c.charCodeAt(0) & 0xff);
}

function pngWithTextChunk(key, value) {
  // minimal PNG-ish byte stream: signature + fake IHDR + tEXt chunk
  const sig = Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const data = ascii(key + "\x00" + value);
  const chunk = new Uint8Array(8 + data.length + 4);
  new DataView(chunk.buffer).setUint32(0, data.length);
  chunk.set(ascii("tEXt"), 4);
  chunk.set(data, 8);
  const out = new Uint8Array(sig.length + chunk.length);
  out.set(sig);
  out.set(chunk, sig.length);
  return out;
}

test("detects Stable Diffusion parameters chunk", () => {
  const png = pngWithTextChunk("parameters", "a cat\nSteps: 20, Sampler: Euler a, CFG scale: 7");
  const r = sniffMetadata(png);
  assert.equal(r.hit, true);
  assert.equal(r.reason, "png:sd-parameters");
});

test("detects ComfyUI workflow chunk", () => {
  const png = pngWithTextChunk("workflow", '{"1":{"class_type":"KSampler"}}');
  const r = sniffMetadata(png);
  assert.equal(r.hit, true);
  assert.equal(r.reason, "png:comfyui-workflow");
});

test("detects C2PA trainedAlgorithmicMedia URI anywhere in head", () => {
  const buf = new Uint8Array(4096).fill(0x41);
  buf.set(ascii("http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"), 1000);
  assert.equal(sniffMetadata(buf).hit, true);
});

test("no hit on a clean image head", () => {
  const clean = pngWithTextChunk("Description", "A real photo of OpenAI headquarters, by Reuters");
  assert.equal(sniffMetadata(clean).hit, false);
});

test("no hit on brand names in captions", () => {
  const buf = ascii("\xff\xd8\xff\xe1 XMP: This Midjourney-style photo shows DALL-E research at OpenAI");
  assert.equal(sniffMetadata(buf).hit, false);
});

test("marker beyond 512KB scan cap is ignored", () => {
  const buf = new Uint8Array(600000).fill(0x42);
  buf.set(ascii("compositeWithTrainedAlgorithmicMedia"), 550000);
  assert.equal(sniffMetadata(buf).hit, false);
});
