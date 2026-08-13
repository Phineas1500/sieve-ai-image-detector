// Copies the ONNX Runtime Web runtime from node_modules into the extension.
// Everything the extension runs ships inside the extension directory —
// no CDN, no remote code.
import { cpSync, mkdirSync, readdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, "node_modules", "onnxruntime-web", "dist");
const dst = join(here, "vendor", "ort");

mkdirSync(dst, { recursive: true });
let n = 0;
for (const f of readdirSync(src)) {
  if (f.endsWith(".js") || f.endsWith(".wasm") || f.endsWith(".mjs")) {
    cpSync(join(src, f), join(dst, f));
    n++;
  }
}
console.log(`copied ${n} ONNX Runtime Web files -> vendor/ort/`);
