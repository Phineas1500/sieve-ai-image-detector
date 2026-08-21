// Metadata forensics: high-precision structural markers of AI generation.
// Runs before model inference; a hit short-circuits to 0.99 confidence.
//
// Only unambiguous, structural markers are used. Brand-name strings
// ("OpenAI", "Midjourney", ...) are deliberately excluded: real news photos
// about AI companies carry those words in XMP captions, and false positives
// are far more costly than missed metadata (the model still runs on misses).
//
// Markers are only honored inside the container structures that actually
// carry them — PNG tEXt/iTXt/caBX chunks and JPEG APP11 (JUMBF) segments —
// never via a raw byte search. A raw search let a 13-byte payload appended
// to any JPEG flip a real photo to 0.99 (issue #43); a marker outside its
// valid container is spoofed garbage, not provenance. Crafting a
// structurally valid chunk still triggers the hit, deliberately: an image
// whose metadata genuinely declares AI provenance should flag.

function ascii(s) {
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i) & 0xff;
  return out;
}

// C2PA digitalSourceType URIs — only ever appear inside content-credential
// manifests (JPEG APP11 JUMBF / PNG caBX).
const C2PA_MARKERS = [
  { pat: ascii("http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"), reason: "c2pa:trainedAlgorithmicMedia" },
  { pat: ascii("compositeWithTrainedAlgorithmicMedia"), reason: "c2pa:compositeWithTrainedAlgorithmicMedia" },
];

function indexOfBytes(hay, pat) {
  const n = hay.length - pat.length;
  const first = pat[0];
  outer: for (let i = 0; i <= n; i++) {
    if (hay[i] !== first) continue;
    for (let j = 1; j < pat.length; j++) {
      if (hay[i + j] !== pat[j]) continue outer;
    }
    return i;
  }
  return -1;
}

function sniffC2PA(data) {
  for (const { pat, reason } of C2PA_MARKERS) {
    if (indexOfBytes(data, pat) !== -1) return { hit: true, reason };
  }
  return null;
}

const PNG_SIG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

// Walk PNG chunks up to `cap`. Chunk keywords are case-sensitive per spec and
// match the generators exactly: Stable Diffusion WebUI writes "parameters",
// ComfyUI embeds its node graph as JSON under "prompt"/"workflow".
function sniffPNG(b, cap) {
  let o = PNG_SIG.length;
  while (o + 12 <= cap) {
    const len = ((b[o] << 24) | (b[o + 1] << 16) | (b[o + 2] << 8) | b[o + 3]) >>> 0;
    const type = String.fromCharCode(b[o + 4], b[o + 5], b[o + 6], b[o + 7]);
    if (type === "IEND") break; // trailing bytes after IEND are not metadata
    const data = b.subarray(o + 8, Math.min(o + 8 + len, cap));
    if (type === "tEXt" || type === "iTXt") {
      const nul = data.indexOf(0);
      if (nul > 0) {
        const kw = String.fromCharCode(...data.subarray(0, Math.min(nul, 80)));
        if (kw === "parameters") return { hit: true, reason: "png:sd-parameters" };
        if (kw === "prompt" || kw === "workflow") {
          // tEXt payload must look like ComfyUI's JSON graph; iTXt carries
          // compression/language headers before the text, so keyword suffices.
          const json = data[nul + 1] === 0x7b && data[nul + 2] === 0x22; // {"
          if (type === "iTXt" || json) return { hit: true, reason: `png:comfyui-${kw}` };
        }
        // IPTC digitalSourceType also ships as plain XMP without a full
        // C2PA manifest.
        if (type === "iTXt" && kw === "XML:com.adobe.xmp") {
          const c2pa = sniffC2PA(data);
          if (c2pa) return c2pa;
        }
      }
    }
    if (type === "caBX") {
      const c2pa = sniffC2PA(data);
      if (c2pa) return c2pa;
    }
    o += 12 + len;
  }
  return { hit: false };
}

// Walk JPEG segments up to the scan data; C2PA lives in APP11 (0xFFEB) JUMBF
// payloads. A manifest spanning several APP11 segments can split a marker
// across a boundary — an acceptable miss, the model still runs.
function sniffJPEG(b, cap) {
  let o = 2;
  while (o + 4 <= cap) {
    if (b[o] !== 0xff) break; // desynced — stop rather than scan pixel data
    const marker = b[o + 1];
    if (marker === 0xff) { o++; continue; } // fill byte
    if (marker === 0xd9 || marker === 0xda) break; // EOI / start of scan
    if (marker >= 0xd0 && marker <= 0xd8) { o += 2; continue; } // standalone
    const len = (b[o + 2] << 8) | b[o + 3];
    if (len < 2) break;
    // APP11 carries JUMBF/C2PA manifests; APP1 carries XMP, where the IPTC
    // digitalSourceType URI ships without a full manifest.
    if (marker === 0xeb || marker === 0xe1) {
      const c2pa = sniffC2PA(b.subarray(o + 4, Math.min(o + 2 + len, cap)));
      if (c2pa) return c2pa;
    }
    o += 2 + len;
  }
  return { hit: false };
}

/**
 * @param {Uint8Array} bytes raw image file bytes
 * @returns {{hit: boolean, reason?: string}}
 */
export function sniffMetadata(bytes) {
  // Cap the scan: metadata lives in the head of the file. 512KB covers even
  // bloated XMP/C2PA blocks while keeping worst-case cost trivial.
  const cap = Math.min(bytes.length, 524288);
  if (bytes.length >= 16 && PNG_SIG.every((v, i) => bytes[i] === v)) {
    return sniffPNG(bytes, cap);
  }
  if (bytes.length >= 4 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
    return sniffJPEG(bytes, cap);
  }
  // Other containers (WebP/AVIF/GIF): no structural parser — no metadata
  // verdict. The model runs unconditionally on these.
  return { hit: false };
}
