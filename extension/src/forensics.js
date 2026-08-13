// Metadata forensics: high-precision structural markers of AI generation.
// Runs before model inference; a hit short-circuits to 0.99 confidence.
//
// Only unambiguous, structural markers are used. Brand-name strings
// ("OpenAI", "Midjourney", ...) are deliberately excluded: real news photos
// about AI companies carry those words in XMP captions, and false positives
// are far more costly than missed metadata (the model still runs on misses).

// Markers are byte arrays; PNG text-chunk keys are followed by a NUL (0x00)
// separator, which must be matched exactly.
function ascii(s) {
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i) & 0xff;
  return out;
}

const NUL = 0;

function concatBytes(...parts) {
  const len = parts.reduce((a, p) => a + (typeof p === "number" ? 1 : p.length), 0);
  const out = new Uint8Array(len);
  let o = 0;
  for (const p of parts) {
    if (typeof p === "number") out[o++] = p;
    else {
      out.set(p, o);
      o += p.length;
    }
  }
  return out;
}

const COMPILED = [
  // C2PA digitalSourceType URIs — only ever appear inside content-credential
  // manifests (JPEG APP11 JUMBF / PNG caBX), never in pixel data.
  { pat: ascii("http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"), reason: "c2pa:trainedAlgorithmicMedia" },
  { pat: ascii("compositeWithTrainedAlgorithmicMedia"), reason: "c2pa:compositeWithTrainedAlgorithmicMedia" },
  // Stable Diffusion WebUI: tEXt chunk keyed "parameters" (key + NUL).
  { pat: concatBytes(ascii("tEXtparameters"), NUL), reason: "png:sd-parameters" },
  { pat: concatBytes(ascii("iTXtparameters"), NUL), reason: "png:sd-parameters" },
  // ComfyUI embeds its node graph as JSON under "prompt"/"workflow" keys.
  { pat: concatBytes(ascii("tEXtprompt"), NUL, ascii('{"')), reason: "png:comfyui-prompt" },
  { pat: concatBytes(ascii("tEXtworkflow"), NUL, ascii('{"')), reason: "png:comfyui-workflow" },
  { pat: concatBytes(ascii("iTXtprompt"), NUL), reason: "png:comfyui-prompt" },
  { pat: concatBytes(ascii("iTXtworkflow"), NUL), reason: "png:comfyui-workflow" },
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

/**
 * @param {Uint8Array} bytes raw image file bytes
 * @returns {{hit: boolean, reason?: string}}
 */
export function sniffMetadata(bytes) {
  // Cap the scan: metadata lives in the head of the file. 512KB covers even
  // bloated XMP/C2PA blocks while keeping worst-case cost trivial.
  const head = bytes.length > 524288 ? bytes.subarray(0, 524288) : bytes;
  for (const { pat, reason } of COMPILED) {
    if (indexOfBytes(head, pat) !== -1) return { hit: true, reason };
  }
  return { hit: false };
}
