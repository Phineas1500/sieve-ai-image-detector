// Shared parser for misclassification reports (GitHub issue-form bodies).
// Handles both the auto-filled verdict format ("flagged as AI — score 76.8%")
// and manual badge-style entries ("AI 87%", "unsure 58%", "12%", "not analyzed").

export function extractSection(body, heading) {
  const re = new RegExp(`###\\s*${heading}\\s*\\n+([\\s\\S]*?)(?=\\n###|$)`, "i");
  const m = body.match(re);
  return m ? m[1].trim() : null;
}

export function parseVerdict(text) {
  if (!text) return { score: null, analyzed: null };
  const t = text.trim().toLowerCase();
  if (/not\s*analy[sz]ed|_no response_/.test(t)) return { score: null, analyzed: false };
  const m = t.match(/(\d{1,3}(?:\.\d+)?)\s*%/);
  if (!m) return { score: null, analyzed: null };
  const score = Number(m[1]) / 100;
  if (!(score >= 0 && score <= 1)) return { score: null, analyzed: null };
  return { score, analyzed: true };
}

export function parseTruth(text) {
  if (!text) return null;
  const t = text.toLowerCase();
  if (/ai[- ]generated/.test(t)) return "ai";
  if (/real|human[- ]made/.test(t)) return "real";
  if (/not certain/.test(t)) return "uncertain";
  return null;
}

// Classify a report into the mining taxonomy.
//   false-positive : real image at/above the flag threshold
//   false-negative : AI image below 0.5
//   borderline-miss: AI image in the unsure band [0.5, 0.65)
//   unsure-on-real : real image in the unsure band
//   consistent     : verdict already agrees with the reported truth
//   needs-review   : anything unparseable/uncertain
export function classify(body) {
  const verdict = parseVerdict(extractSection(body, "What did Sieve say\\?"));
  const truth = parseTruth(extractSection(body, "What is the image actually\\?"));
  const out = { score: verdict.score, truth, klass: "needs-review" };
  if (truth === "ai" || truth === "real") {
    if (verdict.score === null) out.klass = verdict.analyzed === false ? "not-analyzed" : "needs-review";
    else if (truth === "real" && verdict.score >= 0.65) out.klass = "false-positive";
    else if (truth === "ai" && verdict.score < 0.5) out.klass = "false-negative";
    else if (truth === "ai" && verdict.score < 0.65) out.klass = "borderline-miss";
    else if (truth === "real" && verdict.score >= 0.5) out.klass = "unsure-on-real";
    else out.klass = "consistent";
  }
  return out;
}
