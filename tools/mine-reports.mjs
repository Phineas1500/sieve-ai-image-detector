#!/usr/bin/env node
// Turn the public misclassification tracker into a training-ready CSV.
// Usage: node tools/mine-reports.mjs [--repo owner/repo] > reports.csv
import { execFileSync } from "node:child_process";
import { classify, extractSection } from "./report-parser.mjs";

const repoArg = process.argv.indexOf("--repo");
const repo = repoArg > -1 ? process.argv[repoArg + 1] : "Phineas1500/sieve-ai-image-detector";

const issues = JSON.parse(execFileSync("gh", [
  "issue", "list", "--repo", repo, "--label", "misclassification", "--state", "all",
  "--limit", "500", "--json", "number,title,body,createdAt,labels",
], { encoding: "utf8" }));

const esc = (s) => `"${String(s ?? "").replaceAll('"', '""').replaceAll("\n", " ")}"`;
console.log("issue,created,klass,score,truth,image_url,attachments,evidence");
for (const it of issues) {
  const { klass, score, truth } = classify(it.body || "");
  const url = extractSection(it.body || "", "Image URL") || "";
  const evidence = extractSection(it.body || "", "How do you know the true label\\?") || "";
  const attachments = [...(it.body || "").matchAll(/https:\/\/github\.com\/user-attachments\/[^\s)]+/g)]
    .map((m) => m[0]).join(" ");
  console.log([it.number, it.createdAt, klass, score ?? "", truth ?? "",
    esc(url), esc(attachments), esc(evidence.slice(0, 300))].join(","));
}
