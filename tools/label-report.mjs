#!/usr/bin/env node
// Auto-label a misclassification issue (run by CI on issue open/edit).
// Env: ISSUE_NUMBER, GH_REPO (owner/repo). Uses the gh CLI for API access.
import { execFileSync } from "node:child_process";
import { classify } from "./report-parser.mjs";

const issue = process.env.ISSUE_NUMBER;
const repo = process.env.GH_REPO;
if (!issue || !repo) throw new Error("ISSUE_NUMBER and GH_REPO required");

const gh = (...args) => execFileSync("gh", args, { encoding: "utf8" });

const data = JSON.parse(gh("api", `repos/${repo}/issues/${issue}`));
const current = (data.labels || []).map((l) => l.name);
if (!(data.title || "").toLowerCase().includes("misclassification") && !current.includes("misclassification")) {
  console.log("not a misclassification report; skipping");
  process.exit(0);
}

const { klass, score, truth } = classify(data.body || "");
const managed = ["false-positive", "false-negative", "borderline-miss", "unsure-on-real",
                 "consistent", "needs-review", "not-analyzed"];
const stale = managed.filter((n) => n !== klass && current.includes(n));
if (stale.length) gh("issue", "edit", issue, "--repo", repo, "--remove-label", stale.join(","));
gh("issue", "edit", issue, "--repo", repo, "--add-label", `misclassification,${klass}`);
console.log(`labeled #${issue}: ${klass} (score=${score}, truth=${truth})`);
