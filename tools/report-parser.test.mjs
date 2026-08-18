// Run: node --test tools/
import { test } from "node:test";
import assert from "node:assert";
import { classify, parseVerdict } from "./report-parser.mjs";

const body = (said, actually) => `### What is the image actually?

${actually}

### Image URL

https://example.com/x.jpg

### What did Sieve say?

${said}

### Extension version

0.5.4`;

test("auto-filled flagged + real -> false-positive", () => {
  const r = classify(body("flagged as AI — score 76.8%", "Real / human-made"));
  assert.equal(r.klass, "false-positive");
  assert.ok(Math.abs(r.score - 0.768) < 1e-9);
});

test("auto-filled low + AI -> false-negative", () => {
  assert.equal(classify(body("low score — score 12.3%", "AI-generated")).klass, "false-negative");
});

test("auto-filled unsure + AI -> borderline-miss", () => {
  assert.equal(classify(body("unsure — score 58.0% (TTA)", "AI-generated")).klass, "borderline-miss");
});

test("manual badge text 'AI 87%' + real -> false-positive", () => {
  assert.equal(classify(body("AI 87%", "Real / human-made")).klass, "false-positive");
});

test("manual badge text 'unsure 58%' + real -> unsure-on-real", () => {
  assert.equal(classify(body("unsure 58%", "Real / human-made")).klass, "unsure-on-real");
});

test("manual bare percent + AI -> false-negative", () => {
  assert.equal(classify(body("12%", "AI-generated")).klass, "false-negative");
});

test("verdict agrees with truth -> consistent", () => {
  assert.equal(classify(body("AI 99%", "AI-generated")).klass, "consistent");
});

test("not analyzed", () => {
  assert.equal(classify(body("not analyzed", "AI-generated")).klass, "not-analyzed");
});

test("garbage verdict -> needs-review", () => {
  assert.equal(classify(body("it said like 80 something maybe", "AI-generated")).klass, "needs-review");
});

test("uncertain truth -> needs-review", () => {
  assert.equal(classify(body("AI 87%", "Not certain")).klass, "needs-review");
});

test("percent out of range -> needs-review", () => {
  assert.equal(parseVerdict("score 870%").score, null);
});
