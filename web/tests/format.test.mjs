// Focused tests for the remaining pure display formatters in lib/format.ts:
// pct, fairOdds, sanitizeErrorMessage, and TIER_LABEL. No API, DOM, or DB.
// Runs with `node --test tests/` (Node >= 23.6 strips types natively).

import assert from "node:assert";
import { test } from "node:test";
import {
  fairOdds,
  pct,
  sanitizeErrorMessage,
  TIER_LABEL,
} from "../src/lib/format.ts";

test("pct formats a 0..1 ratio to one decimal place", () => {
  assert.equal(pct(0.657), "65.7%");
  assert.equal(pct(0.5), "50%");
  assert.equal(pct(0), "0%");
  assert.equal(pct(1), "100%");
});

test("pct renders null and out-of-range as an em dash", () => {
  assert.equal(pct(null), "—");
  assert.equal(pct(-0.1), "—");
  assert.equal(pct(1.1), "—");
});

test("fairOdds is the zero-margin decimal odds implied by p", () => {
  assert.equal(fairOdds(0.5), "2.0");
  assert.equal(fairOdds(0.25), "4.0");
  assert.equal(fairOdds(0.2), "5.0");
});

test("fairOdds renders an em dash for degenerate probabilities", () => {
  assert.equal(fairOdds(0), "—");
  assert.equal(fairOdds(1), "—");
  assert.equal(fairOdds(-0.5), "—");
  assert.equal(fairOdds(NaN), "—");
});

test("sanitizeErrorMessage removes known ids by whole word", () => {
  assert.equal(
    sanitizeErrorMessage("Player S0AG not found", ["S0AG"]),
    "Player not found",
  );
  assert.equal(
    sanitizeErrorMessage("S0AG S0AG unknown S0AG", ["S0AG"]),
    "unknown",
  );
});

test("sanitizeErrorMessage escapes regex metacharacters in ids", () => {
  assert.equal(
    sanitizeErrorMessage("id a.b not found", ["a.b"]),
    "id not found",
  );
});

test("sanitizeErrorMessage collapses whitespace and trims trailing punctuation", () => {
  assert.equal(sanitizeErrorMessage("  a   b  ", []), "a b");
  assert.equal(sanitizeErrorMessage("failed, ", []), "failed");
  assert.equal(sanitizeErrorMessage("error: ", []), "error");
});

test("every tournament tier maps to a human label", () => {
  assert.deepEqual(TIER_LABEL, {
    grand_slam: "Grand Slam",
    masters: "Masters",
    atp_500: "ATP 500",
    atp_250: "ATP 250",
    davis_cup: "Davis Cup",
    atp_finals: "ATP Finals",
    olympics: "Olympics",
    professional: "Pro",
  });
});
