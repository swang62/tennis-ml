// Hermetic tests for the pure ATP overview URL helper in lib/atpProfile.ts.
// Deterministic string derivation only; no network, no DOM, no database.

import assert from "node:assert";
import { test } from "node:test";
import { atpOverviewUrl } from "../src/lib/atpProfile.ts";

test("AG37 fixture resolves to the expected ATP overview URL", () => {
  assert.equal(
    atpOverviewUrl("Felix Auger-Aliassime", "AG37"),
    "https://www.atptour.com/en/players/felix-auger-aliassime/ag37/overview",
  );
});

test("slug strips accents", () => {
  assert.equal(
    atpOverviewUrl("Gaël Monfils", "MC15"),
    "https://www.atptour.com/en/players/gael-monfils/mc15/overview",
  );
});

test("slug collapses punctuation and repeated whitespace", () => {
  assert.equal(
    atpOverviewUrl("  John   McEnroe Jr. ", "M456"),
    "https://www.atptour.com/en/players/john-mcenroe-jr/m456/overview",
  );
});

test("player id is lowercased for the URL segment", () => {
  assert.equal(
    atpOverviewUrl("Novak Djokovic", "D643"),
    "https://www.atptour.com/en/players/novak-djokovic/d643/overview",
  );
});
