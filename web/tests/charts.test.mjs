// Focused tests for the pure color helpers in lib/charts.ts: withAlpha and
// surfaceColor. No API, DOM, or DB. Runs with `pnpm test` (Vitest) from the web/ directory.

import { assert, test } from "vitest";
import { surfaceColor, withAlpha } from "../src/lib/charts.ts";

const dark = {
  theme: "dark",
  text: "#e8e8e8",
  dim: "#9aa0aa",
  faint: "#666666",
  line: "#333333",
  clay: "#c98d63",
  grass: "#3fae7a",
  ice: "#5f9fc9",
  raised: "#1c1c22",
  inset: "#141419",
};

const light = { ...dark, theme: "light" };

test("withAlpha converts 6-digit hex to rgba", () => {
  assert.equal(withAlpha("#ff0000", 0.5), "rgba(255, 0, 0, 0.5)");
  assert.equal(withAlpha("#00ff00", 1), "rgba(0, 255, 0, 1)");
  assert.equal(withAlpha("#0000ff", 0.28), "rgba(0, 0, 255, 0.28)");
});

test("withAlpha passes non-6-digit-hex through unchanged", () => {
  assert.equal(withAlpha("#fff", 0.5), "#fff");
  assert.equal(withAlpha("red", 0.5), "red");
  assert.equal(withAlpha("rgb(1,2,3)", 0.5), "rgb(1,2,3)");
});

test("surfaceColor maps known surfaces to theme tokens", () => {
  assert.equal(surfaceColor("grass", dark), dark.grass);
  assert.equal(surfaceColor("hard", dark), dark.ice);
});

test("surfaceColor mutes clay differently per theme", () => {
  assert.equal(surfaceColor("clay", dark), "#c98d63");
  assert.equal(surfaceColor("clay", light), "#a87850");
});

test("surfaceColor falls back to the carpet token then dim", () => {
  assert.equal(surfaceColor("carpet", dark), "#8d93ad");
  assert.equal(surfaceColor("unknown", dark), dark.dim);
});
