import assert from "node:assert";
import test from "node:test";
import { orientedProbability } from "../src/lib/h2hOrientation.ts";

test("orientedProbability is the predicted player's probability percent", () => {
  // Player A predicted: axis value is Player A's win probability.
  assert.equal(orientedProbability(0.5, true), 50);
  assert.ok(Math.abs(orientedProbability(0.62, true) - 62) < 1e-9);
  // Player B predicted: axis value is Player B's win probability, so Player
  // A's 0.62 maps to 38 and the bar points the other way.
  assert.ok(Math.abs(orientedProbability(0.62, false) - 38) < 1e-9);
  assert.ok(Math.abs(orientedProbability(0.38, false) - 62) < 1e-9);
});

test("orientedProbability stays inside the 0-100 axis domain", () => {
  assert.equal(orientedProbability(0, true), 0);
  assert.equal(orientedProbability(1, true), 100);
  assert.equal(orientedProbability(0, false), 100);
  assert.equal(orientedProbability(1, false), 0);
});
