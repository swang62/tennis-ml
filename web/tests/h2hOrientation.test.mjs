import assert from "node:assert";
import test from "node:test";
import { preferenceEdge } from "../src/lib/h2hOrientation.ts";

test("preferenceEdge is negative when Player A is favored, positive when Player B is", () => {
  assert.ok(Math.abs(preferenceEdge(0.52) + 0.02) < 1e-10);
  assert.ok(Math.abs(preferenceEdge(0.48) - 0.02) < 1e-10);
  assert.equal(preferenceEdge(0.5), 0);
});
