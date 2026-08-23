// Focused test for the full round-code labels shown in the Recent tournaments
// table and H2H meeting metadata. Runs with `pnpm test` (Vitest) from the web/ directory.

import { assert, test } from "vitest";
import { ROUND_LABEL } from "../src/lib/format.ts";

test("every supported round code maps to a full human label", () => {
  assert.deepEqual(ROUND_LABEL, {
    r128: "Round of 128",
    r64: "Round of 64",
    r32: "Round of 32",
    r16: "Round of 16",
    qf: "Quarterfinal",
    sf: "Semifinal",
    f: "Final",
    rr: "Round Robin",
  });
});
