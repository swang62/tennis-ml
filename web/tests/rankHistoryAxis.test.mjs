// Focused tests for the deterministic yearly axis of the rank-history chart.
// Runs with `pnpm test` (Vitest) from the web/ directory.

import { assert, test } from "vitest";
import { yearAxisDomain } from "../src/lib/rankHistoryAxis.ts";

test("one tick per year whose Jan 1 lies inside the displayed domain", () => {
  // 2023's Jan 1 predates the first data point (2023-06-15), so it is dropped.
  const d = yearAxisDomain([
    "2023-06-15",
    "2024-02-01",
    "2025-11-20",
    "2026-03-05",
  ]);
  assert.deepEqual(
    d.ticks.map((t) => new Date(t).getFullYear()),
    [2024, 2025, 2026],
  );
});

test("every tick is Jan 1 local midnight", () => {
  const d = yearAxisDomain(["2023-01-02", "2024-05-01"]);
  assert.ok(d.ticks.length > 0);
  for (const t of d.ticks) {
    const dt = new Date(t);
    assert.equal(dt.getMonth(), 0);
    assert.equal(dt.getDate(), 1);
  }
});

test("min is the earliest data date; max is the last data date", () => {
  const d = yearAxisDomain(["2023-06-15", "2024-02-01", "2025-11-20"]);
  assert.equal(d.min, new Date(2023, 5, 15).getTime());
  assert.equal(d.max, new Date(2025, 10, 20).getTime());
  // 2023's Jan 1 falls before min, so no tick for 2023.
  assert.deepEqual(d.ticks, [
    new Date(2024, 0, 1).getTime(),
    new Date(2025, 0, 1).getTime(),
  ]);
});

test("unsorted input yields the same domain", () => {
  const a = yearAxisDomain(["2023-01-01", "2024-06-01", "2025-12-31"]);
  const b = yearAxisDomain(["2025-12-31", "2023-01-01", "2024-06-01"]);
  assert.deepEqual(a, b);
});

test("single-year data spanning Jan 1 yields exactly one tick", () => {
  const d = yearAxisDomain(["2023-01-01", "2023-11-30"]);
  assert.deepEqual(
    d.ticks.map((t) => new Date(t).getFullYear()),
    [2023],
  );
});

test("single-year data without Jan 1 in the domain yields no ticks", () => {
  // Axis starts at the earliest rank date (2023-02-01), so 2023's Jan 1
  // predates the domain: no tick, no artificial padding.
  const d = yearAxisDomain(["2023-02-01", "2023-11-30"]);
  assert.deepEqual(d.ticks, []);
  assert.equal(d.min, new Date(2023, 1, 1).getTime());
});

test("empty or malformed input yields null", () => {
  assert.equal(yearAxisDomain([]), null);
  assert.equal(yearAxisDomain(["not-a-date"]), null);
  assert.equal(yearAxisDomain(["not-a-date", "also-bad"]), null);
});
