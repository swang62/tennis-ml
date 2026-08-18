// Hermetic tests for the shared idle-gated directory_info scheduling: the
// callback runs through requestIdleCallback when the engine supports it and
// otherwise after a short fixed delay, and canceling prevents the fallback
// from firing. No network, no DOM, no database.

import assert from "node:assert";
import { test } from "node:test";
import { IDLE_FALLBACK_MS, scheduleIdle } from "../src/lib/directoryInfo.ts";

test("schedules through requestIdleCallback when available", () => {
  const options = [];
  const originalRequestIdleCallback = globalThis.requestIdleCallback;
  const originalCancelIdleCallback = globalThis.cancelIdleCallback;
  globalThis.requestIdleCallback = (callback, opts) => {
    options.push(opts);
    callback();
    return 1;
  };
  globalThis.cancelIdleCallback = () => {};
  try {
    let fired = false;
    const cancel = scheduleIdle(() => {
      fired = true;
    });
    assert.strictEqual(fired, true);
    assert.deepStrictEqual(options, [{ timeout: 2000 }]);
    assert.strictEqual(typeof cancel, "function");
  } finally {
    globalThis.requestIdleCallback = originalRequestIdleCallback;
    globalThis.cancelIdleCallback = originalCancelIdleCallback;
  }
});

test("falls back to a delayed timer without requestIdleCallback", async () => {
  assert.strictEqual(typeof globalThis.requestIdleCallback, "undefined");
  let fired = false;
  const cancel = scheduleIdle(() => {
    fired = true;
  });
  assert.strictEqual(fired, false, "not fired before the fallback delay");
  await new Promise((resolve) => setTimeout(resolve, IDLE_FALLBACK_MS + 20));
  assert.strictEqual(fired, true, "fired after the fallback delay");
  cancel();
});

test("cancel prevents the delayed fallback from firing", async () => {
  let fired = false;
  const cancel = scheduleIdle(() => {
    fired = true;
  });
  cancel();
  await new Promise((resolve) => setTimeout(resolve, IDLE_FALLBACK_MS + 20));
  assert.strictEqual(fired, false);
});
