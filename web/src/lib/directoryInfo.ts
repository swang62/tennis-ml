// Shared idle-gated directory metadata source for the Layout footer and Home
// stats. Both consumers watch one query key/result so /api/directory_info is
// requested exactly once, after the browser goes idle: it is not on the
// first-paint critical path, and neither consumer shows a placeholder value
// while it is pending.

import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";

export const directoryInfoKey = ["directory_info"] as const;

// Delayed fallback delay where requestIdleCallback is unavailable.
export const IDLE_FALLBACK_MS = 1000;

type CancelIdle = () => void;

// Runs the callback via requestIdleCallback when the engine supports it;
// otherwise after a short fixed delay. Returns a cancel function for cleanup.
export function scheduleIdle(callback: () => void): CancelIdle {
  if (typeof requestIdleCallback === "function") {
    const handle = requestIdleCallback(callback, { timeout: 2000 });
    return () => cancelIdleCallback(handle);
  }
  const handle = setTimeout(callback, IDLE_FALLBACK_MS);
  return () => clearTimeout(handle);
}

export function useDirectoryInfo() {
  const [ready, setReady] = useState(false);
  useEffect(() => scheduleIdle(() => setReady(true)), []);
  return useQuery({
    queryKey: directoryInfoKey,
    // Dynamic import keeps this hook node-testable: node's ESM loader cannot
    // resolve the bundler-style extensionless ../api, and Vite resolves it.
    queryFn: () =>
      import("../api").then(({ getDirectoryInfo }) => getDirectoryInfo()),
    enabled: ready,
    staleTime: Infinity,
    gcTime: Infinity,
  });
}
