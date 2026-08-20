// Shared API-fetched player-directory source for Home, H2H, and the Layout
// footer. The directory (players + summary) is fetched once per React Query
// cache lifecycle via GET /directory; no generated artifacts are bundled. The
// in-memory MiniSearch fuzzy/prefix index is built lazily from the fetched
// players on the first picker search and shared by every consumer (no
// localStorage, no polling, no background refresh).

import { useQuery } from "@tanstack/react-query";
import type { Player } from "../api";

export const MINISEARCH_OPTS = Object.freeze({
  fields: ["display_name"],
  idField: "player_id",
  searchOptions: { fuzzy: 0.2, prefix: true, boost: { display_name: 2 } },
});

export interface PlayerIndexData {
  players: Player[];
  latest_match_date: string | null;
  total_matches: number;
  loadSearch: () => Promise<PlayerSearch>;
}

export type PlayerSearch = (query: string) => Player[];

// Builds the in-memory MiniSearch index over the fetched players and returns
// a query function. The module import stays lazy so the minisearch chunk is
// loaded only when the first picker search runs; the index is built per
// directory payload, never serialized or fetched.
export async function buildPlayerSearch(
  players: Player[],
): Promise<PlayerSearch> {
  const { default: MiniSearchClass } = await import("minisearch");
  const index = new MiniSearchClass(MINISEARCH_OPTS);
  index.addAll(players);
  const playersById = new Map(
    players.map((player) => [player.player_id, player]),
  );
  return (query: string): Player[] => {
    const q = query.trim();
    if (!q) return [];
    return index
      .search(q, { fuzzy: 0.2, prefix: true })
      .flatMap((result) => playersById.get(String(result.id)) ?? []);
  };
}

// Shared lazy loader: the index is built once per directory payload; later
// calls reuse the resolved in-memory search function.
export function createPlayerSearchLoader(
  players: Player[],
): () => Promise<PlayerSearch> {
  let searchPromise: Promise<PlayerSearch> | undefined;
  return () => {
    searchPromise ??= buildPlayerSearch(players);
    return searchPromise;
  };
}

export const playerIndexQueryKey = ["player-index"] as const;

// One shared query key/source: whichever of Layout/Home/H2H mounts first
// fetches the directory once; the rest read the same cached result for the
// page's lifetime. staleTime/gcTime Infinity keep the payload in memory, so
// a reload retry re-fetches only on error.
export function usePlayerDirectory() {
  return useQuery({
    queryKey: playerIndexQueryKey,
    // Dynamic import keeps the hook node-testable: node's ESM loader cannot
    // resolve the bundler-style extensionless ../api, and Vite resolves it.
    queryFn: async () => {
      const directory = await import("../api").then(({ getDirectory }) =>
        getDirectory(),
      );
      return {
        ...directory,
        loadSearch: createPlayerSearchLoader(directory.players),
      } satisfies PlayerIndexData;
    },
    staleTime: Infinity,
    gcTime: Infinity,
  });
}
