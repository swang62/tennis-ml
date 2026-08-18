// Shared static player-directory source for Home and H2H. The directory
// payload is bundled in the Vite entry, so initial picker data is synchronous
// with no network state. The serialized MiniSearch index is emitted as a
// separate Vite-hashed asset and fetched only after the first non-empty
// picker search; it is never constructed in the browser. HTTP caching of the
// immutable hashed search asset is the only client cache (no localStorage, no
// polling, no API fallback).

import { useQuery } from "@tanstack/react-query";
import type MiniSearch from "minisearch";
import type { Player } from "../api";
// Generated before Vite/dev runs by web/scripts/build-player-index.mjs from
// data/deploy/player-directory.json (missing inputs fail the build on purpose).
import directoryJson from "../assets/generated/player-directory.json" with {
  type: "json",
};
// ?url&no-inline emits the search payload as a hashed /assets/ file instead of
// inlining it into the entry chunk.
import searchAssetUrl from "../assets/generated/player-search.json?url&no-inline" with {
  type: "json",
};

// Must match MINISEARCH_OPTS in web/scripts/build-player-index.mjs.
export const MINISEARCH_OPTS = Object.freeze({
  fields: ["display_name"],
  idField: "player_id",
  searchOptions: { fuzzy: 0.2, prefix: true, boost: { display_name: 2 } },
});

export interface PlayerIndexData {
  players: Player[];
  loadSearch: () => Promise<PlayerSearch>;
}

export type PlayerSearch = (query: string) => Player[];

type MiniSearchConstructor = typeof MiniSearch;

// The MiniSearch module and serialized index are loaded only after the user
// types a query; initial picker defaults use the directory payload alone.
// `MiniSearchCtor` is passed by loadPlayerSearch so the asset fetch and the
// module import run concurrently; the standalone fallback keeps callers that
// only have the payload (tests) working.
export async function deserializePlayerSearch(
  indexPayload: string,
  players: Player[],
  MiniSearchCtor?: MiniSearchConstructor,
): Promise<PlayerSearch> {
  const MiniSearchClass =
    MiniSearchCtor ?? (await import("minisearch")).default;
  const index = MiniSearchClass.loadJSON(indexPayload, MINISEARCH_OPTS);
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

// One fetch + one minisearch chunk load on the first invocation; later calls
// reuse the resolved in-memory search function.
let searchPromise: Promise<PlayerSearch> | undefined;
export function loadPlayerSearch(): Promise<PlayerSearch> {
  searchPromise ??= Promise.all([
    fetch(searchAssetUrl).then(async (res) => {
      if (!res.ok) {
        throw new Error(`player search index: HTTP ${res.status}`);
      }
      return ((await res.json()) as { index: string }).index;
    }),
    import("minisearch"),
  ]).then(([indexPayload, { default: MiniSearchClass }]) =>
    deserializePlayerSearch(
      indexPayload,
      directoryJson.players,
      MiniSearchClass,
    ),
  );
  return searchPromise;
}

const directoryData: PlayerIndexData = {
  players: directoryJson.players,
  loadSearch: loadPlayerSearch,
};

export const playerIndexQueryKey = ["player-index"] as const;

// One shared query key/source: whichever of Layout/Home/H2H mounts first
// exposes the bundled players synchronously; the rest read the same stable
// cached result. initialData keeps isLoading false from first render.
export function usePlayerDirectory() {
  return useQuery({
    queryKey: playerIndexQueryKey,
    queryFn: () => directoryData,
    initialData: directoryData,
    staleTime: Infinity,
    gcTime: Infinity,
  });
}
