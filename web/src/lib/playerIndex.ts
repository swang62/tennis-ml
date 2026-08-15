// Shared static player-directory source for Home and H2H.
// The browser fetches only the deploy-built manifest, then its content-hashed
// payload, and deserializes the MiniSearch index with loadJSON — the index is
// never constructed in the browser. HTTP caching of the immutable hashed asset
// is the only client cache (no localStorage, no polling, no API fallback).

import { useQuery } from "@tanstack/react-query";
import type { Player } from "../api";

export const PLAYER_INDEX_MANIFEST = "/player-index.manifest.json";

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

// The MiniSearch module and serialized index are loaded only after the user
// types a query; initial picker defaults use the directory payload alone.
export async function deserializePlayerSearch(indexPayload: string, players: Player[]): Promise<PlayerSearch> {
  const { default: MiniSearch } = await import("minisearch");
  const index = MiniSearch.loadJSON(indexPayload, MINISEARCH_OPTS);
  const playersById = new Map(players.map((player) => [player.player_id, player]));
  return (query: string): Player[] => {
    const q = query.trim();
    if (!q) return [];
    return index
      .search(q, { fuzzy: 0.2, prefix: true })
      .flatMap((result) => playersById.get(String(result.id)) ?? []);
  };
}

export async function fetchPlayerIndex(): Promise<PlayerIndexData> {
  const manifestRes = await fetch(PLAYER_INDEX_MANIFEST);
  if (!manifestRes.ok) {
    throw new Error(`player index manifest: HTTP ${manifestRes.status}`);
  }
  const manifest = (await manifestRes.json()) as {
    directoryPath?: string;
    searchPath?: string;
    path?: string;
  };
  const payloadRes = await fetch(manifest.directoryPath ?? manifest.path!);
  if (!payloadRes.ok) {
    throw new Error(`player index payload: HTTP ${payloadRes.status}`);
  }
  const payload = (await payloadRes.json()) as {
    players: Player[];
    index?: string;
  };
  let searchPromise: Promise<PlayerSearch> | undefined;
  return {
    ...payload,
    loadSearch: () => {
      searchPromise ??= manifest.searchPath
        ? fetch(manifest.searchPath)
            .then((res) => {
              if (!res.ok) throw new Error(`player search index: HTTP ${res.status}`);
              return res.json() as Promise<{ index: string }>;
            })
            .then(({ index }) => deserializePlayerSearch(index, payload.players))
        : payload.index
          ? deserializePlayerSearch(payload.index, payload.players)
          : Promise.reject(new Error("player search index missing from manifest"));
      return searchPromise;
    },
  };
}

export const playerIndexQueryKey = ["player-index"] as const;

// One shared query key/source: whichever of Layout/Home/H2H mounts first loads
// the static index; the rest read the same cached data.
export function usePlayerDirectory() {
  return useQuery({
    queryKey: playerIndexQueryKey,
    queryFn: fetchPlayerIndex,
    staleTime: Infinity,
    gcTime: Infinity,
  });
}
