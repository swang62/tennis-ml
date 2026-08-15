// Shared static player-directory source for Home, H2H, and the layout footer.
// The browser fetches only the deploy-built manifest, then its content-hashed
// payload, and deserializes the MiniSearch index with loadJSON — the index is
// never constructed in the browser. HTTP caching of the immutable hashed asset
// is the only client cache (no localStorage, no polling, no API fallback).

import { useQuery } from "@tanstack/react-query";
import MiniSearch from "minisearch";
import type { Player } from "../api";

export const PLAYER_INDEX_MANIFEST = "/player-index.manifest.json";

// Must match MINISEARCH_OPTS in web/scripts/build-player-index.mjs: a
// serialized index only deserializes with the exact options it was built with.
export const MINISEARCH_OPTS = Object.freeze({
  fields: ["display_name"],
  idField: "player_id",
  storeFields: [
    "display_name",
    "matches_played",
    "latest_rank_points",
    "current_rank",
    "ioc",
    "iso2",
    "country_name",
  ],
  searchOptions: { fuzzy: 0.2, prefix: true, boost: { display_name: 2 } },
});

export interface PlayerIndexData {
  players: Player[];
  latest_match_date: string | null;
  search: (query: string) => Player[];
}

// Deserialize the deploy-built payload. The index string is the documented
// MiniSearch.loadJSON input; search is bound to the deserialized instance.
export function deserializePlayerIndex(payload: {
  latest_match_date: string | null;
  players: Player[];
  index: string;
}): PlayerIndexData {
  const index = MiniSearch.loadJSON(payload.index, MINISEARCH_OPTS);
  return {
    players: payload.players,
    latest_match_date: payload.latest_match_date,
    search: (query: string): Player[] => {
      const q = query.trim();
      if (!q) return [];
      return index.search(q, { fuzzy: 0.2, prefix: true }).map((r) => ({
        player_id: r.id,
        display_name: r.display_name as string,
        matches_played: r.matches_played as number,
        latest_rank_points: r.latest_rank_points as number | undefined,
        current_rank: r.current_rank as number | null | undefined,
        ioc: r.ioc as string,
        iso2: r.iso2 as string,
        country_name: r.country_name as string,
      }));
    },
  };
}

export async function fetchPlayerIndex(): Promise<PlayerIndexData> {
  const manifestRes = await fetch(PLAYER_INDEX_MANIFEST);
  if (!manifestRes.ok) {
    throw new Error(`player index manifest: HTTP ${manifestRes.status}`);
  }
  const manifest = (await manifestRes.json()) as { path: string };
  const payloadRes = await fetch(manifest.path);
  if (!payloadRes.ok) {
    throw new Error(`player index payload: HTTP ${payloadRes.status}`);
  }
  return deserializePlayerIndex(await payloadRes.json());
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
