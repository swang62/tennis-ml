// Hermetic tests for the shared API-fetched player index: the in-memory
// MiniSearch fuzzy/prefix index is built from the fetched players (no
// serialized search asset, no loadJSON), and the shared lazy loader performs
// no asset fetch and is memoized per directory payload. No network, no DOM,
// no database.

import { assert, test } from "vitest";
import {
  buildPlayerSearch,
  createPlayerSearchLoader,
} from "../src/lib/playerIndex.ts";

const PLAYERS_FIXTURE = [
  {
    player_id: "1",
    display_name: "Roger Federer",
    matches_played: 10,
    current_rank: 3,
    ioc: "SUI",
    iso2: "ch",
  },
  {
    player_id: "2",
    display_name: "Rafael Nadal",
    matches_played: 20,
    current_rank: null,
    ioc: "ESP",
    iso2: "es",
  },
];

test("builds a working in-memory fuzzy/prefix search from fetched players", async () => {
  const search = await buildPlayerSearch(PLAYERS_FIXTURE);
  const hits = search("nadal");
  assert.strictEqual(hits.length, 1);
  assert.strictEqual(hits[0].player_id, "2");
  assert.strictEqual(hits[0].display_name, "Rafael Nadal");
  assert.strictEqual(hits[0].matches_played, 20);
  assert.strictEqual(hits[0].current_rank, null);
  assert.strictEqual(hits[0].ioc, "ESP");
  assert.strictEqual(hits[0].iso2, "es");
});

test("search is fuzzy/prefix over display names with empty-query short circuit", async () => {
  const search = await buildPlayerSearch(PLAYERS_FIXTURE);

  // Prefix + fuzzy: "fed" resolves to Federer, a typo'd "nadel" to Nadal.
  assert.strictEqual(search("fed")[0].player_id, "1");
  assert.strictEqual(search("federer")[0].player_id, "1");
  assert.strictEqual(search("nadel")[0].player_id, "2");
  assert.strictEqual(search("").length, 0);
  assert.strictEqual(search("   ").length, 0);
});

test("shared lazy loader builds once, fetches no asset, and memoizes", async () => {
  const fetches = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url) => {
    fetches.push(url);
    throw new Error("unexpected fetch");
  };
  try {
    const loadSearch = createPlayerSearchLoader(PLAYERS_FIXTURE);
    const first = loadSearch();
    const second = loadSearch();
    assert.strictEqual(first, second, "same memoized promise");
    const search = await first;
    await second;
    assert.strictEqual(fetches.length, 0, "no separate asset fetch");
    assert.strictEqual(search("nadal")[0].player_id, "2");
  } finally {
    globalThis.fetch = originalFetch;
  }
});
