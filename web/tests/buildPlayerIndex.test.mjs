// Hermetic tests for the deploy-time player-index builder. Uses a temp
// fixture directory only: no network, no database, no repo state.

import assert from "node:assert";
import { mkdtemp, readdir, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import MiniSearch from "minisearch";
import {
  buildPlayerIndex,
  MANIFEST_NAME,
  MINISEARCH_OPTS,
} from "../scripts/build-player-index.mjs";

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

async function fixtureDir(directory) {
  const dir = await mkdtemp(path.join(os.tmpdir(), "player-index-test-"));
  await writeFile(
    path.join(dir, "player-directory.json"),
    JSON.stringify(directory),
  );
  return dir;
}

test("builds a content-hashed payload plus manifest and consumes the raw input", async () => {
  const dir = await fixtureDir({ players: PLAYERS_FIXTURE });

  const { directoryFileName, searchFileName } = await buildPlayerIndex(
    path.join(dir, "player-directory.json"),
    dir,
  );

  // One hashed payload + the manifest remain; the raw directory input is gone.
  assert.deepEqual(
    (await readdir(dir)).sort(),
    [MANIFEST_NAME, directoryFileName, searchFileName].sort(),
  );
  assert.match(directoryFileName, /^player-directory\.[0-9a-f]{64}\.json$/);
  assert.match(searchFileName, /^player-search\.[0-9a-f]{64}\.json$/);

  // Manifest points at the hashed payload.
  const manifest = JSON.parse(
    await readFile(path.join(dir, MANIFEST_NAME), "utf8"),
  );
  assert.strictEqual(manifest.directoryPath, `/${directoryFileName}`);
  assert.strictEqual(manifest.searchPath, `/${searchFileName}`);

  // Payload carries all three contract fields.
  const payload = JSON.parse(
    await readFile(path.join(dir, directoryFileName), "utf8"),
  );
  assert.deepEqual(payload.players, PLAYERS_FIXTURE);

  // The serialized index deserializes and searches with the Home options.
  const searchPayload = JSON.parse(
    await readFile(path.join(dir, searchFileName), "utf8"),
  );
  const index = MiniSearch.loadJSON(searchPayload.index, MINISEARCH_OPTS);
  const hit = index.search("nadal", { prefix: true });
  assert.strictEqual(hit.length, 1);
  assert.strictEqual(hit[0].id, "2");
  assert.strictEqual(hit[0].display_name, undefined);
});

test("is deterministic and content-hash sensitive", async () => {
  const dirA = await fixtureDir({ players: PLAYERS_FIXTURE });
  const dirB = await fixtureDir({ players: PLAYERS_FIXTURE });
  const dirC = await fixtureDir({
    players: [...PLAYERS_FIXTURE, { ...PLAYERS_FIXTURE[0], player_id: "3" }],
  });

  const a = await buildPlayerIndex(
    path.join(dirA, "player-directory.json"),
    dirA,
  );
  const b = await buildPlayerIndex(
    path.join(dirB, "player-directory.json"),
    dirB,
  );
  const c = await buildPlayerIndex(
    path.join(dirC, "player-directory.json"),
    dirC,
  );

  // Same input bytes -> same file name; any content change -> a new name.
  assert.strictEqual(a.directoryFileName, b.directoryFileName);
  assert.strictEqual(a.searchFileName, b.searchFileName);
  assert.notStrictEqual(a.directoryFileName, c.directoryFileName);
});

test("cleans stale payloads from earlier builds and rewrites the manifest", async () => {
  const dir = await fixtureDir({ players: PLAYERS_FIXTURE });
  const staleName = `player-search.${"a".repeat(64)}.json`;
  await writeFile(path.join(dir, staleName), "stale");
  await writeFile(path.join(dir, MANIFEST_NAME), "old manifest");

  const { directoryFileName, searchFileName } = await buildPlayerIndex(
    path.join(dir, "player-directory.json"),
    dir,
  );

  const files = await readdir(dir);
  assert.ok(!files.includes(staleName), "stale payload removed");
  assert.ok(files.includes(directoryFileName), "fresh directory present");
  assert.ok(files.includes(searchFileName), "fresh search index present");
  const manifest = JSON.parse(
    await readFile(path.join(dir, MANIFEST_NAME), "utf8"),
  );
  assert.strictEqual(manifest.directoryPath, `/${directoryFileName}`);
  assert.strictEqual(manifest.searchPath, `/${searchFileName}`);
});

test("rejects a missing or malformed directory artifact", async () => {
  const emptyDir = await mkdtemp(
    path.join(os.tmpdir(), "player-index-missing-"),
  );
  await assert.rejects(() =>
    buildPlayerIndex(path.join(emptyDir, "absent.json"), emptyDir),
  );

  const badDir = await fixtureDir({ players: "not-an-array" });
  await assert.rejects(() =>
    buildPlayerIndex(path.join(badDir, "player-directory.json"), badDir),
  );
});
