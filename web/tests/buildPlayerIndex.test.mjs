// Hermetic tests for the deploy-time player-index builder. Uses a temp
// fixture directory only: no network, no database, no repo state.

import assert from "node:assert";
import { createHash } from "node:crypto";
import { mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
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

  // Manifest points at the hashed payload and records the input hash.
  const manifest = JSON.parse(
    await readFile(path.join(dir, MANIFEST_NAME), "utf8"),
  );
  assert.strictEqual(manifest.directoryPath, `/${directoryFileName}`);
  assert.strictEqual(manifest.searchPath, `/${searchFileName}`);
  assert.strictEqual(
    manifest.sourceHash,
    createHash("sha256")
      .update(JSON.stringify({ players: PLAYERS_FIXTURE }))
      .digest("hex"),
  );

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

test("reuses the existing payload when the input is unchanged", async () => {
  const dir = await fixtureDir({ players: PLAYERS_FIXTURE });
  const first = await buildPlayerIndex(
    path.join(dir, "player-directory.json"),
    dir,
  );
  const manifestBefore = await readFile(path.join(dir, MANIFEST_NAME), "utf8");
  const searchBefore = await readFile(
    path.join(dir, first.searchFileName),
    "utf8",
  );

  // Rewrite the identical raw input (the first build consumed it), then build
  // again: the exact same bytes must skip MiniSearch indexing entirely.
  await writeFile(
    path.join(dir, "player-directory.json"),
    JSON.stringify({ players: PLAYERS_FIXTURE }),
  );
  const second = await buildPlayerIndex(
    path.join(dir, "player-directory.json"),
    dir,
  );

  assert.strictEqual(second.reused, true);
  assert.strictEqual(second.directoryFileName, first.directoryFileName);
  assert.strictEqual(second.searchFileName, first.searchFileName);
  // Reuse leaves the manifest and payloads untouched.
  assert.strictEqual(
    await readFile(path.join(dir, MANIFEST_NAME), "utf8"),
    manifestBefore,
  );
  assert.strictEqual(
    await readFile(path.join(dir, first.searchFileName), "utf8"),
    searchBefore,
  );
  // The raw input is still consumed so it never ships.
  assert.deepEqual(
    (await readdir(dir)).sort(),
    [MANIFEST_NAME, first.directoryFileName, first.searchFileName].sort(),
  );
});

test("rebuilds when the input changes", async () => {
  const dir = await fixtureDir({ players: PLAYERS_FIXTURE });
  const first = await buildPlayerIndex(
    path.join(dir, "player-directory.json"),
    dir,
  );

  await writeFile(
    path.join(dir, "player-directory.json"),
    JSON.stringify({
      players: [
        ...PLAYERS_FIXTURE,
        {
          player_id: "3",
          display_name: "Novak Djokovic",
          matches_played: 30,
          current_rank: 1,
          ioc: "SRB",
          iso2: "rs",
        },
      ],
    }),
  );
  const second = await buildPlayerIndex(
    path.join(dir, "player-directory.json"),
    dir,
  );

  assert.strictEqual(second.reused, false);
  assert.notStrictEqual(second.directoryFileName, first.directoryFileName);
  assert.notStrictEqual(second.searchFileName, first.searchFileName);
  const manifest = JSON.parse(
    await readFile(path.join(dir, MANIFEST_NAME), "utf8"),
  );
  assert.match(manifest.sourceHash, /^[0-9a-f]{64}$/);
  assert.strictEqual(manifest.directoryPath, `/${second.directoryFileName}`);
});

test("rebuilds when a payload is missing or corrupt", async () => {
  const dir = await fixtureDir({ players: PLAYERS_FIXTURE });
  const first = await buildPlayerIndex(
    path.join(dir, "player-directory.json"),
    dir,
  );

  // Missing payload -> rebuild restores it.
  await rm(path.join(dir, first.searchFileName));
  await writeFile(
    path.join(dir, "player-directory.json"),
    JSON.stringify({ players: PLAYERS_FIXTURE }),
  );
  const second = await buildPlayerIndex(
    path.join(dir, "player-directory.json"),
    dir,
  );
  assert.strictEqual(second.reused, false);
  assert.ok((await readdir(dir)).includes(second.searchFileName));

  // Corrupt payload (same name, wrong content) -> rebuild replaces it in place.
  await writeFile(path.join(dir, second.searchFileName), "garbage");
  await writeFile(
    path.join(dir, "player-directory.json"),
    JSON.stringify({ players: PLAYERS_FIXTURE }),
  );
  const third = await buildPlayerIndex(
    path.join(dir, "player-directory.json"),
    dir,
  );
  assert.strictEqual(third.reused, false);
  assert.strictEqual(third.searchFileName, second.searchFileName);
  const searchPayload = JSON.parse(
    await readFile(path.join(dir, third.searchFileName), "utf8"),
  );
  const index = MiniSearch.loadJSON(searchPayload.index, MINISEARCH_OPTS);
  assert.strictEqual(index.search("nadal", { prefix: true }).length, 1);
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
