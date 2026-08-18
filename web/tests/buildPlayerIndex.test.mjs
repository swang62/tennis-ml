// Hermetic tests for the host-side player-index builder. Uses temp fixture
// directories only: no network, no database, no repo state.

import assert from "node:assert";
import { createHash } from "node:crypto";
import {
  mkdir,
  mkdtemp,
  readdir,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import MiniSearch from "minisearch";
import {
  buildPlayerIndex,
  DIRECTORY_OUT,
  MANIFEST_NAME,
  MINISEARCH_OPTS,
  SEARCH_OUT,
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

const RAW_INPUT_NAME = "player-directory.json";

function rawInput(players = PLAYERS_FIXTURE) {
  return JSON.stringify({ players });
}

// Mirrors production layout: the raw input and manifest live together (data/
// deploy/), the two generated Vite inputs in a separate directory
// (web/src/assets/generated/).
async function fixtureDir() {
  const base = await mkdtemp(path.join(os.tmpdir(), "player-index-test-"));
  const inputDir = path.join(base, "input");
  const outDir = path.join(base, "generated");
  await mkdir(inputDir);
  await mkdir(outDir);
  const inputPath = path.join(inputDir, RAW_INPUT_NAME);
  await writeFile(inputPath, rawInput());
  return {
    base,
    inputDir,
    outDir,
    inputPath,
    manifestPath: path.join(inputDir, MANIFEST_NAME),
  };
}

function rawHash(raw = rawInput()) {
  return createHash("sha256").update(raw).digest("hex");
}

function optionsHash(options = MINISEARCH_OPTS) {
  return createHash("sha256").update(JSON.stringify(options)).digest("hex");
}

async function assertManifest(manifestPath, expected) {
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  assert.strictEqual(manifest.schema, "player-index-manifest");
  assert.strictEqual(manifest.version, 2);
  for (const key of [
    "sourceHash",
    "optionsHash",
    "directoryHash",
    "searchHash",
  ]) {
    assert.match(manifest[key], /^[0-9a-f]{64}$/);
  }
  // Hash-only build state: no filenames or paths are recorded.
  for (const key of Object.keys(manifest)) {
    assert.ok(!String(manifest[key]).includes("/"), `no path in ${key}`);
    assert.ok(
      !String(manifest[key]).includes(".json"),
      `no filename in ${key}`,
    );
  }
  for (const key of Object.keys(expected)) {
    if (expected[key] !== null) {
      assert.strictEqual(manifest[key], expected[key]);
    }
  }
  return manifest;
}

test("builds the two Vite inputs plus a hash-only manifest and retains the raw input", async () => {
  const { inputDir, outDir, inputPath, manifestPath } = await fixtureDir();

  const { directoryFileName, searchFileName, reused } = await buildPlayerIndex(
    inputPath,
    manifestPath,
    outDir,
  );

  assert.strictEqual(reused, false);
  assert.strictEqual(directoryFileName, DIRECTORY_OUT);
  assert.strictEqual(searchFileName, SEARCH_OUT);
  // Only the two fixed-name generated outputs live in the Vite inputs dir.
  assert.deepEqual(
    (await readdir(outDir)).sort(),
    [DIRECTORY_OUT, SEARCH_OUT].sort(),
  );
  // The raw input is retained (never consumed) alongside the manifest so
  // restaging identical bytes can reuse.
  assert.deepEqual(
    (await readdir(inputDir)).sort(),
    [MANIFEST_NAME, RAW_INPUT_NAME].sort(),
  );

  const directoryBytes = await readFile(path.join(outDir, DIRECTORY_OUT));
  const searchBytes = await readFile(path.join(outDir, SEARCH_OUT));
  await assertManifest(manifestPath, {
    sourceHash: rawHash(),
    optionsHash: optionsHash(),
    directoryHash: createHash("sha256").update(directoryBytes).digest("hex"),
    searchHash: createHash("sha256").update(searchBytes).digest("hex"),
  });

  // Directory output carries the normalized { players } payload.
  const payload = JSON.parse(directoryBytes);
  assert.deepEqual(payload.players, PLAYERS_FIXTURE);

  // The serialized index deserializes and searches with the Home options.
  const searchPayload = JSON.parse(searchBytes);
  const index = MiniSearch.loadJSON(searchPayload.index, MINISEARCH_OPTS);
  const hit = index.search("nadal", { prefix: true });
  assert.strictEqual(hit.length, 1);
  assert.strictEqual(hit[0].id, "2");
  assert.strictEqual(hit[0].display_name, undefined);
});

test("is deterministic and content-hash sensitive", async () => {
  const a = await fixtureDir();
  const b = await fixtureDir();
  const c = await fixtureDir();
  await writeFile(
    c.inputPath,
    rawInput([
      ...PLAYERS_FIXTURE,
      {
        player_id: "3",
        display_name: "Novak Djokovic",
        matches_played: 30,
        current_rank: 1,
        ioc: "SRB",
        iso2: "rs",
      },
    ]),
  );

  await buildPlayerIndex(a.inputPath, a.manifestPath, a.outDir);
  await buildPlayerIndex(b.inputPath, b.manifestPath, b.outDir);
  await buildPlayerIndex(c.inputPath, c.manifestPath, c.outDir);

  const readManifest = async (f) =>
    JSON.parse(await readFile(f.manifestPath, "utf8"));
  const manifestA = await readManifest(a);
  const manifestB = await readManifest(b);
  const manifestC = await readManifest(c);

  // Same input bytes -> identical payload hashes; any change -> new hashes.
  assert.strictEqual(manifestB.directoryHash, manifestA.directoryHash);
  assert.strictEqual(manifestB.searchHash, manifestA.searchHash);
  assert.notStrictEqual(manifestC.directoryHash, manifestA.directoryHash);
  assert.notStrictEqual(manifestC.searchHash, manifestA.searchHash);
});

test("reuses the existing outputs when the input and options are unchanged", async () => {
  const { outDir, inputPath, manifestPath } = await fixtureDir();
  const first = await buildPlayerIndex(inputPath, manifestPath, outDir);
  const manifestBefore = await readFile(manifestPath, "utf8");
  const directoryBefore = await readFile(
    path.join(outDir, DIRECTORY_OUT),
    "utf8",
  );
  const searchBefore = await readFile(path.join(outDir, SEARCH_OUT), "utf8");

  const second = await buildPlayerIndex(inputPath, manifestPath, outDir);

  assert.strictEqual(second.reused, true);
  assert.strictEqual(second.directoryFileName, first.directoryFileName);
  assert.strictEqual(second.searchFileName, first.searchFileName);
  // Reuse leaves the manifest and outputs untouched.
  assert.strictEqual(await readFile(manifestPath, "utf8"), manifestBefore);
  assert.strictEqual(
    await readFile(path.join(outDir, DIRECTORY_OUT), "utf8"),
    directoryBefore,
  );
  assert.strictEqual(
    await readFile(path.join(outDir, SEARCH_OUT), "utf8"),
    searchBefore,
  );
});

test("rebuilds when the raw input changes", async () => {
  const { outDir, inputPath, manifestPath } = await fixtureDir();
  await buildPlayerIndex(inputPath, manifestPath, outDir);
  const manifestA = JSON.parse(await readFile(manifestPath, "utf8"));

  await writeFile(
    inputPath,
    rawInput([
      ...PLAYERS_FIXTURE,
      {
        player_id: "3",
        display_name: "Novak Djokovic",
        matches_played: 30,
        current_rank: 1,
        ioc: "SRB",
        iso2: "rs",
      },
    ]),
  );
  const second = await buildPlayerIndex(inputPath, manifestPath, outDir);

  assert.strictEqual(second.reused, false);
  const manifestB = JSON.parse(await readFile(manifestPath, "utf8"));
  assert.notStrictEqual(manifestB.sourceHash, manifestA.sourceHash);
  assert.notStrictEqual(manifestB.directoryHash, manifestA.directoryHash);
  assert.notStrictEqual(manifestB.searchHash, manifestA.searchHash);
});

test("rebuilds when the MiniSearch options change", async () => {
  const { outDir, inputPath, manifestPath } = await fixtureDir();
  await buildPlayerIndex(inputPath, manifestPath, outDir);
  const manifestA = JSON.parse(await readFile(manifestPath, "utf8"));
  const searchBefore = await readFile(path.join(outDir, SEARCH_OUT), "utf8");

  // Changing the serialized schema (fields) invalidates reuse and produces a
  // different serialized index; runtime-only searchOptions cannot change the
  // serialized bytes, so the schema is what optionsHash must capture.
  const otherOptions = {
    fields: ["display_name", "ioc"],
    idField: "player_id",
    searchOptions: { fuzzy: 0.1, prefix: true },
  };
  const rebuilt = await buildPlayerIndex(
    inputPath,
    manifestPath,
    outDir,
    otherOptions,
  );

  assert.strictEqual(rebuilt.reused, false);
  const manifestB = JSON.parse(await readFile(manifestPath, "utf8"));
  assert.notStrictEqual(manifestB.optionsHash, manifestA.optionsHash);
  assert.notStrictEqual(manifestB.searchHash, manifestA.searchHash);
  // New serialized index reflects the new options and is still loadable.
  const searchPayload = JSON.parse(
    await readFile(path.join(outDir, SEARCH_OUT)),
  );
  assert.notStrictEqual(searchPayload.index, JSON.parse(searchBefore).index);
  const index = MiniSearch.loadJSON(searchPayload.index, otherOptions);
  assert.strictEqual(index.search("nadal", { prefix: true }).length, 1);

  // Reverting to the default options changes optionsHash and rebuilds the
  // index under the default schema.
  const reverted = await buildPlayerIndex(inputPath, manifestPath, outDir);
  assert.strictEqual(reverted.reused, false);
});

test("rebuilds both outputs when one is missing or corrupt", async () => {
  const { outDir, inputPath, manifestPath } = await fixtureDir();
  await buildPlayerIndex(inputPath, manifestPath, outDir);
  const manifestA = JSON.parse(await readFile(manifestPath, "utf8"));
  const searchBefore = await readFile(path.join(outDir, SEARCH_OUT), "utf8");

  // Missing output -> rebuild restores it.
  await rm(path.join(outDir, SEARCH_OUT));
  const second = await buildPlayerIndex(inputPath, manifestPath, outDir);
  assert.strictEqual(second.reused, false);
  assert.ok((await readdir(outDir)).includes(SEARCH_OUT));

  // Corrupt output (same name, wrong content) -> rebuild replaces it in place;
  // the regenerated bytes are identical to the original, so the hash is too.
  await writeFile(path.join(outDir, SEARCH_OUT), "garbage");
  const third = await buildPlayerIndex(inputPath, manifestPath, outDir);
  assert.strictEqual(third.reused, false);
  const manifestB = JSON.parse(await readFile(manifestPath, "utf8"));
  assert.strictEqual(manifestB.searchHash, manifestA.searchHash);
  assert.strictEqual(
    await readFile(path.join(outDir, SEARCH_OUT), "utf8"),
    searchBefore,
  );
  const searchPayload = JSON.parse(
    await readFile(path.join(outDir, SEARCH_OUT)),
  );
  const index = MiniSearch.loadJSON(searchPayload.index, MINISEARCH_OPTS);
  assert.strictEqual(index.search("nadal", { prefix: true }).length, 1);
  // The untouched directory output still matches the rebuilt manifest.
  const directoryBytes = await readFile(path.join(outDir, DIRECTORY_OUT));
  assert.strictEqual(
    createHash("sha256").update(directoryBytes).digest("hex"),
    manifestB.directoryHash,
  );
});

test("rebuilds when the manifest is missing, stale, or has wrong hashes", async () => {
  const { outDir, inputPath, manifestPath } = await fixtureDir();
  await buildPlayerIndex(inputPath, manifestPath, outDir);

  // Stale legacy manifest (old path-based schema) -> rebuild.
  await writeFile(
    manifestPath,
    JSON.stringify({ directoryPath: "/x.json", sourceHash: "old" }),
  );
  const fresh = await buildPlayerIndex(inputPath, manifestPath, outDir);
  assert.strictEqual(fresh.reused, false);
  await assertManifest(manifestPath, {
    sourceHash: rawHash(),
    optionsHash: optionsHash(),
    directoryHash: null,
    searchHash: null,
  });

  // Manifest hash lies about an output -> rebuild corrects it.
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  manifest.searchHash = "0".repeat(64);
  await writeFile(manifestPath, JSON.stringify(manifest));
  const corrected = await buildPlayerIndex(inputPath, manifestPath, outDir);
  assert.strictEqual(corrected.reused, false);
  await assertManifest(manifestPath, {
    sourceHash: rawHash(),
    optionsHash: optionsHash(),
    directoryHash: null,
    searchHash: null,
  });
});

test("rejects a missing or malformed directory artifact", async () => {
  const emptyDir = await mkdtemp(
    path.join(os.tmpdir(), "player-index-missing-"),
  );
  await assert.rejects(() =>
    buildPlayerIndex(
      path.join(emptyDir, "absent.json"),
      path.join(emptyDir, MANIFEST_NAME),
      emptyDir,
    ),
  );

  const bad = await fixtureDir();
  await writeFile(bad.inputPath, rawInput("not-an-array"));
  await assert.rejects(() =>
    buildPlayerIndex(bad.inputPath, bad.manifestPath, bad.outDir),
  );
});
