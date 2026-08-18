// Build-time serialization of the deploy-time player directory into the two
// Vite source inputs the app imports, with a hash-only build-state manifest.
//
// Input:    data/deploy/player-directory.json  (written by src/flows/deploy.py)
// Output:   web/src/assets/generated/player-directory.json ({ players })
//           web/src/assets/generated/player-search.json    ({ index })
//           data/deploy/player-index.manifest.json         (hash-only state)
//
// Vite applies its own content hash to the two generated inputs; the manifest
// therefore records content, never filenames or paths. It carries the schema
// and version plus four sha256 hashes: the raw input bytes, the serialized
// MiniSearch options/schema, and each generated payload. Reuse is valid only
// when the source and options hashes match the manifest AND both generated
// files exist with bytes matching the manifest's directory/search hashes;
// otherwise both outputs are rebuilt and the manifest is rewritten last.
//
// Runs host-side (just deploy / just dev) before Vite starts. The raw input is
// retained under data/deploy/ — never deleted, so restaging identical bytes
// lets this builder reuse its payloads. Failures abort the build: Vite must
// never consume a missing or stale directory.

import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import MiniSearch from "minisearch";

export const MANIFEST_NAME = "player-index.manifest.json";
export const DIRECTORY_OUT = "player-directory.json";
export const SEARCH_OUT = "player-search.json";
const MANIFEST_SCHEMA = "player-index-manifest";
const MANIFEST_VERSION = 2;
const color =
  process.stdout.isTTY || process.env.COURTSIDE_COLOR === "1" ? "\x1b[35m" : "";
const reset = color ? "\x1b[0m" : "";

// Mirrors the current Home search options exactly; the consumer must pass
// compatible options to MiniSearch.loadJSON at runtime. Serialized to derive
// optionsHash, so a change here invalidates reuse and regenerates the index.
export const MINISEARCH_OPTS = Object.freeze({
  fields: ["display_name"],
  idField: "player_id",
  searchOptions: { fuzzy: 0.2, prefix: true, boost: { display_name: 2 } },
});

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

/** Read `filePath` and return its bytes when their sha256 equals `expected`,
 * else null (missing, replaced, or corrupt). Validates a cached payload before
 * reuse cheaply — no MiniSearch construction needed. */
async function readVerifiedFile(filePath, expected) {
  try {
    const bytes = await readFile(filePath);
    return sha256(bytes) === expected ? bytes : null;
  } catch {
    return null;
  }
}

/** Write `bytes` to `target` via a same-directory temp file and rename so a
 * crash never leaves a partially written output or manifest in place. */
async function atomicWrite(target, bytes) {
  const tmp = `${target}.tmp-${process.pid}`;
  await mkdir(path.dirname(target), { recursive: true });
  await writeFile(tmp, bytes);
  await rename(tmp, target);
}

/** Build the Vite inputs and manifest from `inputPath` (raw directory JSON).
 * Reuses the existing generated files when the manifest matches the current
 * source and options hashes and both files verify byte-for-byte; rebuilds both
 * otherwise, writing the manifest last. Never deletes the raw input. */
export async function buildPlayerIndex(
  inputPath,
  manifestPath,
  outDir,
  options = MINISEARCH_OPTS,
) {
  const inputBytes = await readFile(inputPath);
  const sourceHash = sha256(inputBytes);
  const optionsHash = sha256(Buffer.from(JSON.stringify(options), "utf8"));

  let manifest = null;
  try {
    manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  } catch {
    manifest = null;
  }
  const manifestMatches =
    manifest &&
    manifest.schema === MANIFEST_SCHEMA &&
    manifest.version === MANIFEST_VERSION &&
    typeof manifest.sourceHash === "string" &&
    manifest.sourceHash === sourceHash &&
    typeof manifest.optionsHash === "string" &&
    manifest.optionsHash === optionsHash &&
    typeof manifest.directoryHash === "string" &&
    typeof manifest.searchHash === "string";

  const directoryBytes = manifestMatches
    ? await readVerifiedFile(
        path.join(outDir, DIRECTORY_OUT),
        manifest.directoryHash,
      )
    : null;
  const searchBytes = manifestMatches
    ? await readVerifiedFile(path.join(outDir, SEARCH_OUT), manifest.searchHash)
    : null;
  if (directoryBytes !== null && searchBytes !== null) {
    return {
      directoryFileName: DIRECTORY_OUT,
      searchFileName: SEARCH_OUT,
      payloadBytes: directoryBytes.length + searchBytes.length,
      reused: true,
    };
  }

  const directory = JSON.parse(inputBytes.toString("utf8"));
  const players = directory.players;
  if (!Array.isArray(players)) {
    throw new Error(
      `invalid directory artifact: "players" must be an array in ${inputPath}`,
    );
  }
  const index = new MiniSearch(options);
  index.addAll(players);

  const newDirectoryBytes = Buffer.from(JSON.stringify({ players }), "utf8");
  const newSearchBytes = Buffer.from(
    JSON.stringify({ index: JSON.stringify(index) }),
    "utf8",
  );
  const directoryHash = sha256(newDirectoryBytes);
  const searchHash = sha256(newSearchBytes);

  await Promise.all([
    atomicWrite(path.join(outDir, DIRECTORY_OUT), newDirectoryBytes),
    atomicWrite(path.join(outDir, SEARCH_OUT), newSearchBytes),
  ]);
  await atomicWrite(
    manifestPath,
    Buffer.from(
      `${JSON.stringify(
        {
          schema: MANIFEST_SCHEMA,
          version: MANIFEST_VERSION,
          sourceHash,
          optionsHash,
          directoryHash,
          searchHash,
        },
        null,
        2,
      )}\n`,
      "utf8",
    ),
  );

  return {
    directoryFileName: DIRECTORY_OUT,
    searchFileName: SEARCH_OUT,
    payloadBytes: newDirectoryBytes.length + newSearchBytes.length,
    reused: false,
  };
}

const isMain =
  process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const webRoot = path.resolve(
    path.dirname(fileURLToPath(import.meta.url)),
    "..",
  );
  const deployDir = path.resolve(webRoot, "..", "data", "deploy");
  const generatedDir = path.join(webRoot, "src", "assets", "generated");
  try {
    const { directoryFileName, searchFileName, payloadBytes, reused } =
      await buildPlayerIndex(
        path.join(deployDir, "player-directory.json"),
        path.join(deployDir, MANIFEST_NAME),
        generatedDir,
      );
    console.log(
      `${color}[minisearch]${reset} ${reused ? "reused" : "built"} ${directoryFileName} + ${searchFileName} (${payloadBytes} bytes) in ${generatedDir}; ${MANIFEST_NAME} in ${deployDir}`,
    );
  } catch (err) {
    console.error(
      `build-player-index failed: ${err instanceof Error ? err.message : err}`,
    );
    process.exitCode = 1;
  }
}
