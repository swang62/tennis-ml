// Build-time serialization of the deploy-time player directory into a
// content-hashed static MiniSearch payload plus a tiny discovery manifest.
//
// Input:  public/player-directory.json  (written by src/flows/deploy.py)
// Output: public/player-directory.<sha256>.json (players for picker defaults)
//         public/player-search.<sha256>.json    (serialized MiniSearch index;
//                                                fetched only after search input)
//         public/player-index.manifest.json     (paths to both immutable assets)
//
// The manifest records a sha256 of the raw input, so when the exact directory
// JSON is unchanged and both hashed payloads still exist and match their names
// (a cheap corruption check), the expensive MiniSearch indexing is skipped and
// the cached payloads are reused as-is. Any input change, missing/corrupt
// payload, or missing/old manifest rebuilds from scratch.
//
// Runs inside the web image build (`node scripts/build-player-index.mjs`)
// before `vite build`, so Vite copies both generated files into dist/. The raw
// input is consumed (deleted) after serialization so it never ships. Failures
// abort the build: a web image must never bake a missing or stale directory.

import { createHash } from "node:crypto";
import { mkdir, readdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import MiniSearch from "minisearch";

export const MANIFEST_NAME = "player-index.manifest.json";
const HASHED_PAYLOAD_RE =
  /^(?:player-index|player-directory|player-search)\.([0-9a-f]{64})\.json$/;
const color =
  process.stdout.isTTY || process.env.COURTSIDE_COLOR === "1" ? "\x1b[35m" : "";
const reset = color ? "\x1b[0m" : "";

// Mirrors the current Home search options exactly; the consumer must pass
// compatible options to MiniSearch.loadJSON at runtime.
export const MINISEARCH_OPTS = Object.freeze({
  fields: ["display_name"],
  idField: "player_id",
  searchOptions: { fuzzy: 0.2, prefix: true, boost: { display_name: 2 } },
});

/** Byte length of `filePath` when its content matches the sha256 embedded in
 * its file name, else null (missing, replaced, or corrupt). Validates a cached
 * payload before reuse cheaply — no MiniSearch construction needed. */
async function readVerifiedPayload(outDir, filePath) {
  const match = filePath && HASHED_PAYLOAD_RE.exec(path.basename(filePath));
  if (!match) return null;
  try {
    const bytes = await readFile(path.join(outDir, filePath));
    return createHash("sha256").update(bytes).digest("hex") === match[1]
      ? bytes.length
      : null;
  } catch {
    return null;
  }
}

/** Build the hashed payload + manifest into `outDir` from `inputPath`.
 * Reuses the existing payloads when the input is unchanged (same sha256 as the
 * manifest) and both payload files verify; rebuilds otherwise. Deletes the raw
 * input afterwards (consumed, never shipped). Returns the payload file names. */
export async function buildPlayerIndex(inputPath, outDir) {
  const inputBytes = await readFile(inputPath);
  const sourceHash = createHash("sha256").update(inputBytes).digest("hex");

  let manifest = null;
  try {
    manifest = JSON.parse(
      await readFile(path.join(outDir, MANIFEST_NAME), "utf8"),
    );
  } catch {
    manifest = null;
  }
  const reuse =
    manifest &&
    manifest.sourceHash === sourceHash &&
    typeof manifest.directoryPath === "string" &&
    typeof manifest.searchPath === "string";
  if (reuse) {
    const directoryBytes = await readVerifiedPayload(
      outDir,
      manifest.directoryPath,
    );
    const searchBytes = await readVerifiedPayload(outDir, manifest.searchPath);
    if (directoryBytes !== null && searchBytes !== null) {
      await rm(inputPath, { force: true });
      return {
        directoryFileName: path.basename(manifest.directoryPath),
        searchFileName: path.basename(manifest.searchPath),
        payloadBytes: directoryBytes + searchBytes,
        reused: true,
      };
    }
  }

  const directory = JSON.parse(inputBytes.toString("utf8"));
  const players = directory.players;
  if (!Array.isArray(players)) {
    throw new Error(
      `invalid directory artifact: "players" must be an array in ${inputPath}`,
    );
  }
  const index = new MiniSearch(MINISEARCH_OPTS);
  index.addAll(players);

  const directoryPayload = { players };
  const directoryBytes = Buffer.from(JSON.stringify(directoryPayload), "utf8");
  const searchBytes = Buffer.from(
    JSON.stringify({ index: JSON.stringify(index) }),
    "utf8",
  );
  const directoryFileName = `player-directory.${createHash("sha256").update(directoryBytes).digest("hex")}.json`;
  const searchFileName = `player-search.${createHash("sha256").update(searchBytes).digest("hex")}.json`;

  await mkdir(outDir, { recursive: true });
  // Drop stale payloads from earlier builds so public/ never accumulates.
  for (const name of (await readdir(outDir)).filter((n) =>
    HASHED_PAYLOAD_RE.test(n),
  )) {
    await rm(path.join(outDir, name), { force: true });
  }
  await Promise.all([
    writeFile(path.join(outDir, directoryFileName), directoryBytes),
    writeFile(path.join(outDir, searchFileName), searchBytes),
    writeFile(
      path.join(outDir, MANIFEST_NAME),
      `${JSON.stringify({
        directoryPath: `/${directoryFileName}`,
        searchPath: `/${searchFileName}`,
        sourceHash,
      })}\n`,
    ),
  ]);
  // Consume the raw directory input so it is never copied into dist/.
  await rm(inputPath, { force: true });

  return {
    directoryFileName,
    searchFileName,
    payloadBytes: directoryBytes.length + searchBytes.length,
    reused: false,
  };
}

const isMain =
  process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const publicDir = path.resolve(
    path.dirname(fileURLToPath(import.meta.url)),
    "..",
    "public",
  );
  try {
    const { directoryFileName, searchFileName, payloadBytes, reused } =
      await buildPlayerIndex(
        path.join(publicDir, "player-directory.json"),
        publicDir,
      );
    console.log(
      `${color}[minisearch]${reset} ${reused ? "reused" : "built"} ${directoryFileName} + ${searchFileName} (${payloadBytes} bytes) + ${MANIFEST_NAME} in ${publicDir}`,
    );
  } catch (err) {
    console.error(
      `build-player-index failed: ${err instanceof Error ? err.message : err}`,
    );
    process.exitCode = 1;
  }
}
