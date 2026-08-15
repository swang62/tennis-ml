// Build-time serialization of the deploy-time player directory into a
// content-hashed static MiniSearch payload plus a tiny discovery manifest.
//
// Input:  public/player-directory.json  (written by src/flows/deploy.py)
// Output: public/player-index.<sha256>.json     (serialized index + players +
//                                                latest_match_date; immutable)
//         public/player-index.manifest.json     ({"path": "/player-index.<hash>.json"})
//
// Runs inside the web image build (`node scripts/build-player-index.mjs`)
// before `vite build`, so Vite copies both generated files into dist/. The raw
// input is consumed (deleted) after serialization so it never ships. Failures
// abort the build: a web image must never bake a missing or stale directory.

import { createHash } from 'node:crypto'
import { mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import MiniSearch from 'minisearch'

export const MANIFEST_NAME = 'player-index.manifest.json'
const HASHED_PAYLOAD_RE = /^player-index\.[0-9a-f]{64}\.json$/

// Mirrors the current Home search options/store fields exactly; the consumer
// must pass compatible options to MiniSearch.loadJSON at runtime.
export const MINISEARCH_OPTS = Object.freeze({
  fields: ['display_name'],
  idField: 'player_id',
  storeFields: [
    'display_name',
    'matches_played',
    'latest_rank_points',
    'current_rank',
    'ioc',
    'iso2',
    'country_name',
  ],
  searchOptions: { fuzzy: 0.2, prefix: true, boost: { display_name: 2 } },
})

/** Build the hashed payload + manifest into `outDir` from `inputPath`.
 * Deletes the raw input afterwards (consumed, never shipped). Returns the
 * payload file name. */
export async function buildPlayerIndex(inputPath, outDir) {
  const directory = JSON.parse(await readFile(inputPath, 'utf8'))
  const players = directory.players
  if (!Array.isArray(players)) {
    throw new Error(`invalid directory artifact: "players" must be an array in ${inputPath}`)
  }
  const index = new MiniSearch(MINISEARCH_OPTS)
  index.addAll(players)

  const payload = {
    latest_match_date: directory.latest_match_date ?? null,
    players,
    // String form: the documented MiniSearch.loadJSON(json, options) input.
    index: JSON.stringify(index),
  }
  const bytes = Buffer.from(JSON.stringify(payload), 'utf8')
  const hash = createHash('sha256').update(bytes).digest('hex')
  const fileName = `player-index.${hash}.json`

  await mkdir(outDir, { recursive: true })
  // Drop stale payloads from earlier builds so public/ never accumulates.
  for (const name of (await readdir(outDir)).filter((n) => HASHED_PAYLOAD_RE.test(n))) {
    await rm(path.join(outDir, name), { force: true })
  }
  await Promise.all([
    writeFile(path.join(outDir, fileName), bytes),
    writeFile(
      path.join(outDir, MANIFEST_NAME),
      JSON.stringify({ path: `/${fileName}` }) + '\n',
    ),
  ])
  // Consume the raw directory input so it is never copied into dist/.
  await rm(inputPath, { force: true })

  return { fileName, payloadBytes: bytes.length }
}

const isMain =
  process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
if (isMain) {
  const publicDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'public')
  try {
    const { fileName, payloadBytes } = await buildPlayerIndex(
      path.join(publicDir, 'player-directory.json'),
      publicDir,
    )
    console.log(`Built ${fileName} (${payloadBytes} bytes) + ${MANIFEST_NAME} in ${publicDir}`)
  } catch (err) {
    console.error(`build-player-index failed: ${err instanceof Error ? err.message : err}`)
    process.exitCode = 1
  }
}