// Hermetic tests for the deploy-time player-index builder. Uses a temp
// fixture directory only: no network, no database, no repo state.

import { test } from 'node:test'
import assert from 'node:assert'
import { mkdtemp, mkdir, readFile, writeFile, readdir } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import MiniSearch from 'minisearch'
import {
  MANIFEST_NAME,
  MINISEARCH_OPTS,
  buildPlayerIndex,
} from '../scripts/build-player-index.mjs'

const PLAYERS_FIXTURE = [
  {
    player_id: '1',
    display_name: 'Roger Federer',
    matches_played: 10,
    latest_rank_points: 1000,
    current_rank: 3,
    ioc: 'SUI',
    iso2: 'ch',
    country_name: 'Switzerland',
  },
  {
    player_id: '2',
    display_name: 'Rafael Nadal',
    matches_played: 20,
    latest_rank_points: null,
    current_rank: null,
    ioc: 'ESP',
    iso2: 'es',
    country_name: 'Spain',
  },
]

async function fixtureDir(directory) {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'player-index-test-'))
  await writeFile(
    path.join(dir, 'player-directory.json'),
    JSON.stringify(directory),
  )
  return dir
}

test('builds a content-hashed payload plus manifest and consumes the raw input', async () => {
  const dir = await fixtureDir({ latest_match_date: '2026-08-10', players: PLAYERS_FIXTURE })

  const { fileName } = await buildPlayerIndex(path.join(dir, 'player-directory.json'), dir)

  // One hashed payload + the manifest remain; the raw directory input is gone.
  assert.deepEqual((await readdir(dir)).sort(), [MANIFEST_NAME, fileName].sort())
  assert.match(fileName, /^player-index\.[0-9a-f]{64}\.json$/)

  // Manifest points at the hashed payload.
  const manifest = JSON.parse(await readFile(path.join(dir, MANIFEST_NAME), 'utf8'))
  assert.strictEqual(manifest.path, `/${fileName}`)

  // Payload carries all three contract fields.
  const payload = JSON.parse(await readFile(path.join(dir, fileName), 'utf8'))
  assert.strictEqual(payload.latest_match_date, '2026-08-10')
  assert.deepEqual(payload.players, PLAYERS_FIXTURE)

  // The serialized index deserializes and searches with the Home options.
  const index = MiniSearch.loadJSON(payload.index, MINISEARCH_OPTS)
  const hit = index.search('nadal', { prefix: true })
  assert.strictEqual(hit.length, 1)
  assert.strictEqual(hit[0].id, '2')
  assert.strictEqual(hit[0].display_name, 'Rafael Nadal')
})

test('is deterministic and content-hash sensitive', async () => {
  const dirA = await fixtureDir({ latest_match_date: '2026-08-10', players: PLAYERS_FIXTURE })
  const dirB = await fixtureDir({ latest_match_date: '2026-08-10', players: PLAYERS_FIXTURE })
  const dirC = await fixtureDir({ latest_match_date: '2026-08-17', players: PLAYERS_FIXTURE })

  const a = await buildPlayerIndex(path.join(dirA, 'player-directory.json'), dirA)
  const b = await buildPlayerIndex(path.join(dirB, 'player-directory.json'), dirB)
  const c = await buildPlayerIndex(path.join(dirC, 'player-directory.json'), dirC)

  // Same input bytes -> same file name; any content change -> a new name.
  assert.strictEqual(a.fileName, b.fileName)
  assert.notStrictEqual(a.fileName, c.fileName)
})

test('cleans stale payloads from earlier builds and rewrites the manifest', async () => {
  const dir = await fixtureDir({ latest_match_date: '2026-08-10', players: PLAYERS_FIXTURE })
  const staleName = 'player-index.' + 'a'.repeat(64) + '.json'
  await writeFile(path.join(dir, staleName), 'stale')
  await writeFile(path.join(dir, MANIFEST_NAME), 'old manifest')

  const { fileName } = await buildPlayerIndex(path.join(dir, 'player-directory.json'), dir)

  const files = await readdir(dir)
  assert.ok(!files.includes(staleName), 'stale payload removed')
  assert.ok(files.includes(fileName), 'fresh payload present')
  const manifest = JSON.parse(await readFile(path.join(dir, MANIFEST_NAME), 'utf8'))
  assert.strictEqual(manifest.path, `/${fileName}`)
})

test('rejects a missing or malformed directory artifact', async () => {
  const emptyDir = await mkdtemp(path.join(os.tmpdir(), 'player-index-missing-'))
  await assert.rejects(() =>
    buildPlayerIndex(path.join(emptyDir, 'absent.json'), emptyDir),
  )

  const badDir = await fixtureDir({ latest_match_date: null, players: 'not-an-array' })
  await assert.rejects(() =>
    buildPlayerIndex(path.join(badDir, 'player-directory.json'), badDir),
  )
})