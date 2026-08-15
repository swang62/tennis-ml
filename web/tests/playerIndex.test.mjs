// Hermetic tests for the shared static player-index loader: the deploy-built
// search payload deserializes only through MiniSearch.loadJSON (never
// browser-side addAll) after the picker asks for it. No
// network, no DOM, no database.

import { test } from 'node:test'
import assert from 'node:assert'
import MiniSearch from 'minisearch'
import {
  MINISEARCH_OPTS,
  deserializePlayerSearch,
} from '../src/lib/playerIndex.ts'

const PLAYERS_FIXTURE = [
  {
    player_id: '1',
    display_name: 'Roger Federer',
    matches_played: 10,
    current_rank: 3,
    ioc: 'SUI',
    iso2: 'ch',
  },
  {
    player_id: '2',
    display_name: 'Rafael Nadal',
    matches_played: 20,
    current_rank: null,
    ioc: 'ESP',
    iso2: 'es',
  },
]

// Mirrors the deploy-time builder: serialize with the shared options so the
// string payload has exactly the shape the loader receives in the browser.
function makeSearchPayload(players) {
  const index = new MiniSearch(MINISEARCH_OPTS)
  index.addAll(players)
  return JSON.stringify(index)
}

test('deserializes the lazy search payload', async () => {
  const search = await deserializePlayerSearch(makeSearchPayload(PLAYERS_FIXTURE), PLAYERS_FIXTURE)
  const hits = search('nadal')
  assert.strictEqual(hits.length, 1)
  assert.strictEqual(hits[0].player_id, '2')
  assert.strictEqual(hits[0].display_name, 'Rafael Nadal')
  assert.strictEqual(hits[0].matches_played, 20)
  assert.strictEqual(hits[0].current_rank, null)
  assert.strictEqual(hits[0].ioc, 'ESP')
  assert.strictEqual(hits[0].iso2, 'es')
})

test('search is fuzzy/prefix over display names with empty-query short circuit', async () => {
  const search = await deserializePlayerSearch(makeSearchPayload(PLAYERS_FIXTURE), PLAYERS_FIXTURE)

  // Prefix + fuzzy: "fed" and a typo'd "federer" both resolve to Federer.
  assert.strictEqual(search('fed')[0].player_id, '1')
  assert.strictEqual(search('federer')[0].player_id, '1')
  assert.strictEqual(search('').length, 0)
  assert.strictEqual(search('   ').length, 0)
})
