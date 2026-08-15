// Hermetic tests for the shared static player-index loader: the deploy-built
// payload deserializes only through MiniSearch.loadJSON (never browser-side
// addAll) and exposes players, latest_match_date, and a working search. No
// network, no DOM, no database.

import { test } from 'node:test'
import assert from 'node:assert'
import MiniSearch from 'minisearch'
import {
  MINISEARCH_OPTS,
  deserializePlayerIndex,
} from '../src/lib/playerIndex.ts'

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

// Mirrors the deploy-time builder: serialize with the shared options so the
// string payload has exactly the shape the loader receives in the browser.
function makePayload(latest_match_date, players) {
  const index = new MiniSearch(MINISEARCH_OPTS)
  index.addAll(players)
  return { latest_match_date, players, index: JSON.stringify(index) }
}

test('deserializes the payload and exposes players, date, and search', () => {
  const data = deserializePlayerIndex(
    makePayload('2026-08-10', PLAYERS_FIXTURE),
  )

  assert.strictEqual(data.latest_match_date, '2026-08-10')
  assert.deepEqual(data.players, PLAYERS_FIXTURE)

  const hits = data.search('nadal')
  assert.strictEqual(hits.length, 1)
  assert.strictEqual(hits[0].player_id, '2')
  assert.strictEqual(hits[0].display_name, 'Rafael Nadal')
  assert.strictEqual(hits[0].matches_played, 20)
  assert.strictEqual(hits[0].current_rank, null)
})

test('search is fuzzy/prefix over display names with empty-query short circuit', () => {
  const data = deserializePlayerIndex(
    makePayload('2026-08-10', PLAYERS_FIXTURE),
  )

  // Prefix + fuzzy: "fed" and a typo'd "federer" both resolve to Federer.
  assert.strictEqual(data.search('fed')[0].player_id, '1')
  assert.strictEqual(data.search('federer')[0].player_id, '1')
  assert.strictEqual(data.search('').length, 0)
  assert.strictEqual(data.search('   ').length, 0)
})

test('a null latest_match_date passes through', () => {
  const data = deserializePlayerIndex(makePayload(null, PLAYERS_FIXTURE))
  assert.strictEqual(data.latest_match_date, null)
})
