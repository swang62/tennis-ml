// Focused tests for the On serve / On return metric-card formatters.
// Percentage values and percentage deltas always carry one decimal place
// ("80.0%", "▲ 9.0%") so decimal points line up in the right-aligned value
// and delta columns; rates keep two-decimal precision. Runs with
// `node --test tests/` (Node >= 23.6 strips types from the imported .ts
// helpers natively); no test framework dependency.

import { test } from 'node:test'
import assert from 'node:assert'
import { formatDelta, formatMetric, formatRate } from '../src/lib/format.ts'

test('formatMetric always shows one decimal place', () => {
  assert.equal(formatMetric(0.8), '80.0%')
  assert.equal(formatMetric(0.552), '55.2%')
  assert.equal(formatMetric(0.575), '57.5%')
  assert.equal(formatMetric(0.5755), '57.6%')
})

test('formatMetric renders null as n/a', () => {
  assert.equal(formatMetric(null), 'n/a')
})

test('formatRate keeps two-decimal rate precision', () => {
  assert.equal(formatRate(0.8), '0.80')
  assert.equal(formatRate(0.3), '0.30')
  assert.equal(formatRate(0.577), '0.58')
})

test('formatRate renders null as n/a', () => {
  assert.equal(formatRate(null), 'n/a')
})

test('formatDelta percentages always show one decimal place', () => {
  assert.equal(formatDelta(0.09), '▲ 9.0%')
  assert.equal(formatDelta(0.057), '▲ 5.7%')
  assert.equal(formatDelta(0.008), '▲ 0.8%')
})

test('formatDelta negative deltas use the down arrow', () => {
  assert.equal(formatDelta(-0.057), '▼ 5.7%')
})

test('formatDelta zero has no arrow', () => {
  assert.equal(formatDelta(0), '0.0%')
})

test('formatDelta rate deltas keep two-decimal precision', () => {
  assert.equal(formatDelta(0.08, true), '▲ 0.08')
  assert.equal(formatDelta(-0.08, true), '▼ 0.08')
})

test('formatDelta renders null as null (empty delta cell)', () => {
  assert.equal(formatDelta(null), null)
  assert.equal(formatDelta(null, true), null)
})
