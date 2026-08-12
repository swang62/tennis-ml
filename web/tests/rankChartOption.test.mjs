// Focused SSR assertion for the rank-history chart's decal artifact.
//
// `baseChartOption` enables aria decals (`decal.show: true`), and ECharts
// paints the default palette decal — a rotated (diagonal) dashed hatch — as
// the fill of the line series' area polygon (see aria visual handler +
// LineView polygon rendering). Profile.tsx builds the rank chart straight
// from baseChartOption, so the hatch is present there. Render the rank-chart
// option with ECharts' SVG SSR and assert the hatch appears together with the
// vertical year grid lines and yearly ticks, on an unpadded domain that
// starts at the earliest rank date. Runs with `node --test tests/`
// (Node >= 23.6 strips types from the imported .ts helpers natively).

import { test } from 'node:test'
import assert from 'node:assert'
import * as echarts from 'echarts/core'
import { LineChart } from 'echarts/charts'
import { GridComponent, LegendComponent, TooltipComponent, AriaComponent } from 'echarts/components'
import { SVGRenderer } from 'echarts/renderers'
import { axisOption, baseChartOption } from '../src/lib/charts.ts'
import { yearAxisDomain } from '../src/lib/rankHistoryAxis.ts'

echarts.use([LineChart, GridComponent, LegendComponent, TooltipComponent, AriaComponent, SVGRenderer])

const tokens = {
  theme: 'dark',
  text: '#e8e8e8',
  dim: '#9aa0aa',
  faint: '#666666',
  line: '#333333',
  clay: '#c98d63',
  grass: '#3fae7a',
  ice: '#5f9fc9',
  raised: '#1c1c22',
  inset: '#141419',
}

// Same shape as Profile.tsx builds for the rank chart: base options
// (including the aria decal) plus the chart-specific axis configuration.
function rankOption() {
  const t = tokens
  const ax = axisOption(t)
  const base = baseChartOption(t)
  const dates = ['2020-03-01', '2021-04-15', '2022-09-01']
  const rankAxis = yearAxisDomain(dates)
  return {
    ...base,
    tooltip: { ...base.tooltip, trigger: 'axis' },
    grid: { left: 50, right: 24, top: 16, bottom: 28, containLabel: false },
    xAxis: {
      type: 'time',
      min: rankAxis?.min,
      max: rankAxis?.max,
      axisLine: ax.axisLine,
      axisTick: { show: false, customValues: rankAxis?.ticks },
      axisLabel: {
        ...ax.axisLabel,
        margin: 8,
        customValues: rankAxis?.ticks,
        formatter: (value) => String(new Date(value).getFullYear()),
      },
      splitLine: { ...ax.splitLine, show: true },
    },
    yAxis: {
      type: 'value',
      inverse: true,
      min: 1,
      max: 200,
      minInterval: 50,
      splitLine: { ...ax.splitLine, lineStyle: { ...ax.splitLine.lineStyle, type: 'dashed' } },
    },
    series: [
      {
        type: 'line',
        data: dates.map((d, i) => [d, [90, 40, 12][i]]),
        smooth: true,
        showSymbol: false,
        lineStyle: { color: t.ice, width: 2.5 },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(95, 159, 201, 0.28)' },
              { offset: 1, color: 'rgba(95, 159, 201, 0.02)' },
            ],
          },
        },
      },
    ],
  }
}

function renderSvg(option) {
  const chart = echarts.init(null, null, {
    renderer: 'svg',
    ssr: true,
    width: 800,
    height: 320,
  })
  try {
    chart.setOption(option)
    return chart.renderToSVGString()
  } finally {
    chart.dispose()
  }
}

test('baseChartOption enables the aria decal that paints the hatch', () => {
  const base = baseChartOption(tokens)
  assert.equal(base.aria?.enabled, true)
  assert.equal(base.aria?.decal?.show, true)
})

test('SSR: default decal renders a diagonal hatch pattern over the area', () => {
  const svg = renderSvg(rankOption())
  const patterns = svg.match(/<pattern/g) ?? []
  assert.equal(patterns.length, 1, 'expected the aria decal hatch pattern')
  assert.match(svg, /patternTransform="rotate\([^"]+\)"/, 'expected a rotated (diagonal) hatch')
})

test('SSR: rank chart keeps the hatch, yearly grid lines/ticks, and unpadded domain', () => {
  const opt = rankOption()
  const svg = renderSvg(opt)

  // The hatch is inherited from baseChartOption's aria decal.
  const patterns = svg.match(/<pattern/g) ?? []
  assert.equal(patterns.length, 1, 'expected the aria decal hatch pattern')
  assert.match(svg, /patternTransform="rotate\([^"]+\)"/, 'expected a rotated (diagonal) hatch')

  // Dashed grid split lines (vertical year lines + horizontal rank lines).
  const dashed = (svg) => svg.match(/<path[^>]*stroke-dasharray[^>]*>/g) ?? []
  const vertical = (svg) =>
    dashed(svg).filter((p) => /M(\d+(?:\.\d+)?) (?:[\d.]+)L\1 /.test(p))
  assert.ok(vertical(svg).length > 0, 'expected vertical year grid lines')

  // Yearly tick labels render for years whose Jan 1 lies in the domain.
  // The domain starts at the first data date (2020-03-01), so 2020's Jan-1
  // tick predates it and is not rendered.
  const years = [...svg.matchAll(/>(\d{4})<\/text>/g)].map((m) => m[1])
  for (const year of ['2021', '2022']) {
    assert.ok(years.includes(year), `expected yearly tick label ${year}`)
  }
  assert.ok(!years.includes('2020'), 'no tick before the first data date')

  // No padded domain: the axis starts at the earliest rank date, not Jan 1.
  assert.equal(opt.xAxis.min, new Date(2020, 2, 1).getTime())
  assert.notEqual(opt.xAxis.min, new Date(2020, 0, 1).getTime())
})
