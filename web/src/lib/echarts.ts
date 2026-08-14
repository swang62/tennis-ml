import { createElement } from 'react'
import { use, init, dispose, getInstanceByDom } from 'echarts/core'
import { BarChart, LineChart, RadarChart } from 'echarts/charts'
import {
  AriaComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  RadarComponent,
  TooltipComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import ReactEChartsCore from 'echarts-for-react/esm/core'
import type { EChartsReactProps } from 'echarts-for-react/esm/types'

// Tree-shaken ECharts: register only the charts, components, and renderer the
// app actually uses (Profile rank line, H2H bar/radar/cumulative-wins line).
use([
  LineChart,
  BarChart,
  RadarChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  MarkLineComponent,
  AriaComponent,
  RadarComponent,
  CanvasRenderer,
])

// echarts-for-react's core only calls init/dispose/getInstanceByDom on the
// injected instance; pass the tree-shaken core's functions. Import from the
// ESM build (esm/core) — the CJS lib/core double-wraps its default export, so
// a default import resolves to a namespace object, not the component.
const echarts = { init, dispose, getInstanceByDom }

// Ready-to-use chart component with the tree-shaken instance injected; pages
// no longer pass an `echarts` prop or import the full bundle.
export default function ReactECharts(props: EChartsReactProps) {
  return createElement(ReactEChartsCore, { ...props, echarts })
}
