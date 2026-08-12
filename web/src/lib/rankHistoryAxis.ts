// Deterministic calendar-year axis for the rank-history chart.
//
// ECharts' time axis ignores `interval` and picks ticks from a fixed
// d3-style interval ladder (half-year, year, ...) based on the chart width
// and span. A 3-5 year span gets half-year ticks, which collapse to
// duplicate year labels; wider/narrower layouts skip years. We compute the
// yearly ticks ourselves and pin them onto the axis with `customValues`,
// which echarts renders exactly (see axisTickLabelBuilder.createAxisLabels /
// createAxisTicks).
//
// All timestamps are local midnights (built from date parts), matching how
// the axis label formatter reads `new Date(value).getFullYear()`.

const ISO_DATE = /^(\d{4})-(\d{2})-(\d{2})/;

/** Local-midnight timestamp for a YYYY-MM-DD string; null when malformed. */
function isoToTime(date: string): number | null {
  const m = ISO_DATE.exec(date);
  if (!m) return null;
  const time = new Date(+m[1], +m[2] - 1, +m[3]).getTime();
  return Number.isFinite(time) ? time : null;
}

export interface YearAxisDomain {
  /** Earliest data date (local midnight); no Jan-1 padding. */
  min: number;
  /** Last data date (local midnight). */
  max: number;
  /** Jan 1 of each year spanned by the data that lies within [min, max] (local midnight). */
  ticks: number[];
}

/** Yearly tick values + domain for YYYY-MM-DD rank dates; null when unusable. */
export function yearAxisDomain(dates: string[]): YearAxisDomain | null {
  const times = dates.map(isoToTime).filter((t): t is number => t != null);
  if (times.length === 0) return null;
  const firstTime = Math.min(...times);
  const lastTime = Math.max(...times);
  const firstYear = new Date(firstTime).getFullYear();
  const lastYear = new Date(lastTime).getFullYear();
  const ticks: number[] = [];
  for (let year = firstYear; year <= lastYear; year++) {
    const tick = new Date(year, 0, 1).getTime();
    // Drop Jan-1 ticks before the first data point: the axis starts at the
    // earliest actual rank date, so a leading year tick would hang in empty
    // space. A single-year domain can therefore yield zero ticks.
    if (tick >= firstTime) ticks.push(tick);
  }
  return { min: firstTime, max: lastTime, ticks };
}
