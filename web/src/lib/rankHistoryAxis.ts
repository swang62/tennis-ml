// Deterministic year-start tick marks for the rank-history chart's dotted
// grid lines, independent of ECharts' width-based label thinning.
//
// ECharts' time axis ignores `interval` and picks tick positions from a
// fixed d3-style interval ladder (half-year, year, ...) based on the chart
// width and span, so the year-start grid lines would drift off calendar-year
// boundaries. We compute the year-start ticks ourselves and pin them onto
// the axis with `axisTick.customValues`, which echarts renders exactly (see
// axisTickLabelBuilder.createAxisTicks). Labels are left to ECharts: the
// axis clamps its minimum label interval to one year and ECharts thins the
// year labels by available width.
//
// All timestamps are local midnights (built from date parts).

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

export interface CareerBestRank {
  rank: number;
  rank_date: string;
}

/** Lowest (best) rank, keeping the first date it was achieved. */
export function careerBestRank(
  points: Array<{ rank_date: string; rank: number }>,
): CareerBestRank | null {
  if (points.length === 0) return null;
  return points.reduce((best, point) =>
    point.rank < best.rank ||
    (point.rank === best.rank && point.rank_date < best.rank_date)
      ? point
      : best,
  );
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
