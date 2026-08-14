import { useEffect, useState } from "react";
import ReactECharts from "../lib/echarts";
import type { EChartsOption } from "echarts";
import type {
  MatchHistory,
  PlayerProfile,
  RankHistory,
  ReturnMetrics,
  ServeMetrics,
  SimilarPlayersResponse,
} from "../api";
import { Card, Empty, Kicker, Loading, PlayerFlag } from "../components";
import {
  axisOption,
  baseChartOption,
  chartTokens,
  withAlpha,
} from "../lib/charts";
import { careerBestRank, yearAxisDomain } from "../lib/rankHistoryAxis";
import {
  ROUND_LABEL,
  TIER_LABEL,
  formatDelta,
  formatMetric,
  formatRate,
  scoreSegments,
} from "../lib/format";

const SURFACE_COLORS: Record<string, string> = {
  clay: "var(--clay)",
  grass: "var(--grass)",
  hard: "var(--ice)",
  carpet: "var(--text-dim)",
};

// Milliseconds in a year; the rank x-axis' minimum label interval, so the
// time-axis ladder never drops below year granularity.
const ONE_YEAR_MS = 365 * 24 * 60 * 60 * 1000;

// Same mobile breakpoint the app uses elsewhere (nav, H2H charts); the rank
// chart drops its axis title on narrow screens so the plot keeps its width.
function useIsNarrow(): boolean {
  const mq = "(max-width: 720px)";
  const [narrow, setNarrow] = useState(() => window.matchMedia(mq).matches);
  useEffect(() => {
    const media = window.matchMedia(mq);
    const fn = (e: MediaQueryListEvent) => setNarrow(e.matches);
    media.addEventListener("change", fn);
    return () => media.removeEventListener("change", fn);
  }, []);
  return narrow;
}

// Small up/down marker next to the current rank: the current rank number
// compared with the latest rank-history point one month back. A lower rank
// number is an improvement (up), a higher number a decline (down). The
// triangle and the place count both take the direction color. Rendered only
// when movement exists (unchanged or missing data never reaches this
// component).
function RankMove({ move, count }: { move: "up" | "down"; count: number }) {
  const label =
    move === "up" ? `Rank improved by ${count}` : `Rank declined by ${count}`;
  return (
    <span
      className={`rank-move ${move === "up" ? "is-up" : "is-down"}`}
      role="img"
      aria-label={label}
      title={label}
    >
      <svg aria-hidden="true" width="9" height="9" viewBox="0 0 9 9">
        {move === "up" ? (
          <path d="M4.5 1 L8.5 8 H0.5 Z" fill="currentColor" />
        ) : (
          <path d="M4.5 8 L8.5 1 H0.5 Z" fill="currentColor" />
        )}
      </svg>
      <span className="rank-move-count num" aria-hidden="true">
        {count}
      </span>
    </span>
  );
}

function Metric({
  label,
  value,
  delta,
  rate,
}: {
  label: string;
  value: number | null;
  delta: number | null;
  rate?: boolean;
}) {
  const fmt = rate ? formatRate(value) : formatMetric(value);
  const deltaText = formatDelta(delta, rate);
  const deltaTone =
    delta == null ? "" : delta > 0 ? " is-grass" : delta < 0 ? " is-down" : "";
  return (
    <div className="sr-metric">
      <span className="sr-label">{label}</span>
      <span className="sr-value num">{fmt}</span>
      {/* Delta cell always renders so the column stays aligned across rows;
          hidden (aria-hidden, invisible) when there is no benchmark. */}
      <span
        className={`sr-delta num${deltaTone} whitespace-pre`}
        aria-hidden={deltaText == null}
      >
        {deltaText ?? ""}
      </span>
    </div>
  );
}

const serveMetrics: { label: string; key: keyof ServeMetrics }[] = [
  { label: "1st serve points won", key: "first_serve_points_won_pct" },
  { label: "2nd serve points won", key: "second_serve_points_won_pct" },
  { label: "Break points saved", key: "break_points_saved_pct" },
  { label: "Aces per game", key: "aces_per_service_game" },
];
const serveRates = new Set(["aces_per_service_game"]);

const returnMetrics: { label: string; key: keyof ReturnMetrics }[] = [
  { label: "1st serve returns won", key: "first_serve_return_points_won_pct" },
  { label: "2nd serve returns won", key: "second_serve_return_points_won_pct" },
  { label: "Break points converted", key: "break_point_conversion_pct" },
  {
    label: "Break point chances per game",
    key: "break_point_opportunities_per_return_game",
  },
];
const returnRates = new Set(["break_point_opportunities_per_return_game"]);

export default function ProfileContent({
  profile,
  directoryRank,
  rankHistory,
  rankLoading,
  matchHistory,
  matchesLoading,
  similarQ,
  theme,
  onSelectSimilar,
}: {
  profile: PlayerProfile;
  directoryRank?: number | null;
  rankHistory: RankHistory | undefined;
  rankLoading: boolean;
  matchHistory: MatchHistory | undefined;
  matchesLoading: boolean;
  similarQ: {
    isLoading: boolean;
    isError: boolean;
    data: SimilarPlayersResponse | undefined;
  };
  theme: string;
  onSelectSimilar?: (playerId: string) => void;
}) {
  const t = chartTokens();
  const ax = axisOption(t);
  const narrow = useIsNarrow();

  const handednessLabel =
    profile.handedness === "R"
      ? "Right-handed"
      : profile.handedness === "L"
        ? "Left-handed"
        : "-";

  const bioRows: Array<[string, string]> = [
    ["Turned pro", profile.turned_pro ? String(profile.turned_pro) : "-"],
    ["Country", profile.country_name],
    ["Height", profile.height ? `${(profile.height / 100).toFixed(2)} m` : "-"],
    ["Handedness", handednessLabel],
    ["Backhand", profile.backhand ?? "-"],
  ];

  const rankPoints = (rankHistory?.rank_history ?? []).filter(
    (p) => p.rank != null,
  );
  // Year-start tick values pin the dotted grid lines (shown at every year
  // start on every width); labeled years are ECharts' own width-based choice.
  const rankAxis = yearAxisDomain(rankPoints.map((p) => p.rank_date));
  const careerBest = careerBestRank(rankPoints);
  // Final label is the profile's official rank; the directory rank only backs
  // it while the profile query is loading.
  const currentRank = profile.rank.current_rank ?? directoryRank ?? null;

  // Month-over-month rank movement: the current official rank compared with
  // the latest rank-history point at or before one month before today. A
  // lower rank number is an improvement (up arrow), a higher one a decline
  // (down arrow); unchanged, missing, or out-of-range values render nothing.
  // rank_points_trend is intentionally not used — it is the earliest-to-latest
  // career spread, not a one-month window.
  const today = new Date();
  const cutoffMs = Date.UTC(
    today.getFullYear(),
    today.getMonth() - 1,
    today.getDate(),
  );
  let monthAgoMs = -Infinity;
  let monthAgoRank: number | null = null;
  for (const p of rankPoints) {
    const ms = Date.parse(p.rank_date);
    if (!Number.isFinite(ms) || ms > cutoffMs || ms <= monthAgoMs) continue;
    monthAgoMs = ms;
    monthAgoRank = p.rank;
  }
  const rankMove: { move: "up" | "down"; count: number } | null =
    currentRank == null || monthAgoRank == null || monthAgoRank === currentRank
      ? null
      : {
          move: monthAgoRank > currentRank ? "up" : "down",
          count: Math.abs(monthAgoRank - currentRank),
        };

  const bioFacts = (
    <dl className="bio-grid">
      {bioRows.map(([label, value]) => (
        <div key={label}>
          <dt className="bio-label">{label}</dt>
          <dd className="bio-value">{value}</dd>
        </div>
      ))}
    </dl>
  );

  const similarFooter = (
    <div className="profile-footer">
      <span className="field-label">Similar playstyle to</span>
      {similarQ.isLoading ? (
        <span className="text-sm text-[var(--text-faint)]">Loading...</span>
      ) : similarQ.isError ||
        (similarQ.data?.similar_players ?? []).length === 0 ? null : (
        <span className="similar-inline">
          {similarQ.data!.similar_players.map((sp, i, arr) => (
            <span key={sp.player_id}>
              <button
                type="button"
                className="similar-link"
                onClick={() => onSelectSimilar?.(sp.player_id)}
              >
                {sp.display_name}
              </button>
              <span className="num text-xs text-[var(--text-faint)]">
                {(Number(sp.score) * 100).toFixed(1)}%
              </span>
              {i < arr.length - 1 && (
                <span className="text-[var(--line)]">·</span>
              )}
            </span>
          ))}
        </span>
      )}
    </div>
  );

  const matches = matchHistory?.matches ?? [];
  // Newest first; the API already caps the fetch at 20 (Home requests limit=20).
  // All fetched rows render; the wrap container scrolls natively (~5 rows
  // visible at once) so later matches stay reachable without a widget.
  const sortedMatches = [...matches].sort((a, b) =>
    b.match_date.localeCompare(a.match_date),
  );

  const tourneyTable = (
    <Card title="Recent matches">
      {matchesLoading ? (
        <Loading label="Loading matches" />
      ) : sortedMatches.length === 0 ? (
        <Empty message="No match history" />
      ) : (
        <div
          className="tourney-table-wrap"
          role="region"
          tabIndex={0}
          aria-label={`Recent matches for ${profile.display_name}`}
        >
          <table className="tourney-table">
            <colgroup>
              <col style={{ width: "10%" }} />
              <col style={{ width: "15%" }} />
              <col style={{ width: "6%" }} />
              <col style={{ width: "15%" }} />
              <col style={{ width: "25%" }} />
              <col style={{ width: "15%" }} />
              <col />
            </colgroup>
            <thead>
              <tr>
                <th>Date</th>
                <th>Tournament</th>
                <th className="tourney-th-c">Surface</th>
                <th className="tourney-th-c">Round</th>
                <th>Opponent</th>
                <th className="tourney-th-c">Score</th>
                <th className="tourney-th-c">Result</th>
              </tr>
            </thead>
            <tbody>
              {sortedMatches.map((m) => {
                const roundLabel =
                  ROUND_LABEL[m.round as keyof typeof ROUND_LABEL] ?? m.round;
                return (
                  <tr key={m.match_id}>
                    <td className="num" data-label="Date">
                      {m.match_date}
                    </td>
                    <td className="tourney-name" data-label="Tournament">
                      {m.tournament_name ||
                        (TIER_LABEL[m.tournament as keyof typeof TIER_LABEL] ??
                          m.tournament)}
                    </td>
                    <td className="tourney-td-c" data-label="Surface">
                      <span
                        className="surface-pill"
                        style={{
                          color: SURFACE_COLORS[m.surface] ?? "var(--text-dim)",
                          borderColor:
                            SURFACE_COLORS[m.surface] ?? "var(--text-dim)",
                        }}
                      >
                        {m.surface}
                      </span>
                    </td>
                    <td className="tourney-td-c" data-label="Round">
                      {roundLabel}
                    </td>
                    <td className="tourney-name" data-label="Opponent">
                      <span className="opponent-vs">VS</span>{" "}
                      {m.opponent_name ?? m.opponent_id}{" "}
                      <span className="opponent-rank pl-1 num">
                        {m.opponent_ranking != null
                          ? `#${m.opponent_ranking}`
                          : "N/A"}
                      </span>
                    </td>
                    <td className="tourney-td-c num" data-label="Score">
                      {scoreSegments(
                        m.score,
                        m.result === "won" ? "winner" : "loser",
                      )?.map((s, i) =>
                        s.bold ? (
                          <strong key={i}>{s.text}</strong>
                        ) : (
                          <span key={i}>{s.text}</span>
                        ),
                      ) ?? "—"}
                    </td>
                    <td className="tourney-td-c" data-label="Result">
                      <span
                        className={`result-text ${
                          m.result === "won" ? "is-win" : "is-loss"
                        }`}
                      >
                        {m.result === "won" ? "Win" : "Loss"}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );

  const rankOption: EChartsOption = {
    ...baseChartOption(t),
    tooltip: {
      ...baseChartOption(t).tooltip,
      trigger: "axis",
      formatter: (params: any) => {
        const d = new Date(params[0].value[0] as number);
        const date = `${d.getFullYear()}-${d.getMonth() + 1}-${d.getDate()}`;
        return `${date}<br/>Rank #${params[0].value[1]}`;
      },
    },
    // Desktop leaves room for the vertical "Rank" axis name; mobile hides the
    // name, so side margins shrink to just the y-axis value labels.
    grid: {
      left: narrow ? 36 : 50,
      right: narrow ? 8 : 24,
      top: 16,
      bottom: 28,
      containLabel: false,
    },
    xAxis: {
      type: "time",
      min: rankAxis?.min,
      max: rankAxis?.max,
      // Minimum granularity stays at one year so the time-axis ladder never
      // drops to sub-year labels; ECharts picks the tick count itself from
      // the chart width, guided by splitNumber.
      minInterval: ONE_YEAR_MS,
      splitNumber: narrow ? 5 : 8,
      axisLine: ax.axisLine,
      axisTick: { show: false, customValues: rankAxis?.ticks },
      axisLabel: {
        ...ax.axisLabel,
        margin: 8,
        // Year-only labels via the time-axis string template.
        formatter: "{yyyy}",
      },
      splitLine: { ...ax.splitLine, show: true },
    },
    yAxis: {
      type: "value",
      inverse: true,
      min: 1,
      max: 200,
      minInterval: 50,
      name: narrow ? "" : "Rank",
      nameLocation: "middle",
      nameGap: 36,
      nameTextStyle: { color: t.dim, fontSize: 11 },
      axisLabel: {
        ...ax.axisLabel,
        margin: 8,
        formatter: (value: number) => String(Math.round(value)),
      },
      splitLine: {
        ...ax.splitLine,
        lineStyle: { ...ax.splitLine.lineStyle, type: "dashed" },
      },
    },
    series: [
      {
        type: "line",
        data: rankPoints.map((p) => [p.rank_date, p.rank]),
        smooth: true,
        showSymbol: false,
        // Hardcourt blue token (--ice), the same color the surface legend and
        // current-rank stat use; green signals win data, not rank position.
        lineStyle: { color: t.ice, width: 2.5 },
        areaStyle: {
          color: {
            type: "linear",
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: withAlpha(t.ice, 0.28) },
              { offset: 1, color: withAlpha(t.ice, 0.02) },
            ],
          },
        },
        ...(careerBest
          ? {
              markPoint: {
                symbol: "circle",
                symbolSize: 8,
                itemStyle: {
                  color: t.ice,
                  borderColor: t.raised,
                  borderWidth: 2,
                },
                label: {
                  show: true,
                  position: "bottom",
                  distance: 8,
                  formatter: `Career High · #${careerBest.rank}`,
                  color: t.ice,
                  fontSize: 12,
                  fontWeight: 600,
                  textBorderColor: t.raised,
                  textBorderWidth: 3,
                },
                data: [
                  {
                    name: "Career best",
                    coord: [careerBest.rank_date, careerBest.rank],
                  },
                ],
              },
            }
          : {}),
      },
    ],
  };

  return (
    <div className="space-y-5">
      <section className="card">
        <Kicker>Player profile</Kicker>
        <div className="profile-head">
          <h1 className="page-title">
            <PlayerFlag
              iso2={profile.iso2}
              countryName={profile.country_name}
            />
            {profile.display_name}
          </h1>
          <div className="profile-head-stats">
            <div className="stat">
              <span className="stat-label">Matches</span>
              <span className="stat-num num">
                {profile.career.matches_played}
              </span>
            </div>
            {currentRank != null && (
              <div className="stat">
                <span className="stat-label">Current rank</span>
                <span className="stat-num is-ice num">
                  #{currentRank}
                  {rankMove && (
                    <RankMove move={rankMove.move} count={rankMove.count} />
                  )}
                </span>
              </div>
            )}
          </div>
        </div>
        <div className="profile-main">
          <div className="profile-bio">{bioFacts}</div>
          <div className="profile-summary">
            {profile.summary && (
              <p className="highlights-summary">{profile.summary}</p>
            )}
            {!profile.summary && (
              <p className="text-[var(--text-faint)] text-sm">
                No description available
              </p>
            )}
          </div>
        </div>
        {similarFooter}
      </section>

      <div className="stats-row">
        <Card title="On service">
          <div className="sr-list">
            {serveMetrics.map((m) => (
              <Metric
                key={m.key}
                label={m.label}
                value={profile.serve[m.key]}
                delta={profile.tour_comparisons[m.key]}
                rate={serveRates.has(m.key)}
              />
            ))}
          </div>
        </Card>
        <Card title="On return">
          <div className="sr-list">
            {returnMetrics.map((m) => (
              <Metric
                key={m.key}
                label={m.label}
                value={profile.return[m.key]}
                delta={profile.tour_comparisons[m.key]}
                rate={returnRates.has(m.key)}
              />
            ))}
          </div>
        </Card>
      </div>

      {tourneyTable}

      <Card title="Rank history">
        {rankLoading ? (
          <Loading label="Loading rank history" />
        ) : rankPoints.length === 0 ? (
          <Empty message="No rank history for this player" />
        ) : (
          <ReactECharts
            key={theme}
            option={rankOption}
            style={{ height: 320, width: "100%" }}
            className="chart-frame"
            aria-label={`ATP rank history for ${profile.display_name}`}
          />
        )}
      </Card>
    </div>
  );
}
