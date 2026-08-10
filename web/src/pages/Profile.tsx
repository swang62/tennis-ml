import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";
import type {
  PlayerProfile,
  RankHistory,
  ReturnMetrics,
  ServeMetrics,
  SimilarPlayersResponse,
} from "../api";
import { Card, Empty, Kicker, Loading, ResultBadge } from "../components";
import {
  axisOption,
  baseChartOption,
  chartTokens,
  withAlpha,
} from "../lib/charts";
import { ROUND_LABEL, TIER_LABEL } from "../lib/format";

const SURFACE_COLORS: Record<string, string> = {
  clay: "var(--clay)",
  grass: "var(--grass)",
  hard: "var(--ice)",
  carpet: "var(--text-dim)",
};

type MetricUnit = "pct" | "rate";

// pct metrics are 0..1 fractions shown as XX.X%; rate metrics are per-game /
// per-point ratios shown as X.XX. Deltas are player minus tour benchmark.
function formatMetric(unit: MetricUnit, value: number | null): string {
  if (value == null) return "n/a";
  return unit === "pct"
    ? `${Math.round(value * 1000) / 10}%`
    : value.toFixed(2);
}

function formatDelta(unit: MetricUnit, delta: number | null): string | null {
  if (delta == null) return null;
  if (unit === "pct") {
    const pp = Math.round(delta * 1000) / 10;
    return `${pp > 0 ? "+" : ""}${pp}pp`;
  }
  return `${delta > 0 ? "+" : ""}${delta.toFixed(2)}`;
}

function Metric({
  label,
  value,
  delta,
  unit,
}: {
  label: string;
  value: number | null;
  delta: number | null;
  unit: MetricUnit;
}) {
  const deltaText = formatDelta(unit, delta);
  const deltaTone =
    delta == null ? "" : delta > 0 ? " is-grass" : delta < 0 ? " is-clay" : "";
  return (
    <div className="sr-metric">
      <span className="sr-label">{label}</span>
      <span className="sr-value-row">
        <span className="sr-value num">{formatMetric(unit, value)}</span>
        {deltaText != null && (
          <span className={`sr-delta num${deltaTone}`}>{deltaText}</span>
        )}
      </span>
    </div>
  );
}

const serveMetrics: { label: string; key: keyof ServeMetrics; unit: MetricUnit }[] = [
  { label: "First serve in", key: "first_serve_in_pct", unit: "pct" },
  { label: "Aces / first serve", key: "aces_per_first_serve", unit: "rate" },
  { label: "1st serve points won", key: "first_serve_points_won_pct", unit: "pct" },
  { label: "2nd serve points won", key: "second_serve_points_won_pct", unit: "pct" },
  { label: "Serve points won", key: "overall_serve_points_won_pct", unit: "pct" },
  { label: "Double faults / serve pt", key: "double_faults_per_serve_point", unit: "rate" },
  { label: "Aces / service game", key: "aces_per_service_game", unit: "rate" },
  { label: "Break points saved", key: "break_points_saved_pct", unit: "pct" },
];

const returnMetrics: { label: string; key: keyof ReturnMetrics; unit: MetricUnit }[] = [
  { label: "Return points won", key: "return_points_won_pct", unit: "pct" },
  { label: "1st serve return won", key: "first_serve_return_points_won_pct", unit: "pct" },
  { label: "2nd serve return won", key: "second_serve_return_points_won_pct", unit: "pct" },
  { label: "Break point conversion", key: "break_point_conversion_pct", unit: "pct" },
  { label: "BP opp. / return game", key: "break_point_opportunities_per_return_game", unit: "rate" },
];

export default function ProfileContent({
  profile,
  estimatedRank,
  rankHistory,
  rankLoading,
  matchHistory,
  matchesLoading,
  similarQ,
  theme,
  onSelectSimilar,
}: {
  profile: PlayerProfile;
  estimatedRank?: number | null;
  rankHistory: RankHistory | undefined;
  rankLoading: boolean;
  matchHistory: { matches: Array<any> } | undefined;
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

  const trend = profile.rank_points_trend;
  let trendBadge = null;
  if (trend && trend.delta !== 0) {
    const improved = trend.delta > 0;
    trendBadge = (
      <span className={`badge ${improved ? "badge-grass" : "badge-clay"}`}>
        {improved ? "▲" : "▼"} {improved ? "+" : ""}
        {trend.delta} pts
      </span>
    );
  }

  const handednessLabel =
    profile.handedness === "R"
      ? "Right-handed"
      : profile.handedness === "L"
        ? "Left-handed"
        : "-";

  const bioRows: Array<[string, string]> = [
    ["Turned pro", profile.turned_pro ? String(profile.turned_pro) : "-"],
    ["Birthplace", profile.birthplace ?? "-"],
    ["Height", profile.height ? `${(profile.height / 100).toFixed(2)} m` : "-"],
    ["Handedness", handednessLabel],
    ["Backhand", profile.backhand ?? "-"],
  ];

  const rankPoints = (rankHistory?.rank_history ?? []).filter(
    (p) => p.rank != null,
  );
  // Final label is the profile's materialized estimate; the directory estimate
  // only backs it while the profile query is loading.
  const currentRank = profile.rank.estimated_rank ?? estimatedRank ?? null;

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
  const sortedMatches = [...matches].sort((a, b) =>
    b.match_date.localeCompare(a.match_date),
  );
  const recentMatches = sortedMatches.slice(0, 5);

  const tourneyTable = (
    <Card title="Recent tournaments">
      {matchesLoading ? (
        <Loading label="Loading tournaments" />
      ) : recentMatches.length === 0 ? (
        <Empty message="No tournament history" />
      ) : (
        <div className="tourney-table-wrap">
          <table className="tourney-table">
            <thead>
              <tr>
                <th>Tournament</th>
                <th className="tourney-th-c">Round</th>
                <th className="tourney-th-c">Surface</th>
                <th className="tourney-th-c">Result</th>
                <th className="tourney-th-r">Date</th>
              </tr>
            </thead>
            <tbody>
              {recentMatches.map((m: any) => {
                const roundLabel =
                  ROUND_LABEL[m.round as keyof typeof ROUND_LABEL] ?? m.round;
                return (
                  <tr key={m.match_id}>
                    <td className="tourney-name">
                      {m.tournament_name ||
                        (TIER_LABEL[m.tournament as keyof typeof TIER_LABEL] ??
                          m.tournament)}
                    </td>
                    <td className="tourney-td-c">{roundLabel}</td>
                    <td className="tourney-td-c">
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
                    <td className="tourney-td-c">
                      <ResultBadge won={m.result === "won"} />
                    </td>
                    <td className="tourney-td-r num">{m.match_date}</td>
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
    grid: { left: 50, right: 24, top: 16, bottom: 28, containLabel: false },
    xAxis: {
      type: "time",
      axisLine: ax.axisLine,
      axisTick: { show: false },
      axisLabel: {
        ...ax.axisLabel,
        margin: 8,
        formatter: (value: number) => {
          const d = new Date(value);
          return `${d.getFullYear()}-${d.getMonth() + 1}-${d.getDate()}`;
        },
      },
      splitLine: { show: false },
    },
    yAxis: {
      type: "value",
      inverse: true,
      min: 1,
      minInterval: 1,
      name: "Rank",
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
        lineStyle: { color: t.grass, width: 2.5 },
        areaStyle: {
          color: {
            type: "linear",
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: withAlpha(t.grass, 0.28) },
              { offset: 1, color: withAlpha(t.grass, 0.02) },
            ],
          },
        },
      },
    ],
  };

  return (
    <div className="space-y-5">
      <section className="card">
        <Kicker>Player profile</Kicker>
        <div className="profile-head">
          <h1 className="page-title">{profile.display_name}</h1>
          {trendBadge}
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
                <span className="stat-num is-ice num">#{currentRank}</span>
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
        <Card title="Service">
          <div className="sr-list">
            {serveMetrics.map((m) => (
              <Metric
                key={m.key}
                label={m.label}
                value={profile.serve[m.key]}
                delta={profile.tour_comparisons[m.key]}
                unit={m.unit}
              />
            ))}
          </div>
        </Card>
        <Card title="Return">
          <div className="sr-list">
            {returnMetrics.map((m) => (
              <Metric
                key={m.key}
                label={m.label}
                value={profile.return[m.key]}
                delta={profile.tour_comparisons[m.key]}
                unit={m.unit}
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
