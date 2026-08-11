import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";
import type {
  PlayerProfile,
  RankHistory,
  ReturnMetrics,
  ServeMetrics,
  SimilarPlayersResponse,
} from "../api";
import {
  Card,
  Empty,
  Kicker,
  Loading,
  PlayerFlag,
  ResultBadge,
} from "../components";
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

function formatMetric(value: number | null): string {
  if (value == null) return "n/a";
  return `${Math.round(value * 1000) / 10}%`;
}

function formatRate(value: number | null): string {
  if (value == null) return "n/a";
  return (Math.round(value * 1000) / 1000).toFixed(2);
}

function formatDelta(delta: number | null, rate?: boolean): string | null {
  if (delta == null) return null;
  const change = rate
    ? (Math.round(Math.abs(delta) * 1000) / 1000).toFixed(2)
    : `${Math.round(Math.abs(delta) * 1000) / 10}%`;
  return delta > 0 ? `▲ ${change}` : delta < 0 ? `▼ ${change}` : change;
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
      <span className="sr-value-row">
        <span className="sr-value num">{fmt}</span>
        {deltaText != null && (
          <span className={`sr-delta num${deltaTone}`}>{deltaText}</span>
        )}
      </span>
    </div>
  );
}

const serveMetrics: { label: string; key: keyof ServeMetrics }[] = [
  { label: "1st serve points won", key: "first_serve_points_won_pct" },
  { label: "2nd serve points won", key: "second_serve_points_won_pct" },
  { label: "Aces per game", key: "aces_per_service_game" },
  { label: "Break points saved", key: "break_points_saved_pct" },
];
const serveRates = new Set(["aces_per_service_game"]);

const returnMetrics: { label: string; key: keyof ReturnMetrics }[] = [
  { label: "1st serve returns won", key: "first_serve_return_points_won_pct" },
  { label: "2nd serve returns won", key: "second_serve_return_points_won_pct" },
  { label: "Break point opportunities", key: "break_point_opportunities_per_return_game" },
  { label: "Break points converted", key: "break_point_conversion_pct" },
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
    ["Country", profile.country_name],
    ["Height", profile.height ? `${(profile.height / 100).toFixed(2)} m` : "-"],
    ["Handedness", handednessLabel],
    ["Backhand", profile.backhand ?? "-"],
  ];

  const rankPoints = (rankHistory?.rank_history ?? []).filter(
    (p) => p.rank != null,
  );
  const firstRankYear = Number(rankPoints[0]?.rank_date.slice(0, 4));
  const lastRankYear = Number(rankPoints.at(-1)?.rank_date.slice(0, 4));
  const rankYears =
    Number.isFinite(firstRankYear) && Number.isFinite(lastRankYear)
      ? Array.from({ length: lastRankYear - firstRankYear + 1 }, (_, index) =>
          String(firstRankYear + index),
        )
      : [];
  // Final label is the profile's official rank; the directory rank only backs
  // it while the profile query is loading.
  const currentRank = profile.rank.current_rank ?? directoryRank ?? null;

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
          return String(d.getFullYear());
        },
      },
      splitLine: { show: false },
    },
    yAxis: {
      type: "value",
      inverse: true,
      min: 1,
      max: 200,
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
        markLine: {
          silent: true,
          symbol: "none",
          label: { show: false },
          lineStyle: { color: t.line, type: "dashed" },
          data: rankYears.map((year) => ({ xAxis: `${year}-01-01` })),
        },
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
          <h1 className="page-title">
            <PlayerFlag
              iso2={profile.iso2}
              countryName={profile.country_name}
            />
            {profile.display_name}
          </h1>
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
        <Card title="On serve">
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
