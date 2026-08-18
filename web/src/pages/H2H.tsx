import { useMutation, useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import type { EChartsOption } from "echarts";
import { format } from "echarts/core";
import { useEffect, useState } from "react";
import {
  getHeadToHead,
  getPlayerProfile,
  type MatchRound,
  predictFromIds,
  type Surface,
  type TournamentTier,
} from "../api";
import {
  Card,
  Empty,
  ErrorBox,
  Loading,
  PlayerFlag,
  PlayerPicker,
} from "../components";
import { axisOption, baseChartOption, chartTokens } from "../lib/charts";
import ReactECharts from "../lib/echarts";
import {
  fairOdds,
  pct,
  ROUND_LABEL,
  sanitizeErrorMessage,
  scoreSegments,
  TIER_LABEL,
} from "../lib/format";
import { preferenceEdge } from "../lib/h2hOrientation";
import { usePlayerDirectory } from "../lib/playerIndex";
import { h2hRoute } from "../routes";
import { useTheme } from "../theme";

const SURFACES: Surface[] = ["clay", "grass", "hard", "carpet"];
const TOURNAMENT_TIERS: { value: TournamentTier; label: string }[] = [
  { value: "grand_slam", label: "Grand Slam" },
  { value: "masters", label: "Masters" },
  { value: "atp_500", label: "ATP 500" },
  { value: "atp_250", label: "ATP 250" },
  { value: "davis_cup", label: "Davis Cup" },
  { value: "atp_finals", label: "ATP Finals" },
  { value: "olympics", label: "Olympics" },
  { value: "professional", label: "Professional" },
];
const ROUNDS: { value: MatchRound; label: string }[] = [
  { value: "r128", label: "R128" },
  { value: "r64", label: "R64" },
  { value: "r32", label: "R32" },
  { value: "r16", label: "R16" },
  { value: "qf", label: "Quarterfinal" },
  { value: "sf", label: "Semifinal" },
  { value: "f", label: "Final" },
  { value: "rr", label: "Round Robin" },
];

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
const today = () => new Date().toLocaleDateString("en-CA");

// Chart side margins are fixed in px for desktop player-name labels; on
// narrow screens the plot area shrinks to almost nothing. Track the breakpoint
// so grid margins and axis labels can compact.
function useIsNarrow(mq = "(max-width: 720px)"): boolean {
  const [narrow, setNarrow] = useState(() => window.matchMedia(mq).matches);
  useEffect(() => {
    const media = window.matchMedia(mq);
    const fn = (e: MediaQueryListEvent) => setNarrow(e.matches);
    media.addEventListener("change", fn);
    return () => media.removeEventListener("change", fn);
  }, [mq]);
  return narrow;
}

// Latest non-null rank is this page's sole rank signal; the picker directory
// carries the same official rank as the profile view.
function lastRank(
  player: { current_rank?: number | null } | undefined,
): number | null {
  return player?.current_rank ?? null;
}

// Centered direct-comparison row.
interface MirrorRow {
  label: string;
  a: number | null;
  b: number | null;
  aText: string;
  bText: string;
  invert?: boolean;
  color?: string;
}

const SURFACE_COLORS: Record<string, string> = {
  clay: "var(--clay)",
  grass: "var(--grass)",
  hard: "var(--ice)",
  carpet: "var(--text-dim)",
};

export default function H2H() {
  const { theme } = useTheme();
  const narrow = useIsNarrow();
  const { playerA: searchPlayerA, playerB: searchPlayerB } =
    h2hRoute.useSearch();
  const navigate = h2hRoute.useNavigate();
  const [playerA, setPlayerA] = useState<string | null>(searchPlayerA ?? null);
  const [playerB, setPlayerB] = useState<string | null>(searchPlayerB ?? null);
  const [surface, setSurface] = useState<Surface>("hard");
  const [tournament, setTournament] = useState<TournamentTier | "">("");
  const [round, setRound] = useState<MatchRound | "">("");
  const [asOfDate, setAsOfDate] = useState(today);
  const [isIndoor, setIsIndoor] = useState<"" | 0 | 1>("");

  const directoryQ = usePlayerDirectory();

  const ready = playerA !== null && playerB !== null && playerA !== playerB;
  // Queries and prediction callbacks below run only once both ids are set
  // (enabled/rendered under the ready gate); the guard keeps ids non-null in
  // those paths without per-site assertions.
  const requireId = (id: string | null): string => {
    if (id === null) throw new Error("player not selected");
    return id;
  };
  const h2hQ = useQuery({
    queryKey: ["h2h", playerA, playerB],
    queryFn: () => getHeadToHead(requireId(playerA), requireId(playerB)),
    enabled: ready,
  });
  const profileAQ = useQuery({
    // Home's profile key so both pages share the cache; profile data is
    // immutable history, so match Home's Infinity staleness to avoid
    // refetching a cached profile on page switch.
    queryKey: ["profile", playerA],
    queryFn: () => getPlayerProfile(requireId(playerA)),
    enabled: ready,
    staleTime: Infinity,
    gcTime: Infinity,
  });
  const profileBQ = useQuery({
    queryKey: ["profile", playerB],
    queryFn: () => getPlayerProfile(requireId(playerB)),
    enabled: ready,
    staleTime: Infinity,
    gcTime: Infinity,
  });

  const predict = useMutation({ mutationFn: predictFromIds });
  const selectA = (id: string | null) => {
    setPlayerA(id);
    navigate({
      search: { playerA: id ?? undefined, playerB: playerB ?? undefined },
      replace: true,
    });
    predict.reset();
  };
  const selectB = (id: string | null) => {
    setPlayerB(id);
    navigate({
      search: { playerA: playerA ?? undefined, playerB: id ?? undefined },
      replace: true,
    });
    predict.reset();
  };
  const players = directoryQ.data?.players ?? [];
  const playerById = new Map(players.map((p) => [p.player_id, p]));
  // Display names only; an unknown player gets a neutral label, never the raw id.
  const name = (id: string | null) =>
    playerById.get(id ?? "")?.display_name ?? "Unknown player";

  useEffect(() => {
    if (!ready) return;
    for (const playerId of [playerA, playerB]) {
      const iso2 = players
        .find((p) => p.player_id === playerId)
        ?.iso2?.trim()
        .toLowerCase();
      if (iso2?.length === 2)
        new Image().src = `https://flagcdn.com/w40/${iso2}.png`;
    }
  }, [playerA, playerB, players, ready]);

  if (directoryQ.isLoading) return <Loading label="Loading players" />;
  if (directoryQ.isError)
    return (
      <ErrorBox error={directoryQ.error} onRetry={() => directoryQ.refetch()} />
    );

  const h2h = h2hQ.data;
  const meetings = h2h?.meetings ?? [];
  const summary = h2h?.summary;
  const sortedMeetings = [...meetings].sort((a, b) =>
    b.match_date.localeCompare(a.match_date),
  );

  const pred = predict.data;
  const orientA = pred ? pred.p_win : 0;
  const winnerP = pred ? Math.max(pred.p_win, 1 - pred.p_win) : 0;

  const rankOf = (id: string) =>
    lastRank(players.find((p) => p.player_id === id));

  const t = chartTokens();
  const ax = axisOption(t);
  // Winner takes the player color: grass if player A wins, clay otherwise.
  const winnerIsA = pred ? pred.predicted_winner === playerA : false;
  const winnerColor = winnerIsA ? t.grass : t.clay;
  const winnerName = winnerIsA ? name(playerA) : name(playerB);

  const compOption: EChartsOption | null = pred
    ? {
        ...baseChartOption(t),
        tooltip: {
          renderMode: "html",
          backgroundColor: "transparent",
          borderWidth: 0,
          padding: 0,
          extraCssText: "box-shadow: none; white-space: pre-line;",
          textStyle: { fontSize: 12 },
          trigger: "item",
          formatter: (params: unknown) => {
            const item = Array.isArray(params) ? params[0] : params;
            if (!item || typeof item !== "object") return "";
            const { value } = item as {
              componentType?: string;
              value?: unknown;
            };
            const edge = Number(value);
            const pA = ((1 - edge) / 2) * 100;
            const pB = 100 - pA;
            return `<span style="color:${t.grass};font-weight:700">${format.encodeHTML(name(playerA))}</span>: <span style="color:${t.grass};font-weight:700">${pA.toFixed(1)}%</span>\n<span style="color:${t.clay};font-weight:700">${format.encodeHTML(name(playerB))}</span>: <span style="color:${t.clay};font-weight:700">${pB.toFixed(1)}%</span>`;
          },
        },
        grid: {
          left: narrow ? 44 : 120,
          right: narrow ? 44 : 120,
          top: 40,
          bottom: 24,
          containLabel: false,
        },
        xAxis: {
          type: "value",
          min: -1,
          max: 1,
          axisLine: ax.axisLine,
          axisTick: {
            show: true,
            length: 4,
            customValues: [
              1, 0.8, 0.6, 0.4, 0.2, 0, -0.2, -0.4, -0.6, -0.8, -1,
            ],
            lineStyle: { color: t.line },
          },
          axisLabel: {
            ...ax.axisLabel,
            customValues: [0.8, 0.4, 0, -0.4, -0.8],
            formatter: (v: number) => {
              const edge = v as number;
              const sign = Math.round(Math.abs(edge) * 100);
              return edge === 0 ? "Even" : `+${sign}`;
            },
          },
          splitLine: ax.splitLine,
        },
        yAxis: {
          type: "category",
          data: ["Linear", "GBDT", "Neural Net"],
          axisLine: ax.axisLine,
          axisTick: { show: false },
          axisLabel: ax.axisLabel,
        },
        series: [
          {
            name: "",
            label: {
              show: false,
              color: t.text,
              fontSize: 12,
              formatter: (params) =>
                `${Math.round(Number(params.value) * 100)}`,
              position: "inside",
            },
            type: "bar",
            barWidth: "44%",
            data: [pred.p_linear, pred.p_gbdt, pred.p_nn].map((probability) => {
              const edge = preferenceEdge(probability);
              return {
                value: edge * 2,
                label: { position: edge <= 0 ? "left" : "right" },
              };
            }),
            itemStyle: {
              color: `${t.ice}B3`,
              borderRadius: 0,
            },
            markLine: {
              symbol: "none",
              lineStyle: { color: winnerColor, type: "solid", width: 2 },
              label: {
                color: winnerColor,
                fontSize: 12,
                formatter: winnerName,
                position: "end",
                rotate: 0,
              },
              data: [{ xAxis: preferenceEdge(pred.p_win) * 2 }],
            },
          },
        ],
      }
    : null;

  // Compare direct-meeting results and labeled current ranks only.
  const p1 = h2h ? name(h2h.player1_id) : "";
  const p2 = h2h ? name(h2h.player2_id) : "";
  const mirrorRows: MirrorRow[] = [];
  if (h2h && summary) {
    const r1 = rankOf(h2h.player1_id);
    const r2 = rankOf(h2h.player2_id);
    if (r1 != null || r2 != null) {
      mirrorRows.push({
        label: "Current rank",
        a: r1,
        b: r2,
        aText: r1 != null ? `#${r1}` : "-",
        bText: r2 != null ? `#${r2}` : "-",
        invert: true,
      });
    }
    mirrorRows.push({
      label: "All-time Wins",
      a: summary.player1_wins,
      b: summary.player2_wins,
      aText: String(summary.player1_wins),
      bText: String(summary.player2_wins),
    });
    for (const s of ["hard", "grass", "clay"] as const) {
      const list = meetings.filter((m) => m.surface === s);
      const w1 = list.filter((m) => m.player1_won).length;
      mirrorRows.push({
        label: cap(s),
        a: w1,
        b: list.length - w1,
        aText: String(w1),
        bText: String(list.length - w1),
        color: SURFACE_COLORS[s],
      });
    }
  }
  const mirrorSummary = mirrorRows
    .map((r) => `${r.label}: ${p1} ${r.aText} — ${p2} ${r.bText}`)
    .join(". ");

  return (
    <div className="space-y-5">
      <section className="page-head">
        <h1 className="page-title">Matchup Predictions</h1>
        <p className="page-sub">
          Pick two players for a model prediction and implied odds, then compare
          head-to-head statistics.
        </p>
      </section>

      {/* Players */}
      <section aria-label="Players" className="space-y-4">
        <div className="grid gap-4 sm:grid-cols-2">
          <PlayerPicker
            players={players}
            value={playerA}
            onChange={selectA}
            placeholder="Player A"
            exclude={playerB}
            searchLoader={directoryQ.data?.loadSearch}
            tone="grass"
            centered
          />
          <PlayerPicker
            players={players}
            value={playerB}
            onChange={selectB}
            placeholder="Player B"
            exclude={playerA}
            searchLoader={directoryQ.data?.loadSearch}
            tone="clay"
            centered
          />
        </div>
      </section>

      {/* Match predictor */}
      <section className="card pred-card">
        {!ready ? (
          <Empty message="Select two different players to predict" />
        ) : (
          <>
            <div className="pred-controls">
              <label className="field">
                <span className="field-label">Surface</span>
                <select
                  value={surface}
                  onChange={(e) => setSurface(e.target.value as Surface)}
                  className="select"
                >
                  {SURFACES.map((s) => (
                    <option key={s} value={s}>
                      {cap(s)}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span className="field-label">Venue</span>
                <select
                  value={isIndoor}
                  onChange={(e) =>
                    setIsIndoor(
                      e.target.value === ""
                        ? ""
                        : (Number(e.target.value) as 0 | 1),
                    )
                  }
                  className="select"
                >
                  <option value="">-</option>
                  <option value="0">Outdoor</option>
                  <option value="1">Indoor</option>
                </select>
              </label>
              <label className="field">
                <span className="field-label">Tournament</span>
                <select
                  value={tournament}
                  onChange={(e) =>
                    setTournament(e.target.value as TournamentTier | "")
                  }
                  className="select"
                >
                  <option value="">-</option>
                  {TOURNAMENT_TIERS.map((tier) => (
                    <option key={tier.value} value={tier.value}>
                      {tier.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span className="field-label">Round</span>
                <select
                  value={round}
                  onChange={(e) => setRound(e.target.value as MatchRound | "")}
                  className="select"
                >
                  <option value="">-</option>
                  {ROUNDS.map((matchRound) => (
                    <option key={matchRound.value} value={matchRound.value}>
                      {matchRound.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span className="field-label">Match date</span>
                <input
                  type="date"
                  value={asOfDate}
                  onChange={(e) => setAsOfDate(e.target.value)}
                  className="input"
                />
              </label>
            </div>

            <div className="pred-actions">
              <button
                type="button"
                onClick={() =>
                  predict.mutate({
                    player_id: requireId(playerA),
                    opponent_id: requireId(playerB),
                    surface,
                    ...(tournament ? { tournament } : {}),
                    ...(round ? { round } : {}),
                    ...(asOfDate ? { as_of_date: asOfDate } : {}),
                    ...(isIndoor !== "" ? { is_indoor: isIndoor } : {}),
                  })
                }
                disabled={predict.isPending}
                className="btn btn-primary btn-predict"
              >
                {predict.isPending ? "Predicting..." : "Predict winner"}
              </button>
            </div>

            {predict.isError && (
              <div className="mt-4 error-box">
                <p className="error-title">Prediction failed</p>
                <p className="error-msg">
                  {sanitizeErrorMessage(
                    predict.error instanceof Error
                      ? predict.error.message
                      : String(predict.error),
                    [playerA ?? "", playerB ?? ""],
                  )}
                </p>
              </div>
            )}

            {pred && compOption && (
              <div className="pred-result" aria-live="polite">
                <div className="pred-main">
                  <span className="pred-winner">
                    {name(pred.predicted_winner)} wins
                  </span>
                  <span className="pred-pct num" style={{ color: winnerColor }}>
                    +{pct(2 * winnerP - 1)}
                  </span>
                  <span className="pred-caption num">
                    {pct(orientA)} : {pct(1 - orientA)}
                  </span>
                </div>
                <div className="odds-row">
                  <div
                    className="odds"
                    style={
                      pred.predicted_winner === playerA
                        ? {
                            borderColor: winnerColor,
                          }
                        : undefined
                    }
                  >
                    <span className="odds-label">{name(playerA)}</span>
                    <span
                      className={`odds-num num ${orientA >= 0.5 ? "is-fav" : ""}`}
                    >
                      {fairOdds(orientA)}
                    </span>
                  </div>
                  <div
                    className="odds"
                    style={
                      pred.predicted_winner === playerB
                        ? {
                            borderColor: winnerColor,
                          }
                        : undefined
                    }
                  >
                    <span className="odds-label">{name(playerB)}</span>
                    <span
                      className={`odds-num num ${orientA < 0.5 ? "is-fav" : ""}`}
                    >
                      {fairOdds(1 - orientA)}
                    </span>
                  </div>
                </div>
                <p className="mt-2 text-center text-[0.65rem] text-(--text-faint)">
                  Decimal odds show total return per 1 unit staked. A price of{" "}
                  {fairOdds(orientA)} returns {fairOdds(orientA)} units,
                  including the stake, if {name(playerA)} wins.
                </p>
                <div className="mt-4">
                  <ReactECharts
                    key={theme}
                    option={compOption}
                    style={{ height: 250, width: "100%" }}
                    className="chart-frame"
                  />
                </div>
                <p className="mt-2 text-center text-[0.65rem] text-(--text-faint)">
                  Relative edge: left favors {name(playerA)}, right favors{" "}
                  {name(playerB)}
                </p>
              </div>
            )}
          </>
        )}
      </section>

      {ready && h2hQ.isLoading && <Loading label="Loading head-to-head" />}
      {ready && h2hQ.isError && (
        <ErrorBox
          error={h2hQ.error}
          onRetry={() => h2hQ.refetch()}
          knownIds={[playerA ?? "", playerB ?? ""]}
        />
      )}

      {ready && h2h && summary && (
        <>
          {/* Matchup comparison */}
          <div className="grid gap-5 lg:grid-cols-2">
            <Card title="Win comparison">
              <div className="mirror">
                <div className="mirror-head">
                  <Link
                    to="/"
                    search={{ player: h2h.player1_id }}
                    className="mirror-name mirror-link"
                  >
                    <PlayerFlag
                      iso2={playerById.get(h2h.player1_id)?.iso2}
                      countryName={profileAQ.data?.country_name}
                    />
                    {p1}
                  </Link>
                  <span className="mirror-vs">vs</span>
                  <Link
                    to="/"
                    search={{ player: h2h.player2_id }}
                    className="mirror-name mirror-link"
                  >
                    {p2}
                    <PlayerFlag
                      iso2={playerById.get(h2h.player2_id)?.iso2}
                      countryName={profileBQ.data?.country_name}
                    />
                  </Link>
                </div>
                {mirrorRows.map((row) => {
                  return (
                    <div
                      className={
                        row.label === "All-time Wins"
                          ? "mirror-row is-summary"
                          : "mirror-row"
                      }
                      key={row.label}
                    >
                      <div className="mirror-half is-left">
                        <span className="mirror-value num">{row.aText}</span>
                      </div>
                      <span
                        className="mirror-label"
                        style={row.color ? { color: row.color } : undefined}
                      >
                        {row.label}
                      </span>
                      <div className="mirror-half is-right">
                        <span className="mirror-value num">{row.bText}</span>
                      </div>
                    </div>
                  );
                })}
                <p className="sr-only">{mirrorSummary}</p>
              </div>
            </Card>
            <Card title="Strength comparison">
              {profileAQ.isLoading || profileBQ.isLoading ? (
                <Loading label="Loading player strengths" />
              ) : profileAQ.data && profileBQ.data ? (
                (() => {
                  const radarMetrics = [
                    [
                      "1st serves won",
                      profileAQ.data.serve.first_serve_points_won_pct,
                      profileBQ.data.serve.first_serve_points_won_pct,
                    ],
                    [
                      "1st returns won",
                      profileAQ.data.return.first_serve_return_points_won_pct,
                      profileBQ.data.return.first_serve_return_points_won_pct,
                    ],
                    [
                      "2nd serves won",
                      profileAQ.data.serve.second_serve_points_won_pct,
                      profileBQ.data.serve.second_serve_points_won_pct,
                    ],
                    [
                      "2nd returns won",
                      profileAQ.data.return.second_serve_return_points_won_pct,
                      profileBQ.data.return.second_serve_return_points_won_pct,
                    ],
                    [
                      "Breaks saved",
                      profileAQ.data.serve.break_points_saved_pct,
                      profileBQ.data.serve.break_points_saved_pct,
                    ],
                    [
                      "Breaks converted",
                      profileAQ.data.return.break_point_conversion_pct,
                      profileBQ.data.return.break_point_conversion_pct,
                    ],
                  ] as const;
                  const radarOption: EChartsOption = {
                    ...baseChartOption(t),
                    aria: { enabled: false },
                    tooltip: {
                      show: true,
                      trigger: "item",
                      formatter: (params: unknown) => {
                        if (!params || typeof params !== "object") return "";
                        const { color, name: seriesName } = params as {
                          color?: unknown;
                          name?: unknown;
                        };
                        return `<span style="color:${String(color ?? "")};font-weight:700">${format.encodeHTML(String(seriesName ?? ""))}</span>`;
                      },
                      renderMode: "html",
                      backgroundColor: "transparent",
                      borderWidth: 0,
                      padding: 0,
                      extraCssText: "box-shadow: none; white-space: pre-line;",
                      textStyle: { fontSize: 12 },
                    },
                    legend: {
                      top: 0,
                      textStyle: { color: t.dim, fontSize: 11 },
                    },
                    radar: {
                      center: ["50%", "55%"],
                      radius: narrow ? "53%" : "62%",
                      indicator: radarMetrics.map(([metric, a, b]) => ({
                        name: `{metric|${metric}}\n{grass|${Math.round((a ?? 0) * 100)}%} {clay|${Math.round((b ?? 0) * 100)}%}`,
                        max: 1,
                      })),
                      axisName: {
                        fontSize: 10,
                        rich: {
                          metric: { color: t.dim, lineHeight: 14 },
                          grass: { color: t.grass, fontWeight: "bold" },
                          clay: { color: t.clay, fontWeight: "bold" },
                        },
                      },
                      splitLine: { lineStyle: { color: t.line } },
                      splitArea: { areaStyle: { color: [t.inset] } },
                      axisLine: { lineStyle: { color: t.line } },
                    },
                    series: [
                      {
                        type: "radar",
                        label: { show: false },
                        emphasis: { label: { show: false } },
                        data: [
                          {
                            name: name(playerA),
                            value: radarMetrics.map(([, a]) => a ?? 0),
                            lineStyle: { color: t.grass, width: 2 },
                            itemStyle: { color: t.grass },
                            areaStyle: { color: `${t.grass}66` },
                          },
                          {
                            name: name(playerB),
                            value: radarMetrics.map(([, , b]) => b ?? 0),
                            lineStyle: { color: t.clay, width: 2 },
                            itemStyle: { color: t.clay },
                            areaStyle: { color: `${t.clay}66` },
                          },
                        ],
                      },
                    ],
                  };
                  return (
                    <ReactECharts
                      key={`radar-${theme}`}
                      option={radarOption}
                      style={{ height: 310, width: "100%" }}
                      className="chart-frame"
                      aria-label={`Strength comparison: ${name(playerA)} versus ${name(playerB)}`}
                    />
                  );
                })()
              ) : (
                <Empty message="Player strengths unavailable" />
              )}
            </Card>
          </div>

          <Card title="H2H History">
            {meetings.length === 0 ? (
              <Empty message="No prior meetings" />
            ) : (
              <>
                {(() => {
                  const chrono = [...meetings].sort((a, b) =>
                    a.match_date.localeCompare(b.match_date),
                  );
                  const dates = chrono.map((m) => m.match_date);
                  let p1Cum = 0;
                  let p2Cum = 0;
                  const p1Data: number[] = [];
                  const p2Data: number[] = [];
                  for (const m of chrono) {
                    if (m.player1_won) p1Cum++;
                    else p2Cum++;
                    p1Data.push(p1Cum);
                    p2Data.push(p2Cum);
                  }
                  const trendOption: EChartsOption = {
                    ...baseChartOption(t),
                    aria: { enabled: false },
                    legend: {
                      top: 0,
                      left: "center",
                      textStyle: { color: t.dim, fontSize: 11 },
                    },
                    tooltip: {
                      ...baseChartOption(t).tooltip,
                      trigger: "axis",
                      formatter: (params: unknown) => {
                        const items = Array.isArray(params)
                          ? (params as Array<{
                              name: string;
                              seriesName: string;
                              value: number;
                            }>)
                          : [];
                        return [
                          items[0]?.name ?? "",
                          ...items.map(
                            (s) =>
                              `${s.seriesName}: ${s.value} win${s.value === 1 ? "" : "s"}`,
                          ),
                        ].join("<br/>");
                      },
                    },
                    grid: {
                      left: narrow ? 20 : 56,
                      right: narrow ? 20 : 56,
                      top: 38,
                      bottom: 28,
                      containLabel: false,
                    },
                    xAxis: {
                      type: "category",
                      data: dates,
                      axisLine: ax.axisLine,
                      axisTick: { show: false },
                      axisLabel: {
                        ...ax.axisLabel,
                        margin: 8,
                        formatter: (value: string) => {
                          const [y, mo, d] = value.split("-");
                          return `${y}-${Number(mo)}-${Number(d)}`;
                        },
                      },
                    },
                    yAxis: {
                      type: "value",
                      minInterval: 1,
                      max: Math.max(p1Cum, p2Cum, 1),
                      name: narrow ? "" : "Cumulative Wins",
                      nameLocation: "middle",
                      nameGap: 36,
                      nameTextStyle: { color: t.dim, fontSize: 11 },
                      axisLabel: { ...ax.axisLabel, margin: 8 },
                      splitLine: ax.splitLine,
                    },
                    series: [
                      {
                        name: p1,
                        type: "line",
                        data: p1Data,
                        lineStyle: { color: t.grass, width: 2.5 },
                        itemStyle: { color: t.grass },
                        symbol: "circle",
                        symbolSize: 7,
                      },
                      {
                        name: p2,
                        type: "line",
                        data: p2Data,
                        lineStyle: { color: t.clay, width: 2.5 },
                        itemStyle: { color: t.clay },
                        symbol: "circle",
                        symbolSize: 7,
                      },
                    ],
                  };
                  return (
                    <div className="h2h-trend">
                      <ReactECharts
                        key={`trend-${theme}`}
                        option={trendOption}
                        style={{ height: 160, width: "100%" }}
                        className="chart-frame"
                        aria-label={`Cumulative head-to-head wins: ${p1} vs ${p2}`}
                      />
                    </div>
                  );
                })()}
                <section
                  className="h2h-meetings-wrap tourney-table-wrap"
                  aria-label={`Head-to-head match history: ${p1} versus ${p2}`}
                >
                  {sortedMeetings.map((m) => {
                    const aWon = m.player1_won;
                    const winnerName = aWon ? p1 : p2;
                    const tourney =
                      m.tournament_name ??
                      (m.tournament
                        ? (TIER_LABEL[
                            m.tournament as keyof typeof TIER_LABEL
                          ] ?? m.tournament)
                        : "");
                    const roundLabel = m.round
                      ? (ROUND_LABEL[m.round as keyof typeof ROUND_LABEL] ??
                        m.round)
                      : "";
                    return (
                      <div
                        key={`${m.match_date}-${m.winner_id}`}
                        className="meeting"
                      >
                        <div className="meeting-meta">
                          <span className="meeting-date">{m.match_date}</span>
                          <span className="meeting-surface">
                            <span
                              className="surface-pill"
                              style={{
                                color:
                                  SURFACE_COLORS[m.surface] ??
                                  "var(--text-dim)",
                                borderColor:
                                  SURFACE_COLORS[m.surface] ??
                                  "var(--text-dim)",
                              }}
                            >
                              {m.surface}
                            </span>
                          </span>
                          <span className="meeting-tourney">{tourney}</span>
                        </div>
                        <div className="meeting-result">
                          <span
                            className={`meeting-winner result-text ml-2 ${
                              aWon ? "is-win" : "is-loss"
                            }`}
                          >
                            {winnerName}
                          </span>
                          <span className="meeting-round-note">
                            {roundLabel ? (
                              <>
                                Won in{" "}
                                <span className="meeting-round-strong">
                                  {roundLabel}
                                </span>
                              </>
                            ) : (
                              "WON"
                            )}
                            <span className="meeting-sep" aria-hidden="true">
                              ·
                            </span>
                            {m.score ? (
                              <span className="meeting-score">
                                {scoreSegments(m.score, "winner")?.map((s) =>
                                  s.bold ? (
                                    <strong key={s.text}>{s.text}</strong>
                                  ) : (
                                    <span key={s.text}>{s.text}</span>
                                  ),
                                )}
                              </span>
                            ) : null}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </section>
              </>
            )}
          </Card>
        </>
      )}
    </div>
  );
}
