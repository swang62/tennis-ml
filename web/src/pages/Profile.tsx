import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import type { PlayerProfile, RankHistory, SimilarPlayersResponse } from '../api'
import { Card, Empty, Kicker, Loading, ResultBadge, pct } from '../components'
import { axisOption, baseChartOption, chartTokens, withAlpha } from '../lib/charts'
import { ROUND_LABEL, TIER_LABEL } from '../lib/format'

const SURFACE_COLORS: Record<string, string> = {
  clay: 'var(--clay)',
  grass: 'var(--grass)',
  hard: 'var(--ice)',
  carpet: 'var(--text-dim)',
}

export default function ProfileContent({
  profile,
  rankHistory,
  rankLoading,
  matchHistory,
  matchesLoading,
  similarQ,
  theme,
  onSelectSimilar,
}: {
  profile: PlayerProfile
  rankHistory: RankHistory | undefined
  rankLoading: boolean
  matchHistory: { matches: Array<any> } | undefined
  matchesLoading: boolean
  similarQ: { isLoading: boolean; isError: boolean; data: SimilarPlayersResponse | undefined }
  theme: string
  onSelectSimilar?: (playerId: string) => void
}) {
  const t = chartTokens()
  const ax = axisOption(t)

  const trend = profile.rank_points_trend
  let trendBadge = null
  if (trend && trend.delta !== 0) {
    const improved = trend.delta > 0
    trendBadge = (
      <span className={`badge ${improved ? 'badge-grass' : 'badge-clay'}`}>
        {improved ? '▲' : '▼'} {improved ? '+' : ''}
        {trend.delta} pts
      </span>
    )
  }

  const handednessLabel =
    profile.handedness === 'R' ? 'Right-handed' : profile.handedness === 'L' ? 'Left-handed' : '-'

  const bioRows: Array<[string, string]> = [
    ['Turned pro', profile.turned_pro ? String(profile.turned_pro) : '-'],
    ['Birthplace', profile.birthplace ?? '-'],
    ['Height', profile.height ? `${(profile.height / 100).toFixed(2)} m` : '-'],
    ['Handedness', handednessLabel],
    ['Backhand', profile.backhand ?? '-'],
  ]

  const rankPoints = (rankHistory?.rank_history ?? []).filter((p) => p.rank != null)
  const sortedRank = [...rankPoints].sort((a, b) => b.rank_date.localeCompare(a.rank_date))
  const currentRank = sortedRank.length > 0 ? sortedRank[0].rank : null

  const bioFacts = (
    <dl className="bio-grid">
      {bioRows.map(([label, value]) => (
        <div key={label}>
          <dt className="bio-label">{label}</dt>
          <dd className="bio-value">{value}</dd>
        </div>
      ))}
    </dl>
  )

  const similarFooter = (
    <div className="profile-footer">
      <span className="field-label">Similar players</span>
      {similarQ.isLoading ? (
        <span className="text-sm text-[var(--text-faint)]">Loading...</span>
      ) : similarQ.isError || (similarQ.data?.similar_players ?? []).length === 0 ? null : (
        <span className="similar-inline">
          {(similarQ.data!.similar_players).map((sp, i, arr) => (
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
              {i < arr.length - 1 && <span className="text-[var(--line)]">·</span>}
            </span>
          ))}
        </span>
      )}
    </div>
  )

  const matches = matchHistory?.matches ?? []
  const sortedMatches = [...matches].sort((a, b) => b.match_date.localeCompare(a.match_date))
  const recentMatches = sortedMatches.slice(0, 5)

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
                  ROUND_LABEL[m.round as keyof typeof ROUND_LABEL] ?? m.round
                return (
                  <tr key={m.match_id}>
                    <td className="tourney-name">
                      {m.tournament_name || (TIER_LABEL[m.tournament as keyof typeof TIER_LABEL] ?? m.tournament)}
                    </td>
                    <td className="tourney-td-c">{roundLabel}</td>
                    <td className="tourney-td-c">
                      <span
                        className="surface-pill"
                        style={{
                          color: SURFACE_COLORS[m.surface] ?? 'var(--text-dim)',
                          borderColor: SURFACE_COLORS[m.surface] ?? 'var(--text-dim)',
                        }}
                      >
                        {m.surface}
                      </span>
                    </td>
                    <td className="tourney-td-c">
                      <ResultBadge won={m.result === 'won'} />
                    </td>
                    <td className="tourney-td-r num">{m.match_date}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  )

  const rankOption: EChartsOption = {
    ...baseChartOption(t),
    tooltip: {
      ...baseChartOption(t).tooltip,
      trigger: 'axis',
      formatter: (params: any) => {
        const d = new Date(params[0].value[0] as number)
        const date = `${d.getFullYear()}-${d.getMonth() + 1}-${d.getDate()}`
        return `${date}<br/>Rank #${params[0].value[1]}`
      },
    },
    grid: { left: 50, right: 24, top: 16, bottom: 28, containLabel: false },
    xAxis: {
      type: 'time',
      axisLine: ax.axisLine,
      axisTick: { show: false },
      axisLabel: {
        ...ax.axisLabel,
        margin: 8,
        formatter: (value: number) => {
          const d = new Date(value)
          return `${d.getFullYear()}-${d.getMonth() + 1}-${d.getDate()}`
        },
      },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      inverse: true,
      min: 1,
      minInterval: 1,
      name: 'Rank',
      nameLocation: 'middle',
      nameGap: 36,
      nameTextStyle: { color: t.dim, fontSize: 11 },
      axisLabel: { ...ax.axisLabel, margin: 8, formatter: (value: number) => String(Math.round(value)) },
      splitLine: { ...ax.splitLine, lineStyle: { ...ax.splitLine.lineStyle, type: 'dashed' } },
    },
    series: [
      {
        type: 'line',
        data: rankPoints.map((p) => [p.rank_date, p.rank]),
        smooth: true,
        showSymbol: false,
        lineStyle: { color: t.grass, width: 2.5 },
        areaStyle: {
          color: {
            type: 'linear',
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
  }

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
              <span className="stat-num num">{profile.career.matches_played}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Career win rate</span>
              <span className="stat-num is-grass num">{pct(profile.career.win_rate)}</span>
            </div>
            {currentRank !== null && (
              <div className="stat">
                <span className="stat-label">Current rank</span>
                <span className="stat-num is-ice num">#{currentRank}</span>
              </div>
            )}
          </div>
        </div>
        <div className="profile-main">
          <div className="profile-bio">
            {bioFacts}
          </div>
          <div className="profile-summary">
            {profile.summary && (
              <p className="highlights-summary">{profile.summary}</p>
            )}
            {!profile.summary && (
              <p className="text-[var(--text-faint)] text-sm">No description available</p>
            )}
          </div>
        </div>
        {similarFooter}
      </section>

      <div className="stats-row">
        <Card title="Career all-time stats">
          <div className="career-stats-col">
            <div className="stat">
              <span className="stat-label">First serve win %</span>
              <span className="stat-num is-grass num">
                {pct(profile.career.first_serve_win_pct)}
              </span>
            </div>
            <div className="stat">
              <span className="stat-label">Second serve win %</span>
              <span className="stat-num is-grass num">
                {pct(profile.career.second_serve_win_pct)}
              </span>
            </div>
          </div>
          <div className="surface-section" style={{ marginTop: '1rem' }}>
            <span className="field-label">Surface breakdown</span>
            <div className="space-y-3" style={{ marginTop: '0.65rem' }}>
              {profile.surface_rates.length === 0 ? (
                <Empty message="No surface data" />
              ) : (
                profile.surface_rates.map((s) => (
                  <div key={s.surface}>
                    <div className="flex items-baseline justify-between gap-3 text-sm">
                      <span className="capitalize font-semibold">{s.surface}</span>
                      <span className="num text-xs text-[var(--text-faint)]">
                        {s.matches} {s.matches === 1 ? 'match' : 'matches'}
                      </span>
                    </div>
                    <div className="bar mt-1">
                      <div
                        className={`bar-fill ${s.surface === 'clay' ? 'is-clay' : s.surface === 'grass' ? 'is-grass' : 'is-ice'}`}
                        style={{
                          width:
                            s.win_rate == null || s.matches === 0
                              ? '0%'
                              : `${Math.min(100, s.win_rate * 100)}%`,
                        }}
                      />
                    </div>
                    <div className="mt-0.5 text-right text-xs">
                      {s.matches === 0 ? (
                        <span className="text-[var(--text-faint)]">n/a (n=0)</span>
                      ) : (
                        <span className="num font-semibold">{pct(s.win_rate)}</span>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </Card>

        {tourneyTable}
      </div>

      <Card title="Rank history">
        {rankLoading ? (
          <Loading label="Loading rank history" />
        ) : rankPoints.length === 0 ? (
          <Empty message="No rank history for this player" />
        ) : (
          <ReactECharts
            key={theme}
            option={rankOption}
            style={{ height: 320, width: '100%' }}
            className="chart-frame"
            aria-label={`ATP rank history for ${profile.display_name}`}
          />
        )}
      </Card>
    </div>
  )
}
