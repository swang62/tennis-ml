import { useQuery } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import { getMatchHistory, getPlayerProfile, getRankHistory, getSimilarPlayers } from '../api'
import { Card, Empty, ErrorBox, Kicker, Loading, ResultBadge, pct } from '../components'
import { profileRoute } from '../router'
import { useTheme } from '../theme'
import { axisOption, baseChartOption, chartTokens, withAlpha } from '../lib/charts'
import { ROUND_LABEL, TIER_LABEL } from '../lib/format'

const SURFACE_COLORS: Record<string, string> = {
  clay: 'var(--clay)',
  grass: 'var(--grass)',
  hard: 'var(--ice)',
  carpet: 'var(--text-dim)',
}

export default function Profile() {
  const { playerId } = profileRoute.useParams()
  const { theme } = useTheme()

  const profileQ = useQuery({
    queryKey: ['profile', playerId],
    queryFn: () => getPlayerProfile(playerId),
  })
  const rankQ = useQuery({
    queryKey: ['rank_history', playerId],
    queryFn: () => getRankHistory(playerId),
  })
  const matchesQ = useQuery({
    queryKey: ['match_history', playerId, 20],
    queryFn: () => getMatchHistory(playerId, 20),
  })
  const similarQ = useQuery({
    queryKey: ['similar_players', playerId],
    queryFn: () => getSimilarPlayers(playerId, 3),
    enabled: !!playerId,
  })

  const matches = matchesQ.data?.matches ?? []
  const sortedMatches = [...matches].sort((a, b) => b.match_date.localeCompare(a.match_date))
  const recentMatches = sortedMatches.slice(0, 5)

  if (profileQ.isLoading) return <Loading label="Loading profile" />
  if (profileQ.isError)
    return <ErrorBox error={profileQ.error} onRetry={() => profileQ.refetch()} knownIds={[playerId]} />

  const profile = profileQ.data
  if (!profile) return <Loading label="Loading profile" />

  const t = chartTokens()
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

  const bioRows: Array<[string, string]> = [
    ['Handedness', profile.handedness ?? '-'],
    ['Backhand', profile.backhand ?? '-'],
    ['Height', profile.height ? `${profile.height} cm` : '-'],
    ['Turned pro', profile.turned_pro ? String(profile.turned_pro) : '-'],
    ['Birthplace', profile.birthplace ?? '-'],
  ]

  const rankPoints = (rankQ.data?.rank_history ?? []).filter((p) => p.rank != null)
  const ax = axisOption(t)

  // ── Rank history chart (rank on left, date on bottom, no series label) ──
  const rankOption: EChartsOption = {
    ...baseChartOption(t),
    tooltip: {
      ...baseChartOption(t).tooltip,
      trigger: 'axis',
      formatter: '{b}<br/>Rank #{c}',
    },
    grid: { left: 50, right: 24, top: 16, bottom: 28, containLabel: false },
    xAxis: {
      type: 'time',
      axisLine: ax.axisLine,
      axisTick: { show: false },
      axisLabel: { ...ax.axisLabel, margin: 8 },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      inverse: true,
      name: 'Rank',
      nameLocation: 'middle',
      nameGap: 36,
      nameTextStyle: { color: t.dim, fontSize: 11 },
      axisLabel: { ...ax.axisLabel, margin: 8 },
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
              <Link
                to="/players/$playerId"
                params={{ playerId: sp.player_id }}
                className="similar-link"
              >
                {sp.display_name}
              </Link>
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

  const tourneyTable = (
    <Card title="Recent tournaments">
      {matchesQ.isLoading ? (
        <Loading label="Loading tournaments" />
      ) : matchesQ.isError ? (
        <ErrorBox error={matchesQ.error} onRetry={() => matchesQ.refetch()} knownIds={[playerId]} />
      ) : recentMatches.length === 0 ? (
        <Empty message="No tournament history" />
      ) : (
        <div className="tourney-table-wrap">
          <table className="tourney-table">
            <thead>
              <tr>
                <th>Tournament</th>
                <th>Round</th>
                <th>Surface</th>
                <th>Result</th>
                <th className="tourney-th-r">Date</th>
              </tr>
            </thead>
            <tbody>
              {recentMatches.map((m) => {
                const roundLabel =
                  ROUND_LABEL[m.round as keyof typeof ROUND_LABEL] ?? m.round
                return (
                  <tr key={m.match_id}>
                    <td className="tourney-name">
                      {m.tournament_name || (TIER_LABEL[m.tournament as keyof typeof TIER_LABEL] ?? m.tournament)}
                    </td>
                    <td className="tourney-round">{roundLabel}</td>
                    <td>
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
                    <td>
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

  return (
    <div className="space-y-5">
      {/* Hero + bio + similar footer — single card */}
      <section className="card">
        <Kicker>Player profile</Kicker>
        <div className="profile-head">
          <h1 className="page-title">{profile.display_name}</h1>
          {trendBadge}
        </div>
        <div className="profile-main">
          <div className="profile-summary">
            {profile.summary && (
              <p className="highlights-summary">{profile.summary}</p>
            )}
            <div className="profile-stats">
              <div className="stat">
                <span className="stat-label">Career win rate</span>
                <span className="stat-num is-grass num">{pct(profile.career.win_rate)}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Matches</span>
                <span className="stat-num num">{profile.career.matches_played}</span>
              </div>
            </div>
          </div>
          {bioFacts}
        </div>
        {similarFooter}
      </section>

      {/* Career all-time stats (left) + Recent tournaments (right) */}
      <div className="stats-row">
        <Card title="Career all-time stats">
          <div className="space-y-3.5">
            <div className="profile-stats" style={{ marginBottom: '0.5rem' }}>
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
            <div className="surface-section">
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
          </div>
        </Card>

        {tourneyTable}
      </div>

      {/* Rank history — full width at the bottom */}
      <Card title="Rank history">
        {rankQ.isLoading ? (
          <Loading label="Loading rank history" />
        ) : rankQ.isError ? (
          <ErrorBox error={rankQ.error} onRetry={() => rankQ.refetch()} knownIds={[playerId]} />
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
