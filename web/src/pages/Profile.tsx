import { createColumnHelper, flexRender, getCoreRowModel, useReactTable } from '@tanstack/react-table'
import { useQuery } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import { getMatchHistory, getPlayerProfile, getRankHistory, getSimilarPlayers, type MatchRow } from '../api'
import { Card, Empty, ErrorBox, FormStrip, Kicker, Loading, ResultBadge, StatBar, pct } from '../components'
import { profileRoute } from '../router'
import { useTheme } from '../theme'
import { axisOption, baseChartOption, chartTokens, withAlpha } from '../lib/charts'
import { ROUND_LABEL, TIER_LABEL } from '../lib/format'

const columnHelper = createColumnHelper<MatchRow>()

const matchColumns = [
  columnHelper.accessor('match_date', {
    header: 'Date',
    cell: (info) => <span className="mono">{info.getValue()}</span>,
  }),
  columnHelper.accessor('tournament', {
    header: 'Tournament',
    cell: (info) => TIER_LABEL[info.getValue() as keyof typeof TIER_LABEL] ?? info.getValue(),
  }),
  columnHelper.accessor('surface', {
    header: 'Surface',
    cell: (info) => <span className="capitalize cell-dim">{info.getValue()}</span>,
  }),
  columnHelper.accessor('round', {
    header: 'Round',
    cell: (info) => ROUND_LABEL[info.getValue() as keyof typeof ROUND_LABEL] ?? info.getValue(),
  }),
  columnHelper.accessor('opponent_name', {
    header: 'Opponent',
    cell: (info) => info.getValue() ?? '-',
  }),
  columnHelper.accessor('result', {
    header: 'Result',
    cell: (info) => <ResultBadge won={info.getValue() === 'won'} />,
  }),
  columnHelper.accessor('ranking', {
    header: 'Opp. rank',
    cell: (info) => (info.getValue() == null ? '-' : `#${info.getValue()}`),
  }),
  columnHelper.accessor('aces', { header: 'Aces' }),
  columnHelper.accessor('double_faults', { header: 'DF' }),
  columnHelper.display({
    id: 'serve_points_won',
    header: 'Serve pts',
    cell: (info) => {
      const m = info.row.original
      return `${m.first_serve_points_won + m.second_serve_points_won}/${m.total_serve_points}`
    },
  }),
]

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
  const table = useReactTable({
    data: sortedMatches,
    columns: matchColumns,
    getCoreRowModel: getCoreRowModel(),
  })

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
      <span className={`badge ${improved ? 'badge-grass' : 'badge-ice'}`}>
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

  const form = sortedMatches.slice(0, 8).map((m) => m.result).reverse()

  const rankPoints = (rankQ.data?.rank_history ?? []).filter((p) => p.rank != null)
  const ax = axisOption(t)
  const rankOption: EChartsOption = {
    ...baseChartOption(t),
    tooltip: {
      ...baseChartOption(t).tooltip,
      trigger: 'axis',
      formatter: '{b}<br/>Rank #{c}',
    },
    grid: { left: 8, right: 20, top: 10, bottom: 4, containLabel: true },
    xAxis: {
      type: 'time',
      axisLine: ax.axisLine,
      axisTick: { show: false },
      axisLabel: ax.axisLabel,
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      inverse: true,
      name: 'Rank',
      nameTextStyle: { color: t.faint },
      axisLabel: ax.axisLabel,
      splitLine: ax.splitLine,
    },
    series: [
      {
        name: 'ATP rank',
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

  return (
    <div className="space-y-5">
      {/* Hero */}
      <section className="card">
        <Kicker>Player profile</Kicker>
        <div className="profile-head">
          <h1 className="page-title">{profile.display_name}</h1>
          {trendBadge}
        </div>
      </section>

      {/* Player highlights: summary block alongside the bio facts */}
      <section className="card">
        <h2 className="card-title">Player highlights</h2>
        {profile.summary ? (
          <div className="highlights">
            <p className="highlights-summary">{profile.summary}</p>
            {bioFacts}
          </div>
        ) : (
          bioFacts
        )}
      </section>

      {/* Form / career / surfaces — vertical editorial sequence */}
      <div className="space-y-5">
        <Card title="Similar players">
          {similarQ.isLoading ? (
            <Loading label="Loading similar players" />
          ) : similarQ.isError ? (
            <ErrorBox error={similarQ.error} onRetry={() => similarQ.refetch()} knownIds={[playerId]} />
          ) : (similarQ.data?.similar_players ?? []).length === 0 ? (
            <Empty message="No similar players found" />
          ) : (
            <ol className="similar-list">
              {(similarQ.data?.similar_players ?? []).map((sp) => (
                <li key={sp.player_id}>
                  <Link
                    to="/players/$playerId"
                    params={{ playerId: sp.player_id }}
                    className="similar-row"
                    aria-label={`View profile of ${sp.display_name}`}
                  >
                    <span className="similar-name">{sp.display_name}</span>
                    <span className="similar-score num">
                      {Number(sp.score).toFixed(3)}
                    </span>
                  </Link>
                </li>
              ))}
            </ol>
          )}
        </Card>

        <Card title="Form">
          {profile.recent_form ? (
            <div className="space-y-4">
              <div className="flex items-end justify-between gap-3">
                <div className="stat">
                  <span className="stat-label">Win rate · last 10</span>
                  <span className="stat-num is-grass num">
                    {pct(profile.recent_form.last_10_win_rate)}
                  </span>
                </div>
                <span className="mono text-xs text-[var(--text-faint)]">
                  as of {profile.recent_form.snapshot_date}
                </span>
              </div>
              {form.length > 0 && (
                <div>
                  <span className="field-label block">Last {form.length} results</span>
                  <FormStrip results={form} />
                </div>
              )}
              <p className="text-xs text-[var(--text-faint)]">
                {profile.career.matches_played} career matches
              </p>
            </div>
          ) : (
            <Empty message="No recent form snapshot available" />
          )}
        </Card>

        <Card title="Career serve & return">
          <div className="space-y-3.5">
            <StatBar label="First serve win pct" value={profile.career.first_serve_win_pct} />
            <StatBar label="Second serve win pct" value={profile.career.second_serve_win_pct} />
            <StatBar label="Serve win pct" value={profile.career.serve_win_pct} />
            <StatBar label="Break points saved pct" value={profile.career.break_points_saved_pct} />
          </div>
        </Card>

        <Card title="Surface win rate · all-time">
          {profile.surface_rates.length === 0 ? (
            <Empty message="No surface data" />
          ) : (
            <div className="space-y-3">
              {profile.surface_rates.map((s) => (
                <div key={s.surface}>
                  <div className="flex items-baseline justify-between gap-3 text-sm">
                    <span className="capitalize font-semibold">{s.surface}</span>
                    <span className="num text-xs text-[var(--text-faint)]">
                      {s.matches} {s.matches === 1 ? 'match' : 'matches'}
                    </span>
                  </div>
                  <div className="bar mt-1">
                    <div
                      className={`bar-fill ${s.win_rate != null && s.win_rate >= 0.5 ? 'is-grass' : 'is-ice'}`}
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
              ))}
            </div>
          )}
        </Card>
      </div>

      {/* Rank history */}
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

      {/* Recent matches */}
      <Card title="Recent matches">
        {matchesQ.isLoading ? (
          <Loading label="Loading matches" />
        ) : matchesQ.isError ? (
          <ErrorBox error={matchesQ.error} onRetry={() => matchesQ.refetch()} knownIds={[playerId]} />
        ) : matches.length === 0 ? (
          <Empty message="No match history for this player" />
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead>
                {table.getHeaderGroups().map((group) => (
                  <tr key={group.id}>
                    {group.headers.map((header) => (
                      <th key={header.id}>
                        {header.isPlaceholder
                          ? null
                          : flexRender(header.column.columnDef.header, header.getContext())}
                      </th>
                    ))}
                  </tr>
                ))}
              </thead>
              <tbody>
                {table.getRowModel().rows.map((row) => (
                  <tr key={row.id}>
                    {row.getVisibleCells().map((cell) => (
                      <td key={cell.id}>
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  )
}
