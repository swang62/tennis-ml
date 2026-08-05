import { createColumnHelper, flexRender, getCoreRowModel, useReactTable } from '@tanstack/react-table'
import { useQuery } from '@tanstack/react-query'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import { getMatchHistory, getPlayerProfile, getRankHistory, type MatchRow } from '../api'
import { Card, Empty, ErrorBox, Loading, ResultBadge, StatBar, pct } from '../components'
import { profileRoute } from '../router'

const columnHelper = createColumnHelper<MatchRow>()

const matchColumns = [
  columnHelper.accessor('match_date', { header: 'Date' }),
  columnHelper.accessor('tournament', { header: 'Tournament' }),
  columnHelper.accessor('surface', { header: 'Surface' }),
  columnHelper.accessor('round', { header: 'Round' }),
  columnHelper.accessor('opponent_name', {
    header: 'Opponent',
    cell: (info) => info.getValue() ?? '-',
  }),
  columnHelper.accessor('result', {
    header: 'Result',
    cell: (info) => <ResultBadge won={info.getValue() === 'won'} />,
  }),
  columnHelper.accessor('ranking', { header: 'Rank', cell: (info) => info.getValue() ?? '-' }),
  columnHelper.accessor('aces', { header: 'Aces' }),
  columnHelper.accessor('double_faults', { header: 'DF' }),
  columnHelper.display({
    id: 'serve_points_won',
    header: 'Serve Pts',
    cell: (info) => {
      const m = info.row.original
      return `${m.first_serve_points_won + m.second_serve_points_won}/${m.total_serve_points}`
    },
  }),
]

export default function Profile() {
  const { playerId } = profileRoute.useParams()

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

  const matches = matchesQ.data?.matches ?? []
  const table = useReactTable({
    data: matches,
    columns: matchColumns,
    getCoreRowModel: getCoreRowModel(),
  })

  if (profileQ.isLoading) return <Loading label="Loading profile" />
  if (profileQ.isError) return <ErrorBox error={profileQ.error} onRetry={() => profileQ.refetch()} />

  const profile = profileQ.data
  if (!profile) return <Loading label="Loading profile" />
  const trend = profile.rank_points_trend
  const trendBadge =
    trend && trend.delta !== 0 ? (
      <span
        className={`rounded px-2 py-0.5 text-xs font-semibold ${
          trend.delta > 0 ? 'bg-emerald-100 text-emerald-700' : 'bg-rose-100 text-rose-700'
        }`}
      >
        {trend.delta > 0 ? '+' : ''}
        {trend.delta} rank pts
      </span>
    ) : null

  const bioRows: Array<[string, string]> = [
    ['Handedness', profile.handedness ?? '-'],
    ['Backhand', profile.backhand ?? '-'],
    ['Height', profile.height ? `${profile.height} cm` : '-'],
    ['Turned pro', profile.turned_pro ? String(profile.turned_pro) : '-'],
    ['Birthplace', profile.birthplace ?? '-'],
  ]

  const rankPoints = (rankQ.data?.rank_history ?? []).filter((p) => p.rank != null)
  const rankOption: EChartsOption = {
    tooltip: { trigger: 'axis' },
    grid: { left: 48, right: 16, top: 16, bottom: 28 },
    xAxis: { type: 'time' },
    yAxis: { type: 'value', inverse: true, name: 'Rank' },
    series: [
      {
        type: 'line',
        name: 'Rank',
        data: rankPoints.map((p) => [p.rank_date, p.rank]),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2 },
        areaStyle: { opacity: 0.08 },
      },
    ],
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-2xl font-bold">{profile.display_name}</h1>
          <span className="text-sm text-slate-400">{profile.player_id}</span>
          {trendBadge}
        </div>
        {profile.summary && <p className="mt-3 max-w-3xl text-sm text-slate-700">{profile.summary}</p>}
        <dl className="mt-4 grid grid-cols-2 gap-x-8 gap-y-1 text-sm sm:grid-cols-3 lg:grid-cols-5">
          {bioRows.map(([label, value]) => (
            <div key={label}>
              <dt className="text-xs uppercase tracking-wide text-slate-400">{label}</dt>
              <dd className="font-medium">{value}</dd>
            </div>
          ))}
        </dl>
      </div>

      {/* Career vs recent form */}
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Career form">
          <div className="space-y-3">
            <StatBar label="Overall win rate" value={profile.career.win_rate} />
            <StatBar label="First serve win pct" value={profile.career.first_serve_win_pct} />
            <StatBar label="Second serve win pct" value={profile.career.second_serve_win_pct} />
            <StatBar label="Serve win pct" value={profile.career.serve_win_pct} />
            <StatBar label="Break points saved pct" value={profile.career.break_points_saved_pct} />
          </div>
        </Card>
        <Card title="Recent form">
          {profile.recent_form ? (
            <div className="space-y-3">
              <StatBar label="Win rate, last 10 matches" value={profile.recent_form.last_10_win_rate} />
              <p className="text-xs text-slate-400">
                Snapshot date: {profile.recent_form.snapshot_date}
              </p>
              <p className="text-xs text-slate-500">
                Career matches played: {profile.career.matches_played}
              </p>
            </div>
          ) : (
            <Empty message="No recent form snapshot available" />
          )}
        </Card>
      </div>

      {/* All-time surface win rates */}
      <Card title="Surface win rate (all-time)">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-400">
              <th className="py-1 pr-4 font-medium">Surface</th>
              <th className="py-1 pr-4 font-medium">Matches</th>
              <th className="py-1 font-medium">Win rate</th>
            </tr>
          </thead>
          <tbody>
            {profile.surface_rates.map((s) => (
              <tr key={s.surface} className="border-b border-slate-100 last:border-0">
                <td className="py-2 pr-4 capitalize">{s.surface}</td>
                <td className="py-2 pr-4">{s.matches}</td>
                <td className="py-2">{s.matches === 0 ? 'n/a (n=0)' : pct(s.win_rate)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Rank chart */}
      <Card title="Rank history">
        {rankQ.isLoading ? (
          <Loading label="Loading rank history" />
        ) : rankQ.isError ? (
          <ErrorBox error={rankQ.error} onRetry={() => rankQ.refetch()} />
        ) : rankPoints.length === 0 ? (
          <Empty message="No rank history for this player" />
        ) : (
          <ReactECharts option={rankOption} style={{ height: 320 }} />
        )}
      </Card>

      {/* Matches table */}
      <Card title="Recent matches">
        {matchesQ.isLoading ? (
          <Loading label="Loading matches" />
        ) : matchesQ.isError ? (
          <ErrorBox error={matchesQ.error} onRetry={() => matchesQ.refetch()} />
        ) : matches.length === 0 ? (
          <Empty message="No match history for this player" />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                {table.getHeaderGroups().map((group) => (
                  <tr key={group.id} className="border-b border-slate-200 text-left">
                    {group.headers.map((header) => (
                      <th key={header.id} className="px-2 py-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
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
                  <tr key={row.id} className="border-b border-slate-100 last:border-0 hover:bg-slate-50">
                    {row.getVisibleCells().map((cell) => (
                      <td key={cell.id} className="px-2 py-1.5">
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
