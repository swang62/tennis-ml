import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import {
  getHeadToHead,
  getPlayers,
  getRankHistory,
  predictFromIds,
  type MatchRound,
  type Surface,
  type TournamentTier,
} from '../api'
import { Card, Empty, ErrorBox, Loading, PlayerPicker, StatBar, pct } from '../components'

const SURFACES: Surface[] = ['clay', 'grass', 'hard', 'carpet']
const TOURNAMENT_TIERS: { value: TournamentTier; label: string }[] = [
  { value: 'grand_slam', label: 'Grand Slam' },
  { value: 'masters', label: 'Masters' },
  { value: 'atp_500', label: 'ATP 500' },
  { value: 'atp_250', label: 'ATP 250' },
  { value: 'davis_cup', label: 'Davis Cup' },
  { value: 'atp_finals', label: 'ATP Finals' },
  { value: 'olympics', label: 'Olympics' },
  { value: 'professional', label: 'Professional' },
]
const ROUNDS: { value: MatchRound; label: string }[] = [
  { value: 'r128', label: 'R128' },
  { value: 'r64', label: 'R64' },
  { value: 'r32', label: 'R32' },
  { value: 'r16', label: 'R16' },
  { value: 'qf', label: 'Quarterfinal' },
  { value: 'sf', label: 'Semifinal' },
  { value: 'f', label: 'Final' },
]

export default function H2H() {
  const [playerA, setPlayerA] = useState<string | null>(null)
  const [playerB, setPlayerB] = useState<string | null>(null)
  const [surface, setSurface] = useState<Surface>('hard')
  const [tournament, setTournament] = useState<TournamentTier | ''>('')
  const [round, setRound] = useState<MatchRound | ''>('')
  const [asOfDate, setAsOfDate] = useState('')
  const [indoor, setIndoor] = useState<'' | 0 | 1>('')

  const playersQ = useQuery({ queryKey: ['players'], queryFn: getPlayers })

  const ready = playerA !== null && playerB !== null && playerA !== playerB
  const h2hQ = useQuery({
    queryKey: ['h2h', playerA, playerB],
    queryFn: () => getHeadToHead(playerA!, playerB!),
    enabled: ready,
  })
  const rankAQ = useQuery({
    queryKey: ['rank_history', playerA],
    queryFn: () => getRankHistory(playerA!),
    enabled: ready,
  })
  const rankBQ = useQuery({
    queryKey: ['rank_history', playerB],
    queryFn: () => getRankHistory(playerB!),
    enabled: ready,
  })

  const predict = useMutation({ mutationFn: predictFromIds })

  const players = playersQ.data?.players ?? []
  const nameById = new Map(players.map((p) => [p.player_id, p.display_name]))
  const name = (id: string) => nameById.get(id) ?? id

  if (playersQ.isLoading) return <Loading label="Loading players" />
  if (playersQ.isError) return <ErrorBox error={playersQ.error} onRetry={() => playersQ.refetch()} />

  const h2h = h2hQ.data
  const meetings = h2h?.meetings ?? []
  const summary = h2h?.summary

  const surfaceCounts = new Map<string, number>()
  for (const m of meetings) {
    surfaceCounts.set(m.surface, (surfaceCounts.get(m.surface) ?? 0) + 1)
  }

  const rankPointsA = (rankAQ.data?.rank_history ?? []).filter((p) => p.rank != null)
  const rankPointsB = (rankBQ.data?.rank_history ?? []).filter((p) => p.rank != null)
  const rankOption: EChartsOption = {
    tooltip: { trigger: 'axis' },
    legend: { top: 0 },
    grid: { left: 48, right: 16, top: 32, bottom: 28 },
    xAxis: { type: 'time' },
    yAxis: { type: 'value', inverse: true, name: 'Rank' },
    series: [
      {
        name: playerA ? name(playerA) : 'Player A',
        type: 'line',
        data: rankPointsA.map((p) => [p.rank_date, p.rank]),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2 },
      },
      {
        name: playerB ? name(playerB) : 'Player B',
        type: 'line',
        data: rankPointsB.map((p) => [p.rank_date, p.rank]),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2 },
      },
    ],
  }

  const pred = predict.data

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Head-to-Head</h1>

      <div className="grid gap-4 sm:grid-cols-2">
        <PlayerPicker
          players={players}
          value={playerA}
          onChange={setPlayerA}
          placeholder="Player A"
        />
        <PlayerPicker
          players={players}
          value={playerB}
          onChange={setPlayerB}
          placeholder="Player B"
        />
      </div>

      {!ready && (
        <Empty message="Select two different players to compare" />
      )}

      {ready && h2hQ.isLoading && <Loading label="Loading head-to-head" />}
      {ready && h2hQ.isError && (
        <ErrorBox error={h2hQ.error} onRetry={() => h2hQ.refetch()} />
      )}

      {ready && h2h && (
        <div className="space-y-6">
          {/* Scoreboard */}
          <Card title="Scoreboard">
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <div className="text-xl font-bold">{name(h2h.player1_id)}</div>
                <div className="mt-1 text-3xl font-bold text-emerald-600">
                  {summary!.player1_wins}
                </div>
              </div>
              <div>
                <div className="text-sm text-slate-500">Meetings</div>
                <div className="mt-1 text-3xl font-bold">{summary!.meetings}</div>
              </div>
              <div>
                <div className="text-xl font-bold">{name(h2h.player2_id)}</div>
                <div className="mt-1 text-3xl font-bold text-rose-600">
                  {summary!.player2_wins}
                </div>
              </div>
            </div>
            <div className="mt-4 grid gap-2 text-sm sm:grid-cols-2">
              <div className="flex justify-between border-t border-slate-100 pt-2">
                <span className="text-slate-600">All-time win rate ({name(h2h.player1_id)})</span>
                <span className="font-medium">{pct(summary!.player1_win_rate)}</span>
              </div>
              <div className="flex justify-between border-t border-slate-100 pt-2">
                <span className="text-slate-600">Last 5 meetings win rate</span>
                <span className="font-medium">{pct(summary!.last5_player1_win_rate)}</span>
              </div>
            </div>
          </Card>

          {/* Surface split + meetings */}
          <div className="grid gap-4 lg:grid-cols-2">
            <Card title="Surface split">
              {meetings.length === 0 ? (
                <Empty message="These players have never met" />
              ) : (
                <ul className="space-y-2 text-sm">
                  {SURFACES.map((s) => (
                    <li key={s} className="flex justify-between border-b border-slate-100 pb-1 last:border-0">
                      <span className="capitalize">{s}</span>
                      <span className="font-medium">{surfaceCounts.get(s) ?? 0}</span>
                    </li>
                  ))}
                </ul>
              )}
            </Card>
            <Card title="Meetings">
              {meetings.length === 0 ? (
                <Empty message="No prior meetings" />
              ) : (
                <ul className="max-h-80 space-y-2 overflow-y-auto text-sm">
                  {meetings.map((m) => (
                    <li key={`${m.match_date}-${m.winner_id}`} className="flex justify-between border-b border-slate-100 pb-1 last:border-0">
                      <span className="text-slate-500">{m.match_date}</span>
                      <span className="w-1/3 text-slate-600">{m.tournament}</span>
                      <span className="w-16 capitalize text-slate-600">{m.surface}</span>
                      <span className="font-medium">{name(m.winner_id)} won</span>
                    </li>
                  ))}
                </ul>
              )}
            </Card>
          </div>

          {/* Rank lines */}
          <Card title="Rank lines">
            {rankAQ.isLoading || rankBQ.isLoading ? (
              <Loading label="Loading rank histories" />
            ) : rankAQ.isError || rankBQ.isError ? (
              <ErrorBox
                error={rankAQ.error ?? rankBQ.error}
                onRetry={() => {
                  rankAQ.refetch()
                  rankBQ.refetch()
                }}
              />
            ) : rankPointsA.length === 0 && rankPointsB.length === 0 ? (
              <Empty message="No rank history for either player" />
            ) : (
              <ReactECharts option={rankOption} style={{ height: 320 }} />
            )}
          </Card>

          {/* Model overlay */}
          <Card title="Model prediction">
            <div className="flex flex-wrap items-end gap-4">
              <label className="text-sm">
                <span className="mb-1 block text-xs uppercase tracking-wide text-slate-400">
                  Surface
                </span>
                <select
                  value={surface}
                  onChange={(e) => setSurface(e.target.value as Surface)}
                  className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm shadow-sm outline-none focus:border-emerald-500"
                >
                  {SURFACES.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </label>
              <label className="text-sm">
                <span className="mb-1 block text-xs uppercase tracking-wide text-slate-400">
                  Venue
                </span>
                <select
                  value={indoor}
                  onChange={(e) =>
                    setIndoor(e.target.value === '' ? '' : (Number(e.target.value) as 0 | 1))
                  }
                  className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm shadow-sm outline-none focus:border-emerald-500"
                >
                  <option value="">-</option>
                  <option value="0">Outdoor</option>
                  <option value="1">Indoor</option>
                </select>
              </label>
              <label className="text-sm">
                <span className="mb-1 block text-xs uppercase tracking-wide text-slate-400">
                  Tournament tier
                </span>
                <select
                  value={tournament}
                  onChange={(e) => setTournament(e.target.value as TournamentTier | '')}
                  className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm shadow-sm outline-none focus:border-emerald-500"
                >
                  <option value="">-</option>
                  {TOURNAMENT_TIERS.map((tier) => (
                    <option key={tier.value} value={tier.value}>
                      {tier.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="text-sm">
                <span className="mb-1 block text-xs uppercase tracking-wide text-slate-400">
                  Round
                </span>
                <select
                  value={round}
                  onChange={(e) => setRound(e.target.value as MatchRound | '')}
                  className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm shadow-sm outline-none focus:border-emerald-500"
                >
                  <option value="">-</option>
                  {ROUNDS.map((matchRound) => (
                    <option key={matchRound.value} value={matchRound.value}>
                      {matchRound.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="text-sm">
                <span className="mb-1 block text-xs uppercase tracking-wide text-slate-400">
                  As of date
                </span>
                <input
                  type="date"
                  value={asOfDate}
                  onChange={(e) => setAsOfDate(e.target.value)}
                  className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm shadow-sm outline-none focus:border-emerald-500"
                />
              </label>
              <button
                type="button"
                onClick={() =>
                  predict.mutate({
                    player_id: playerA!,
                    opponent_id: playerB!,
                    surface,
                    ...(tournament ? { tournament } : {}),
                    ...(round ? { round } : {}),
                    ...(asOfDate ? { as_of_date: asOfDate } : {}),
                    ...(indoor !== '' ? { indoor } : {}),
                  })
                }
                disabled={predict.isPending}
                className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {predict.isPending ? 'Predicting...' : 'Predict'}
              </button>
            </div>

            {predict.isError && (
              <div className="mt-4 rounded border border-rose-200 bg-rose-50 p-3 text-sm text-rose-800">
                {predict.error instanceof Error ? predict.error.message : String(predict.error)}
              </div>
            )}

            {pred && (
              <div className="mt-4 space-y-4">
                <div className="text-center">
                  <div className="text-sm text-slate-500">
                    {name(pred.predicted_winner)} wins
                  </div>
                  <div className="text-5xl font-bold text-emerald-600">{pct(pred.p_win)}</div>
                  <div className="mt-1 text-xs text-slate-400">
                    Stacked ensemble, {Math.round(pred.response_ms)} ms
                  </div>
                </div>
                <div className="space-y-3">
                  <StatBar label="p_linear" value={pred.p_linear} />
                  <StatBar label="p_gbdt" value={pred.p_gbdt} />
                  <StatBar label="p_nn" value={pred.p_nn} />
                </div>
              </div>
            )}
          </Card>
        </div>
      )}
    </div>
  )
}
