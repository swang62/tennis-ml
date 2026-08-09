import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import {
  getHeadToHead,
  getPlayers,
  getRankHistory,
  predictFromIds,
  type H2HMeeting,
  type MatchRound,
  type RankHistory,
  type Surface,
  type TournamentTier,
} from '../api'
import {
  Card,
  Empty,
  ErrorBox,
  Kicker,
  Loading,
  PlayerPicker,
  pct,
} from '../components'
import { axisOption, baseChartOption, chartTokens } from '../lib/charts'
import { ROUND_LABEL, TIER_LABEL, fairOdds, sanitizeErrorMessage } from '../lib/format'
import { h2hRoute } from '../router'
import { useTheme } from '../theme'

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

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)

// Join known meeting metadata; retain unknown labels rather than ids.
function meetingMeta(m: H2HMeeting): string {
  const parts = [
    m.tournament ? (TIER_LABEL[m.tournament as keyof typeof TIER_LABEL] ?? m.tournament) : '',
    m.surface ? cap(m.surface) : '',
    m.round ? (ROUND_LABEL[m.round as keyof typeof ROUND_LABEL] ?? m.round) : '',
  ]
  return parts.filter(Boolean).join(' · ')
}

// Latest non-null rank is this page's sole rank signal.
function lastRank(history: RankHistory | undefined): { rank: number; date: string } | null {
  const pts = (history?.rank_history ?? []).filter((p) => p.rank != null)
  if (pts.length === 0) return null
  const sorted = [...pts].sort((a, b) => a.rank_date.localeCompare(b.rank_date))
  const latest = sorted[sorted.length - 1]
  return { rank: latest.rank as number, date: latest.rank_date }
}

// Mirrored comparison row; invert makes lower ranks fill farther.
interface MirrorRow {
  label: string
  a: number | null
  b: number | null
  aText: string
  bText: string
  invert?: boolean
}

// Zero totals leave both sides empty; an unranked side does not fill.
function mirrorWidths(row: MirrorRow): [number, number] {
  if (row.a == null || row.b == null) return row.a == null ? [0, 1] : [1, 0]
  const total = row.a + row.b
  if (total <= 0) return [0, 0]
  return row.invert ? [row.b / total, row.a / total] : [row.a / total, row.b / total]
}

export default function H2H() {
  const { theme } = useTheme()
  const { playerA: searchPlayerA } = h2hRoute.useSearch()
  const [playerA, setPlayerA] = useState<string | null>(searchPlayerA ?? null)
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
  const selectA = (id: string | null) => {
    setPlayerA(id)
    predict.reset()
  }
  const selectB = (id: string | null) => {
    setPlayerB(id)
    predict.reset()
  }

  const players = playersQ.data?.players ?? []
  const nameById = new Map(players.map((p) => [p.player_id, p.display_name]))
  // Display names only; an unknown player gets a neutral label, never the raw id.
  const name = (id: string) => nameById.get(id) ?? 'Unknown player'

  if (playersQ.isLoading) return <Loading label="Loading players" />
  if (playersQ.isError) return <ErrorBox error={playersQ.error} onRetry={() => playersQ.refetch()} />

  const h2h = h2hQ.data
  const meetings = h2h?.meetings ?? []
  const summary = h2h?.summary
  const sortedMeetings = [...meetings].sort((a, b) => b.match_date.localeCompare(a.match_date))

  // Map lower-id canonical probabilities back to Player A.
  const orient = (p: number) => (playerA! < playerB! ? p : 1 - p)
  const pred = predict.data
  const orientA = pred ? orient(pred.p_win) : 0
  const winnerP = pred
    ? pred.predicted_winner === pred.player_id
      ? pred.p_win
      : 1 - pred.p_win
    : 0

  const rankOf = (id: string) => lastRank(id === playerA ? rankAQ.data : rankBQ.data)

  const t = chartTokens()
  const ax = axisOption(t)

  const compOption: EChartsOption | null = pred
    ? {
        ...baseChartOption(t),
        tooltip: {
          ...baseChartOption(t).tooltip,
          trigger: 'axis',
          valueFormatter: (v) => pct(v as number),
        },
        grid: { left: 8, right: 16, top: 10, bottom: 4, containLabel: true },
        xAxis: {
          type: 'category',
          data: ['Linear', 'GBDT', 'NN'],
          axisLine: ax.axisLine,
          axisTick: { show: false },
          axisLabel: ax.axisLabel,
        },
        yAxis: {
          type: 'value',
          max: 1,
          axisLabel: { ...ax.axisLabel, formatter: (v) => `${Math.round((v as number) * 100)}%` },
          splitLine: ax.splitLine,
        },
        series: [
          {
            name: 'Model probability',
            type: 'bar',
            barWidth: '44%',
            data: [orient(pred.p_linear), orient(pred.p_gbdt), orient(pred.p_nn)],
            itemStyle: { color: t.clay, borderRadius: [6, 6, 2, 2] },
            markLine: {
              silent: true,
              symbol: 'none',
              lineStyle: { color: t.grass, type: 'dashed', width: 1.5 },
              label: {
                color: t.grass,
                fontSize: 11,
                formatter: `Ensemble ${pct(orient(pred.p_win))}`,
                position: 'insideEndTop',
              },
              data: [{ yAxis: orient(pred.p_win) }],
            },
          },
        ],
      }
    : null

  // Compare direct-meeting results and labeled current ranks only.
  const p1 = h2h ? name(h2h.player1_id) : ''
  const p2 = h2h ? name(h2h.player2_id) : ''
  const mirrorRows: MirrorRow[] = []
  const mirrorNotes: string[] = []
  if (h2h && summary) {
    mirrorRows.push({
      label: 'All-time wins',
      a: summary.player1_wins,
      b: summary.player2_wins,
      aText: String(summary.player1_wins),
      bText: String(summary.player2_wins),
    })
    const r1 = rankOf(h2h.player1_id)
    const r2 = rankOf(h2h.player2_id)
    if (r1 || r2) {
      mirrorRows.push({
        label: 'Current rank',
        invert: true,
        a: r1 ? r1.rank : null,
        b: r2 ? r2.rank : null,
        aText: r1 ? `#${r1.rank}` : 'n/a',
        bText: r2 ? `#${r2.rank}` : 'n/a',
      })
      mirrorNotes.push('Current-rank bars are inverted — the lower rank gets the longer bar.')
    }
    for (const s of [...new Set(meetings.map((m) => m.surface))].sort()) {
      const list = meetings.filter((m) => m.surface === s)
      const w1 = list.filter((m) => m.player1_won).length
      mirrorRows.push({
        label: `On ${cap(s)}`,
        a: w1,
        b: list.length - w1,
        aText: String(w1),
        bText: String(list.length - w1),
      })
    }
  }
  const mirrorSummary = mirrorRows
    .map((r) => `${r.label}: ${p1} ${r.aText} — ${p2} ${r.bText}`)
    .join('. ')

  return (
    <div className="space-y-5">
      <section className="page-head">
        <h1 className="page-title">Matchup Predictions</h1>
        <p className="page-sub">
          Pick two players for a model prediction and implied fair odds, then compare their direct
          head-to-head record and current rank.
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
          />
          <PlayerPicker
            players={players}
            value={playerB}
            onChange={selectB}
            placeholder="Player B"
            exclude={playerA}
          />
        </div>
      </section>

      {/* Match predictor */}
      <section className="card pred-card">
        <div className="pred-head">
          <Kicker>Match predictor</Kicker>
        </div>
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
                  value={indoor}
                  onChange={(e) =>
                    setIndoor(e.target.value === '' ? '' : (Number(e.target.value) as 0 | 1))
                  }
                  className="select"
                >
                  <option value="">-</option>
                  <option value="0">Outdoor</option>
                  <option value="1">Indoor</option>
                </select>
              </label>
              <label className="field">
                <span className="field-label">Tournament tier</span>
                <select
                  value={tournament}
                  onChange={(e) => setTournament(e.target.value as TournamentTier | '')}
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
                  onChange={(e) => setRound(e.target.value as MatchRound | '')}
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
                <span className="field-label">As of date</span>
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
                className="btn btn-primary btn-predict"
              >
                {predict.isPending ? 'Predicting...' : 'Predict winner'}
              </button>
            </div>

            {predict.isError && (
              <div className="mt-4 error-box">
                <p className="error-title">Prediction failed</p>
                <p className="error-msg">
                  {sanitizeErrorMessage(
                    predict.error instanceof Error ? predict.error.message : String(predict.error),
                    [playerA!, playerB!],
                  )}
                </p>
              </div>
            )}

            {pred && compOption && (
              <div className="pred-result" aria-live="polite">
                <div className="pred-main">
                  <span className="pred-winner">{name(pred.predicted_winner)} wins</span>
                  <span className="pred-pct num">{pct(winnerP)}</span>
                  <span className="pred-caption">
                    Stacked ensemble · {Math.round(pred.response_ms)} ms
                  </span>
                </div>
                <div className="odds-row">
                  <div className="odds">
                    <span className="odds-label">{name(playerA!)}</span>
                    <span className={`odds-num num ${orientA >= 0.5 ? 'is-fav' : ''}`}>
                      {fairOdds(orientA)}
                    </span>
                  </div>
                  <div className="odds">
                    <span className="odds-label">{name(playerB!)}</span>
                    <span className={`odds-num num ${orientA < 0.5 ? 'is-fav' : ''}`}>
                      {fairOdds(1 - orientA)}
                    </span>
                  </div>
                </div>
                <p className="mt-2 text-center text-[0.65rem] text-[var(--text-faint)]">
                  Fair odds implied by the model probability (0% margin)
                </p>
                <div className="mt-4">
                  <ReactECharts
                    key={theme}
                    option={compOption}
                    style={{ height: 220, width: '100%' }}
                    className="chart-frame"
                  />
                </div>
                <p className="mt-2 text-center text-[0.65rem] text-[var(--text-faint)]">
                  Model probabilities for {name(playerA!)}
                </p>
              </div>
            )}
          </>
        )}
      </section>

      {ready && h2hQ.isLoading && <Loading label="Loading head-to-head" />}
      {ready && h2hQ.isError && (
        <ErrorBox error={h2hQ.error} onRetry={() => h2hQ.refetch()} knownIds={[playerA!, playerB!]} />
      )}

      {ready && h2h && summary && (
        <>
          {/* Matchup comparison — mirrored bars, split at the center */}
          <Card title="Matchup comparison">
            <div className="mirror">
              <div className="mirror-head">
                <span className="mirror-name">{p1}</span>
                <span className="mirror-vs">vs</span>
                <span className="mirror-name">{p2}</span>
              </div>
              {mirrorRows.map((row) => {
                const [widthA, widthB] = mirrorWidths(row)
                return (
                  <div className="mirror-row" key={row.label}>
                    <div className="mirror-half is-left">
                      <span className="mirror-value num">{row.aText}</span>
                      <div className="mirror-bar-wrap">
                        <div
                          className="mirror-fill is-p1"
                          style={{ width: `${widthA * 100}%` }}
                          aria-hidden="true"
                        />
                      </div>
                    </div>
                    <span className="mirror-label">{row.label}</span>
                    <div className="mirror-half is-right">
                      <div className="mirror-bar-wrap">
                        <div
                          className="mirror-fill is-p2"
                          style={{ width: `${widthB * 100}%` }}
                          aria-hidden="true"
                        />
                      </div>
                      <span className="mirror-value num">{row.bText}</span>
                    </div>
                  </div>
                )
              })}
              <p className="sr-only">{mirrorSummary}</p>
              {mirrorNotes.length > 0 && <p className="mirror-note">{mirrorNotes.join(' ')}</p>}
            </div>
          </Card>

          {/* Direct meetings */}
          <Card title="Meetings">
            {meetings.length === 0 ? (
              <Empty message="No prior meetings" />
            ) : (
              <div className="meetings-list">
                {sortedMeetings.map((m) => {
                  // player1_won is canonical lower-id, not picker order.
                  const aWon = m.player1_won === (playerA === h2h.player1_id)
                  return (
                    <div key={`${m.match_date}-${m.winner_id}`} className="meeting">
                      <span className="meeting-date mono">{m.match_date}</span>
                      <span className="meeting-players">
                        <span className={`meeting-name${aWon ? ' is-winner' : ''}`}>
                          {name(playerA!)}
                        </span>
                        <span className="meeting-vs">beat</span>
                        <span className={`meeting-name${aWon ? '' : ' is-winner'}`}>
                          {name(playerB!)}
                        </span>
                      </span>
                      <span className="meeting-meta">{meetingMeta(m)}</span>
                    </div>
                  )
                })}
              </div>
            )}
          </Card>
        </>
      )}
    </div>
  )
}
