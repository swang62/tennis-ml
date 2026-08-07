import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import {
  getHeadToHead,
  getMatchHistory,
  getPlayers,
  getRankHistory,
  predictFromIds,
  type MatchHistory,
  type MatchRound,
  type RankHistory,
  type Surface,
  type TournamentTier,
} from '../api'
import {
  Card,
  Empty,
  ErrorBox,
  FormStrip,
  Kicker,
  Loading,
  PlayerPicker,
  StatBar,
  pct,
} from '../components'
import { axisOption, baseChartOption, chartTokens, surfaceColor } from '../lib/charts'
import { ROUND_LABEL, TIER_LABEL, fairOdds } from '../lib/format'
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
const FORM_LIMIT = 10

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)

// Latest known rank: the last non-null point of the rank history. The rank
// graph is gone; the current rank is the only rank signal this page shows.
function lastRank(history: RankHistory | undefined): { rank: number; date: string } | null {
  const pts = (history?.rank_history ?? []).filter((p) => p.rank != null)
  if (pts.length === 0) return null
  const sorted = [...pts].sort((a, b) => a.rank_date.localeCompare(b.rank_date))
  const latest = sorted[sorted.length - 1]
  return { rank: latest.rank as number, date: latest.rank_date }
}

// Recent-form summary from the last FORM_LIMIT matches. The strip is the 8
// most recent results, oldest first (same convention as the profile page);
// the win rate is derived over every returned match.
function formOf(history: MatchHistory | undefined) {
  const sorted = [...(history?.matches ?? [])].sort((a, b) => b.match_date.localeCompare(a.match_date))
  const results = sorted.slice(0, 8).map((m) => m.result).reverse()
  const won = sorted.filter((m) => m.result === 'won').length
  return {
    results,
    won,
    total: sorted.length,
    lastDate: sorted.length > 0 ? sorted[0].match_date : null,
  }
}

// One diverging row of the mirrored comparison: both halves measured against
// each other, longer bar = bigger share. `invert` flips the encoding for rank
// (lower is better), so the better-ranked player gets the longer bar.
interface MirrorRow {
  label: string
  a: number | null
  b: number | null
  aText: string
  bText: string
  invert?: boolean
}

function shareOfA(row: MirrorRow): number {
  if (row.a == null || row.b == null) return row.a == null ? 0 : 1
  const total = row.a + row.b
  if (total <= 0) return 0
  return row.invert ? row.b / total : row.a / total
}

export default function H2H() {
  const { theme } = useTheme()
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
  const formAQ = useQuery({
    queryKey: ['match_history', playerA, FORM_LIMIT],
    queryFn: () => getMatchHistory(playerA!, FORM_LIMIT),
    enabled: ready,
  })
  const formBQ = useQuery({
    queryKey: ['match_history', playerB, FORM_LIMIT],
    queryFn: () => getMatchHistory(playerB!, FORM_LIMIT),
    enabled: ready,
  })

  const predict = useMutation({ mutationFn: predictFromIds })
  const selectA = (id: string) => {
    setPlayerA(id)
    predict.reset()
  }
  const selectB = (id: string) => {
    setPlayerB(id)
    predict.reset()
  }

  const players = playersQ.data?.players ?? []
  const nameById = new Map(players.map((p) => [p.player_id, p.display_name]))
  const name = (id: string) => nameById.get(id) ?? id

  if (playersQ.isLoading) return <Loading label="Loading players" />
  if (playersQ.isError) return <ErrorBox error={playersQ.error} onRetry={() => playersQ.refetch()} />

  const h2h = h2hQ.data
  const meetings = h2h?.meetings ?? []
  const summary = h2h?.summary
  const sortedMeetings = [...meetings].sort((a, b) => b.match_date.localeCompare(a.match_date))

  const surfaceCounts = new Map<string, number>()
  for (const m of meetings) {
    surfaceCounts.set(m.surface, (surfaceCounts.get(m.surface) ?? 0) + 1)
  }

  // Canonical orientation: the model and the h2h summary are computed for the
  // lower-id player. `orient` maps a canonical probability onto Player A so
  // displayed values always belong to the name they sit next to.
  const orient = (p: number) => (playerA! < playerB! ? p : 1 - p)
  const pred = predict.data
  const orientA = pred ? orient(pred.p_win) : 0
  const winnerP = pred
    ? pred.predicted_winner === pred.player_id
      ? pred.p_win
      : 1 - pred.p_win
    : 0

  const rankOf = (id: string) => lastRank(id === playerA ? rankAQ.data : rankBQ.data)
  const rankText = (q: { isLoading: boolean; isError: boolean; data?: RankHistory }) => {
    if (q.isLoading) return ['—', 'loading rank history'] as const
    if (q.isError) return ['n/a', 'rank history unavailable'] as const
    const latest = lastRank(q.data)
    return latest
      ? ([`#${latest.rank}`, `as of ${latest.date}`] as const)
      : (['n/a', 'no rank history (unranked)'] as const)
  }
  const [rankAText, rankACaption] = rankText(rankAQ)
  const [rankBText, rankBCaption] = rankText(rankBQ)

  const formA = formOf(formAQ.data)
  const formB = formOf(formBQ.data)

  const t = chartTokens()
  const ax = axisOption(t)

  const surfaceOption: EChartsOption = {
    ...baseChartOption(t),
    title: {
      text: String(meetings.length),
      subtext: 'meetings',
      left: 'center',
      top: '34%',
      textStyle: { color: t.text, fontSize: 28, fontWeight: 800 },
      subtextStyle: { color: t.faint, fontSize: 10 },
    },
    tooltip: {
      ...baseChartOption(t).tooltip,
      trigger: 'item',
      formatter: '{b}: {c} meetings ({d}%)',
    },
    legend: {
      ...baseChartOption(t).legend,
      bottom: 0,
      left: 'center',
      icon: 'circle',
      itemWidth: 8,
      itemHeight: 8,
    },
    series: [
      {
        type: 'pie',
        radius: ['55%', '78%'],
        center: ['50%', '42%'],
        avoidLabelOverlap: true,
        itemStyle: { borderColor: t.raised, borderWidth: 2 },
        label: { show: false },
        emphasis: { scaleSize: 6 },
        data: SURFACES.filter((s) => (surfaceCounts.get(s) ?? 0) > 0).map((s) => ({
          name: cap(s),
          value: surfaceCounts.get(s) ?? 0,
          itemStyle: { color: surfaceColor(s, t) },
        })),
      },
    ],
  }

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

  // Mirrored comparison rows: every row splits at the center line, the left
  // player's bar grows left and the right player's bar grows right. Rows are
  // derived purely from the h2h summary, the meetings, the rank histories and
  // the prediction — nothing invented.
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
    if (summary.meetings > 0 && summary.last5_player1_win_rate != null) {
      mirrorRows.push({
        label: 'Last 5 meetings',
        a: summary.last5_player1_win_rate,
        b: 1 - summary.last5_player1_win_rate,
        aText: pct(summary.last5_player1_win_rate),
        bText: pct(1 - summary.last5_player1_win_rate),
      })
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
    if (pred) {
      mirrorRows.push({
        label: 'Model win prob',
        a: pred.p_win,
        b: 1 - pred.p_win,
        aText: pct(pred.p_win),
        bText: pct(1 - pred.p_win),
      })
      mirrorNotes.push(`Model probability is for ${p1}.`)
    }
  }
  const mirrorSummary = mirrorRows
    .map((r) => `${r.label}: ${p1} ${r.aText} — ${p2} ${r.bText}`)
    .join('. ')

  const leadClass = (a: number, b: number) => (a > b ? 'is-grass' : a < b ? 'is-ice' : '')

  return (
    <div className="space-y-5">
      <section className="page-head">
        <Kicker>Match lab</Kicker>
        <h1 className="page-title">Head-to-Head</h1>
        <p className="page-sub">
          Pick two players for a model prediction and implied fair odds, then compare their
          head-to-head history, current rank, and recent form.
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
          />
          <PlayerPicker
            players={players}
            value={playerB}
            onChange={selectB}
            placeholder="Player B"
          />
        </div>

        {ready && (
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="card">
              <div className="stat">
                <span className="stat-label">Current rank · {name(playerA!)}</span>
                <span className="stat-num num">{rankAText}</span>
                <span className="mono text-xs text-[var(--text-faint)]">{rankACaption}</span>
              </div>
            </div>
            <div className="card">
              <div className="stat">
                <span className="stat-label">Current rank · {name(playerB!)}</span>
                <span className="stat-num num">{rankBText}</span>
                <span className="mono text-xs text-[var(--text-faint)]">{rankBCaption}</span>
              </div>
            </div>
          </div>
        )}
      </section>

      {/* Match predictor */}
      <section className="card pred-card">
        <div className="pred-head">
          <Kicker>Match predictor</Kicker>
          <span className="badge badge-clay">Betting signal</span>
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
                  {predict.error instanceof Error ? predict.error.message : String(predict.error)}
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
        <ErrorBox error={h2hQ.error} onRetry={() => h2hQ.refetch()} />
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
              {mirrorRows.map((row) => (
                <div className="mirror-row" key={row.label}>
                  <div className="mirror-half is-left">
                    <span className="mirror-value num">{row.aText}</span>
                    <div className="mirror-bar-wrap">
                      <div
                        className="mirror-fill is-p1"
                        style={{ width: `${shareOfA(row) * 100}%` }}
                        aria-hidden="true"
                      />
                    </div>
                  </div>
                  <span className="mirror-label">{row.label}</span>
                  <div className="mirror-half is-right">
                    <div className="mirror-bar-wrap">
                      <div
                        className="mirror-fill is-p2"
                        style={{ width: `${(1 - shareOfA(row)) * 100}%` }}
                        aria-hidden="true"
                      />
                    </div>
                    <span className="mirror-value num">{row.bText}</span>
                  </div>
                </div>
              ))}
              <p className="sr-only">{mirrorSummary}</p>
              {mirrorNotes.length > 0 && <p className="mirror-note">{mirrorNotes.join(' ')}</p>}
            </div>
          </Card>

          {/* Historical head-to-head */}
          <section className="space-y-5" aria-label="Historical head-to-head">
            <section className="card">
              <Kicker>All-time series</Kicker>
              <div className="h2h-score">
                <div className="h2h-side">
                  <span className="h2h-name">{name(h2h.player1_id)}</span>
                  <span className="h2h-id mono">{h2h.player1_id}</span>
                  <span
                    className={`h2h-wins num ${leadClass(summary.player1_wins, summary.player2_wins)}`}
                  >
                    {summary.player1_wins}
                  </span>
                </div>
                <div className="h2h-middle">
                  <span className="h2h-vs">all-time</span>
                  <span className="h2h-total num">{summary.meetings}</span>
                  <span className="h2h-vs">meetings</span>
                </div>
                <div className="h2h-side">
                  <span className="h2h-name">{name(h2h.player2_id)}</span>
                  <span className="h2h-id mono">{h2h.player2_id}</span>
                  <span
                    className={`h2h-wins num ${leadClass(summary.player2_wins, summary.player1_wins)}`}
                  >
                    {summary.player2_wins}
                  </span>
                </div>
              </div>
              <div className="mt-5 grid gap-4 sm:grid-cols-2">
                <StatBar
                  label={`All-time win rate · ${name(h2h.player1_id)}`}
                  value={summary.player1_win_rate}
                />
                <StatBar
                  label={`Last 5 meetings · ${name(h2h.player1_id)}`}
                  value={summary.last5_player1_win_rate}
                />
              </div>
            </section>

            <div className="grid gap-5 lg:grid-cols-2">
              <Card title="Meeting surface mix">
                {meetings.length === 0 ? (
                  <Empty message="These players have never met" />
                ) : (
                  <ReactECharts
                    key={theme}
                    option={surfaceOption}
                    style={{ height: 260, width: '100%' }}
                    className="chart-frame"
                  />
                )}
              </Card>
              <Card title="Meetings">
                {meetings.length === 0 ? (
                  <Empty message="No prior meetings" />
                ) : (
                  <div className="meetings-list">
                    {sortedMeetings.map((m) => (
                      <div key={`${m.match_date}-${m.winner_id}`} className="meeting">
                        <span className="meeting-date mono">{m.match_date}</span>
                        <span className="meeting-meta">
                          {TIER_LABEL[m.tournament as keyof typeof TIER_LABEL] ?? m.tournament} ·{' '}
                          {cap(m.surface)} ·{' '}
                          {ROUND_LABEL[m.round as keyof typeof ROUND_LABEL] ?? m.round}
                        </span>
                        <span className="meeting-result">{name(m.winner_id)} won</span>
                      </div>
                    ))}
                  </div>
                )}
              </Card>
            </div>
          </section>
        </>
      )}

      {/* Recent form */}
      <section aria-label="Recent form">
        <Card title="Recent form">
          {!ready ? (
            <Empty message="Select two players to compare form" />
          ) : formAQ.isLoading || formBQ.isLoading ? (
            <Loading label="Loading recent form" />
          ) : formAQ.isError || formBQ.isError ? (
            <ErrorBox
              error={formAQ.error ?? formBQ.error}
              onRetry={() => {
                formAQ.refetch()
                formBQ.refetch()
              }}
            />
          ) : formA.total === 0 && formB.total === 0 ? (
            <Empty message="No match history for either player" />
          ) : (
            <div className="grid gap-5 sm:grid-cols-2">
              <div className="space-y-2">
                <div className="flex items-baseline justify-between gap-3">
                  <span className="font-bold">{name(playerA!)}</span>
                  <span className="num text-sm font-semibold">
                    {formA.total > 0 ? pct(formA.won / formA.total) : '—'}
                  </span>
                </div>
                {formA.results.length > 0 ? (
                  <FormStrip results={formA.results} />
                ) : (
                  <p className="text-xs text-[var(--text-faint)]">No recent results</p>
                )}
                <p className="text-xs text-[var(--text-faint)]">
                  {formA.total > 0
                    ? `Win rate over last ${formA.total} ${formA.total === 1 ? 'match' : 'matches'}${formA.lastDate ? ` · last on ${formA.lastDate}` : ''}`
                    : 'No match history'}
                </p>
              </div>
              <div className="space-y-2">
                <div className="flex items-baseline justify-between gap-3">
                  <span className="font-bold">{name(playerB!)}</span>
                  <span className="num text-sm font-semibold">
                    {formB.total > 0 ? pct(formB.won / formB.total) : '—'}
                  </span>
                </div>
                {formB.results.length > 0 ? (
                  <FormStrip results={formB.results} />
                ) : (
                  <p className="text-xs text-[var(--text-faint)]">No recent results</p>
                )}
                <p className="text-xs text-[var(--text-faint)]">
                  {formB.total > 0
                    ? `Win rate over last ${formB.total} ${formB.total === 1 ? 'match' : 'matches'}${formB.lastDate ? ` · last on ${formB.lastDate}` : ''}`
                    : 'No match history'}
                </p>
              </div>
            </div>
          )}
        </Card>
      </section>
    </div>
  )
}
