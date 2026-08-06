// Typed client for the local Bento service (src/serving/service.py).
// All GET endpoints return {ok: true, data: {...}} or {ok: false, error: str}
// (400/404/500); predict_from_ids returns a flat, unwrapped dict. Shapes are
// the dashboard's contract with the backend — keep them in sync.

// Default: relative /api, proxied to the Bento by nginx (production) or the
// Vite dev proxy (local). VITE_API_BASE_URL overrides for local dev against a
// bare backend origin.
const BASE = import.meta.env.VITE_API_BASE_URL || '/api'

export interface Player {
  player_id: string
  display_name: string
  matches_played: number
}

export interface CareerStats {
  matches_played: number
  win_rate: number | null
  first_serve_win_pct: number | null
  second_serve_win_pct: number | null
  serve_win_pct: number | null
  break_points_saved_pct: number | null
}

export interface RecentForm {
  snapshot_date: string
  last_10_win_rate: number | null
}

export interface RankPointsTrend {
  earliest: number
  latest: number
  delta: number
}

export type Surface = 'clay' | 'grass' | 'hard' | 'carpet'
export type TournamentTier = 'grand_slam' | 'masters' | 'atp_500' | 'atp_250' | 'davis_cup' | 'atp_finals' | 'olympics' | 'professional'
export type MatchRound = 'r128' | 'r64' | 'r32' | 'r16' | 'qf' | 'sf' | 'f'

export interface SurfaceRate {
  surface: Surface
  matches: number
  win_rate: number | null
}

export interface PlayerProfile {
  player_id: string
  display_name: string
  handedness: string | null
  backhand: string | null
  height: number | null
  turned_pro: number | null
  birthplace: string | null
  summary: string | null
  career: CareerStats
  surface_rates: SurfaceRate[]
  recent_form: RecentForm | null
  rank_points_trend: RankPointsTrend | null
}

export interface RankPoint {
  rank_date: string
  rank: number | null
}

export interface RankHistory {
  player_id: string
  rank_history: RankPoint[]
}

export interface MatchRow {
  match_id: string
  match_date: string
  tournament: string
  surface: string
  round: string
  opponent_id: string
  opponent_name: string | null
  ranking: number | null
  result: 'won' | 'lost'
  aces: number
  double_faults: number
  first_serve_points_won: number
  second_serve_points_won: number
  total_serve_points: number
  service_games: number
  break_points_saved: number
  break_points_faced: number
}

export interface MatchHistory {
  player_id: string
  matches: MatchRow[]
}

export interface H2HMeeting {
  match_date: string
  surface: string
  tournament: string
  round: string
  winner_id: string
  loser_id: string
  player1_won: boolean
}

export interface H2HSummary {
  meetings: number
  player1_wins: number
  player2_wins: number
  player1_win_rate: number
  last5_player1_win_rate: number
}

export interface H2HResponse {
  player1_id: string
  player2_id: string
  meetings: H2HMeeting[]
  summary: H2HSummary
}

export interface PredictResponse {
  player_id: string
  opponent_id: string
  p_win: number
  p_linear: number
  p_gbdt: number
  p_nn: number
  predicted_winner: string
  response_ms: number
}

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(BASE + path)
  let body: unknown
  try {
    body = await res.json()
  } catch {
    throw new ApiError(res.status, `HTTP ${res.status}`)
  }
  const envelope = body as { ok: boolean; data?: T; error?: string }
  if (!res.ok || !envelope.ok) {
    throw new ApiError(res.status, envelope.error ?? `HTTP ${res.status}`)
  }
  return envelope.data as T
}

export function getPlayers(): Promise<{ players: Player[] }> {
  return get('/players')
}

export function getPlayerProfile(playerId: string): Promise<PlayerProfile> {
  return get(`/player_profile?player_id=${encodeURIComponent(playerId)}`)
}

export function getRankHistory(playerId: string): Promise<RankHistory> {
  return get(`/rank_history?player_id=${encodeURIComponent(playerId)}`)
}

export function getMatchHistory(playerId: string, limit = 20): Promise<MatchHistory> {
  return get(`/match_history?player_id=${encodeURIComponent(playerId)}&limit=${limit}`)
}

export function getHeadToHead(a: string, b: string): Promise<H2HResponse> {
  return get(
    `/head_to_head?player_id=${encodeURIComponent(a)}&opponent_id=${encodeURIComponent(b)}`,
  )
}

export interface PredictInput {
  player_id: string
  opponent_id: string
  surface: Surface
  tournament?: TournamentTier
  round?: MatchRound
  as_of_date?: string
  indoor?: 0 | 1
}

// Raw (unwrapped) response — the backend returns the flat dict directly.
export async function predictFromIds(input: PredictInput): Promise<PredictResponse> {
  const res = await fetch(BASE + '/predict_from_ids', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  })
  let body: { error?: string } | null = null
  try {
    body = await res.json()
  } catch {
    // non-JSON error body; fall through to the HTTP status message
  }
  if (!res.ok) {
    throw new ApiError(res.status, body?.error ?? `HTTP ${res.status}`)
  }
  return body as unknown as PredictResponse
}
