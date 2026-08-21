// Bento client: GET uses envelopes; predict_from_ids returns a flat response.

// Relative /api uses nginx or Vite proxy; VITE_API_BASE_URL overrides it.
const BASE = import.meta.env.VITE_API_BASE_URL || "/api";

export interface Player {
  player_id: string;
  display_name: string;
  matches_played: number;
  current_rank?: number | null;
  ioc: string;
  iso2: string;
}

export interface CareerStats {
  matches_played: number;
  latest_match_date: string | null;
}

export interface ServeMetrics {
  first_serve_in_pct: number | null;
  aces_per_first_serve: number | null;
  first_serve_points_won_pct: number | null;
  second_serve_points_won_pct: number | null;
  overall_serve_points_won_pct: number | null;
  double_faults_per_serve_point: number | null;
  aces_per_service_game: number | null;
  break_points_saved_pct: number | null;
}

export interface ReturnMetrics {
  return_points_won_pct: number | null;
  first_serve_return_points_won_pct: number | null;
  second_serve_return_points_won_pct: number | null;
  return_games_won_pct: number | null;
  break_point_conversion_pct: number | null;
  break_point_opportunities_per_return_game: number | null;
}

export interface RankInfo {
  latest_rank_points: number | null;
  earliest_rank_points: number | null;
  earliest_rank_points_date: string | null;
  latest_rank_points_date: string | null;
  rank_points_delta: number | null;
  current_rank: number | null;
}

export interface TourComparisons {
  first_serve_in_pct: number | null;
  aces_per_first_serve: number | null;
  first_serve_points_won_pct: number | null;
  second_serve_points_won_pct: number | null;
  overall_serve_points_won_pct: number | null;
  double_faults_per_serve_point: number | null;
  aces_per_service_game: number | null;
  break_points_saved_pct: number | null;
  return_points_won_pct: number | null;
  first_serve_return_points_won_pct: number | null;
  second_serve_return_points_won_pct: number | null;
  return_games_won_pct: number | null;
  break_point_conversion_pct: number | null;
  break_point_opportunities_per_return_game: number | null;
}

export interface RecentForm {
  snapshot_date: string;
  last_10_win_rate: number | null;
}

export interface RankPointsTrend {
  earliest: number;
  latest: number;
  delta: number;
}

export interface TourAverages {
  first_serve_win_pct: number | null;
  second_serve_win_pct: number | null;
}

export type Surface = "clay" | "grass" | "hard" | "carpet";
export type TournamentTier =
  | "grand_slam"
  | "masters"
  | "atp_500"
  | "atp_250"
  | "davis_cup"
  | "atp_finals"
  | "olympics"
  | "professional";
export type MatchRound =
  | "r128"
  | "r64"
  | "r32"
  | "r16"
  | "qf"
  | "sf"
  | "f"
  | "rr";

export interface SurfaceRate {
  surface: Surface;
  matches: number;
  win_rate: number | null;
}

export interface PlayerProfile {
  player_id: string;
  display_name: string;
  handedness: string | null;
  backhand: string | null;
  height: number | null;
  turned_pro: number | null;
  birthplace: string | null;
  summary: string | null;
  atp_name: string | null;
  birthdate: string | null;
  weight: number | null;
  coaches: string | null;
  ioc: string;
  iso2: string;
  country_name: string;
  career: CareerStats;
  serve: ServeMetrics;
  return: ReturnMetrics;
  surface_rates: SurfaceRate[];
  recent_form: RecentForm | null;
  rank_points_trend: RankPointsTrend | null;
  rank: RankInfo;
  tour_averages: TourAverages;
  tour_comparisons: TourComparisons;
}

export interface RankPoint {
  rank_date: string;
  rank: number;
}

export interface RankHistory {
  player_id: string;
  rank_history: RankPoint[];
}

export interface MatchRow {
  match_id: string;
  match_date: string;
  tournament: string;
  tournament_name: string | null;
  surface: string;
  round: string;
  opponent_id: string;
  opponent_name: string | null;
  opponent_ranking: number | null;
  result: "won" | "lost";
  score: string | null;
  aces: number;
  double_faults: number;
  first_serve_points_won: number;
  second_serve_points_won: number;
  total_serve_points: number;
  service_games: number;
  break_points_saved: number;
  break_points_faced: number;
}

export interface MatchHistory {
  player_id: string;
  matches: MatchRow[];
}

export interface H2HMeeting {
  match_id: string;
  match_date: string;
  surface: string;
  tournament: string;
  tournament_name: string | null;
  round: string;
  winner_id: string;
  loser_id: string;
  score: string | null;
  player1_won: boolean;
}

export interface H2HSummary {
  meetings: number;
  player1_wins: number;
  player2_wins: number;
  player1_win_rate: number;
  last5_player1_win_rate: number;
}

export interface H2HResponse {
  player1_id: string;
  player2_id: string;
  meetings: H2HMeeting[];
  summary: H2HSummary;
}

export interface SimilarPlayer {
  player_id: string;
  display_name: string;
  score: string;
}

export interface SimilarPlayersResponse {
  player_id: string;
  similar_players: SimilarPlayer[];
}

export interface PredictResponse {
  player_id: string;
  opponent_id: string;
  p_win: number;
  p_linear: number;
  p_gbdt: number;
  p_nn: number;
  predicted_winner: string;
  response_ms: number;
}

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(BASE + path);
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    throw new ApiError(res.status, `HTTP ${res.status}`);
  }
  const envelope = body as { ok: boolean; data?: T; error?: string };
  if (!res.ok || !envelope.ok) {
    throw new ApiError(res.status, envelope.error ?? `HTTP ${res.status}`);
  }
  return envelope.data as T;
}

export interface DirectoryResponse {
  players: Player[];
  total_players: number;
  latest_match_date: string | null;
  total_matches: number;
}

export function getDirectory(): Promise<DirectoryResponse> {
  return get("/directory");
}

export function getPlayerProfile(playerId: string): Promise<PlayerProfile> {
  return get(`/player_profile?player_id=${encodeURIComponent(playerId)}`);
}

export function getRankHistory(playerId: string): Promise<RankHistory> {
  return get(`/rank_history?player_id=${encodeURIComponent(playerId)}`);
}

export function getMatchHistory(
  playerId: string,
  limit = 20,
): Promise<MatchHistory> {
  return get(
    `/match_history?player_id=${encodeURIComponent(playerId)}&limit=${limit}`,
  );
}

export function getHeadToHead(a: string, b: string): Promise<H2HResponse> {
  return get(
    `/head_to_head?player_id=${encodeURIComponent(a)}&opponent_id=${encodeURIComponent(b)}`,
  );
}

export function getSimilarPlayers(
  playerId: string,
  limit = 3,
): Promise<SimilarPlayersResponse> {
  return get(
    `/similar_players?player_id=${encodeURIComponent(playerId)}&limit=${limit}`,
  );
}

export interface PredictInput {
  player_id: string;
  opponent_id: string;
  surface: Surface;
  tournament?: TournamentTier;
  round?: MatchRound;
  as_of_date?: string;
  is_indoor?: 0 | 1;
}

// Raw (unwrapped) response — the backend returns the flat dict directly.
async function doPredictFromIds(input: PredictInput): Promise<PredictResponse> {
  const res = await fetch(`${BASE}/predict_from_ids`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ row: input }),
  });
  let body: { error?: string } | null = null;
  try {
    body = await res.json();
  } catch {
    // non-JSON error body; fall through to the HTTP status message
  }
  if (!res.ok) {
    throw new ApiError(res.status, body?.error ?? `HTTP ${res.status}`);
  }
  return body as unknown as PredictResponse;
}

// Reuse deterministic requests; evict failures so retries reach the network.
const predictCache = new Map<string, Promise<PredictResponse>>();

export function predictFromIds(input: PredictInput): Promise<PredictResponse> {
  const key = JSON.stringify(input);
  const hit = predictCache.get(key);
  if (hit) return hit;
  const pending = doPredictFromIds(input);
  predictCache.set(key, pending);
  pending.then(
    () => {},
    () => predictCache.delete(key),
  );
  return pending;
}
