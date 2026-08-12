import type { H2HResponse } from "../api";

export function probabilityForPlayer(
  probability: number,
  canonicalPlayerId: string,
  playerId: string,
): number {
  return canonicalPlayerId === playerId ? probability : 1 - probability;
}

/** Signed chart edge: negative favors Player A (left), positive Player B (right). */
export function pickerPreferenceEdge(
  canonicalProbability: number,
  canonicalPlayerId: string,
  playerAId: string,
): number {
  const canonicalEdge = canonicalProbability - 0.5;
  return canonicalPlayerId === playerAId ? -canonicalEdge : canonicalEdge;
}

/** Map the API's lower-id canonical H2H response into picker order. */
export function orientH2H(
  h2h: H2HResponse,
  playerA: string,
): H2HResponse {
  if (h2h.player1_id === playerA) return h2h;

  return {
    ...h2h,
    player1_id: h2h.player2_id,
    player2_id: h2h.player1_id,
    meetings: h2h.meetings.map((meeting) => ({
      ...meeting,
      player1_won: !meeting.player1_won,
    })),
    summary: {
      ...h2h.summary,
      player1_wins: h2h.summary.player2_wins,
      player2_wins: h2h.summary.player1_wins,
      player1_win_rate: 1 - h2h.summary.player1_win_rate,
      last5_player1_win_rate: 1 - h2h.summary.last5_player1_win_rate,
    },
  };
}
