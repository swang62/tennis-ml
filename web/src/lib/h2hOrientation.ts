/** 0-100 x-axis value for a Player A win probability: the predicted player's
 *  probability percent, so bars sit on the predicted player's side. */
export function orientedProbability(p: number, winnerIsA: boolean): number {
  return (winnerIsA ? p : 1 - p) * 100;
}
