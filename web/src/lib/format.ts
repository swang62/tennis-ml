import type { MatchRound, TournamentTier } from '../api'

// Label maps and derivations shared by pages. Pure data formatting — no
// backend access, no invented values (fair odds are derived 1/p from the
// model probability and labelled as implied odds).

export const TIER_LABEL: Record<TournamentTier, string> = {
  grand_slam: 'Grand Slam',
  masters: 'Masters',
  atp_500: 'ATP 500',
  atp_250: 'ATP 250',
  davis_cup: 'Davis Cup',
  atp_finals: 'ATP Finals',
  olympics: 'Olympics',
  professional: 'Pro',
}

export const ROUND_LABEL: Record<MatchRound, string> = {
  r128: 'R128',
  r64: 'R64',
  r32: 'R32',
  r16: 'R16',
  qf: 'QF',
  sf: 'SF',
  f: 'F',
}

// Fair decimal odds implied by a probability (0% margin). Derived, not
// sourced — presented as "fair odds" in the UI.
export function fairOdds(p: number): string {
  if (!(p > 0 && p < 1)) return '—'
  return (1 / p).toFixed(2)
}

const escapeRegExp = (s: string): string => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

// Scrub known player ids out of user-facing error text. Backend error strings
// can echo the requested id (e.g. "unknown player_id: <id>"), and the UI
// renders them in ErrorBox; ids the caller knows about are stripped and the
// leftover punctuation/whitespace collapsed.
export function sanitizeErrorMessage(message: string, knownIds: readonly string[]): string {
  let out = message
  for (const id of knownIds) {
    if (!id) continue
    out = out.replace(new RegExp(`\\b${escapeRegExp(id)}\\b`, 'gi'), '')
  }
  return out
    .replace(/\s+/g, ' ')
    .replace(/\s+([,.;:])/g, '$1')
    .replace(/[:,]\s*$/, '')
    .trim()
}
