import type { MatchRound, TournamentTier } from "../api";

// Format a 0..1 ratio as a percentage (e.g. 0.657 -> "65.7%").
export function pct(value: number | null): string {
  if (value == null) return "—";
  if (value < 0 || value > 1) return "—";
  return `${Math.round(value * 1000) / 10}%`;
}

// Profile serve/return metric card formatters. Percentages always carry one
// decimal place ("80.0%", "▲ 9.0%") so decimal points line up vertically in
// the right-aligned value and delta columns; rates keep two-decimal precision.

export function formatMetric(value: number | null): string {
  if (value == null) return "n/a";
  return `${(Math.round(value * 1000) / 10).toFixed(1)}%`;
}

export function formatRate(value: number | null): string {
  if (value == null) return "n/a";
  return `${(Math.round(value * 1000) / 1000).toFixed(2)}`;
}

export function formatLongDate(value: string): string {
  const [year, month, day] = value.split("-").map(Number);
  const suffix = day % 10 === 1 && day !== 11 ? "st" : day % 10 === 2 && day !== 12 ? "nd" : day % 10 === 3 && day !== 13 ? "rd" : "th";
  return new Date(year, month - 1, day).toLocaleDateString(undefined, {
    month: "long",
    day: "numeric",
    year: "numeric",
  }).replace(String(day), `${day}${suffix}`);
}

export function formatDelta(
  delta: number | null,
  rate?: boolean,
): string | null {
  if (delta == null) return null;
  const change = rate
    ? `${(Math.round(Math.abs(delta) * 1000) / 1000).toFixed(2)} `
    : `${(Math.round(Math.abs(delta) * 1000) / 10).toFixed(1)}%`;
  return delta > 0 ? `▲ ${change}` : delta < 0 ? `▼ ${change}` : change;
}

// Shared labels and client-side display derivations.

export const TIER_LABEL: Record<TournamentTier, string> = {
  grand_slam: "Grand Slam",
  masters: "Masters",
  atp_500: "ATP 500",
  atp_250: "ATP 250",
  davis_cup: "Davis Cup",
  atp_finals: "ATP Finals",
  olympics: "Olympics",
  professional: "Pro",
};

export const ROUND_LABEL: Record<MatchRound, string> = {
  r128: "Round of 128",
  r64: "Round of 64",
  r32: "Round of 32",
  r16: "Round of 16",
  qf: "Quarterfinal",
  sf: "Semifinal",
  f: "Final",
  rr: "Round Robin",
};

// Fair decimal odds implied by a zero-margin probability.
export function fairOdds(p: number): string {
  if (!(p > 0 && p < 1)) return "—";
  return (1 / p).toFixed(1);
}

// Split a stored set score into boldable segments. The score is always
// winner-first ("6-4 7-6 RET"), so the first game count is the winner's and
// the second the loser's. `perspective` is the displayed player: "winner"
// bolds the whole set token in sets they won; "loser" bolds the whole token
// in sets they won. Non-set tokens (W/O, RET) stay plain.
export interface ScoreSegment {
  text: string;
  bold: boolean;
}

export function scoreSegments(
  score: string | null,
  perspective: "winner" | "loser",
): ScoreSegment[] | null {
  if (!score) return null;
  return score.trim().split(/\s+/).map((token, i) => {
    const match = /^(\d+)-(\d+)$/.exec(token);
    const playerWonSet =
      match &&
      (perspective === "winner"
        ? Number(match[1]) > Number(match[2])
        : Number(match[2]) > Number(match[1]));
    return { text: i > 0 ? ` ${token}` : token, bold: Boolean(playerWonSet) };
  });
}

const escapeRegExp = (s: string): string =>
  s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

// Remove requested ids from backend errors before rendering them.
export function sanitizeErrorMessage(
  message: string,
  knownIds: readonly string[],
): string {
  let out = message;
  for (const id of knownIds) {
    if (!id) continue;
    out = out.replace(new RegExp(`\\b${escapeRegExp(id)}\\b`, "gi"), "");
  }
  return out
    .replace(/\s+/g, " ")
    .replace(/\s+([,.;:])/g, "$1")
    .replace(/[:,]\s*$/, "")
    .trim();
}
