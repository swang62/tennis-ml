// Deterministic ATP Tour overview URL derived only from data already in
// PlayerProfile: a slug of display_name plus the lowercase canonical
// player_id. No network, no name search, no runtime lookups. The slug is
// informational — ATP resolves the id even when the slug is imperfect.

const ATP_PLAYERS_BASE = "https://www.atptour.com/en/players";

function displayNameSlug(displayName: string): string {
  return displayName
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

export function atpOverviewUrl(displayName: string, playerId: string): string {
  return `${ATP_PLAYERS_BASE}/${displayNameSlug(displayName)}/${playerId.toLowerCase()}/overview`;
}
