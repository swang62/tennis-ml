"""Prefect flow: ATP match-stats enrichment.

Parses the ATP results-archive and tournament pages, fetches Hawkeye Complete
stats on one shared persistent CloakBrowser page (sequential requests), and
upserts winner-first rows into ``bronze.match_events``. Match ids are stable:
a canonical ``(match_date, tournament_id, unordered player pair)`` reuses the
stored row's id, else a deterministic ``YYYY-TOURNAMENT_ID-NNN`` derived from
the ATP ``msNNN`` sequence. Every page/match failure is a counted skip unless
no valid page was fetched at all (then the run fails so Prefect retries).

Parsers/resolvers/mapper touch no network, browser, or DB; upserts and the
flow's watermark/bronze reads are the only DB-touching entry points.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, cast

import pandas as pd
from prefect import flow

from src.constants import BRONZE_MATCHES_TABLE, ROOT, WORK_POOL_NAME, load_env
from src.db.client import connection, execute_df, first_row_dict
from src.db.ingest import (
    LEVEL_MAP,
    _canonical_surface,
    _copy_df_into,
    canonical_match_id,
    canonical_players,
    load_player_metadata,
    load_ranking_player_map,
)
from src.features.columns import (
    BRONZE_COLUMNS,
    BRONZE_COLUMNS_FLOAT,
    BRONZE_COLUMNS_INT,
    BRONZE_COLUMNS_INT32,
    CANONICAL_SURFACES,
)
from src.flows import rankings

MATCHES_DEPLOYMENT_NAME = "matches"
MATCHES_CRON = "30 22 * * 1"

# ── Tournament discovery parsers (fixture-testable, no network) ──

# Archive entries are <li> blocks holding the tournament-info (badge, profile
# link, name, date) plus results CTA; <li> also appears in page navigation, so
# only blocks that carry a tournament-info div are treated as entries.
_ARCHIVE_LI_RE = re.compile(r"<li\b[^>]*>(.*?)</li>", re.S)
_PROFILE_LINK_RE = re.compile(r"/en/tournaments/([^/]+)/(\d+)/overview")
RESULTS_LINK_RE = re.compile(
    r'/en/scores/archive/([^/]+)/(\d+)/(\d{4})/(?:results|country-results)"'
)
_BADGE_RE = re.compile(r"categorystamps_([a-z0-9_]+)\.png")
_NAME_SPAN_RE = re.compile(r'class="name">([^<]+)</span>')
_DATE_SPAN_RE = re.compile(r'class="Date">([^<]+)</span>')

# Archive badge images encode the tournament category as a raw ATP level code;
# LEVEL_MAP (src/db/ingest.py) stays the single tier vocabulary. Badges outside
# the tier set (itf/unitedcup/lvr/...) map to None and the caller skips.
_BADGE_TO_LEVEL = {
    "grandslam": "G",
    "1000": "M",
    "500": "500",
    "250": "250",
}

# "2 - 11 January, 2026" puts the month only on the end; "18 January - 1
# February, 2026" carries it on both parts and the year only on the end.
_DATE_PART_RE = re.compile(r"(\d{1,2})(?:\s+([A-Za-z]+))?(?:,\s*(\d{4}))?")
_MONTHS = {
    name: index
    for index, name in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        1,
    )
}


def _tier_from_badge(badge: str | None) -> str | None:
    """Bronze tier for an archive badge image; None when absent or non-tier."""
    level = _BADGE_TO_LEVEL.get(badge or "")
    return LEVEL_MAP.get(level) if level is not None else None


def _parse_archive_dates(text: str, year: int) -> tuple[date | None, date | None]:
    """Start/end dates of an archive 'Date' span; (None, None) when unparseable.

    Handles day-only starts ("2 - 11 January, 2026"), month on both parts
    ("18 January - 1 February, 2026") and single dates (start == end). A
    start month later than the end month implies a cross-year range, so the
    start year steps back one.
    """
    parts = [part.strip() for part in re.split(r"\s*[-\u2013]\s*", text.strip())]
    if not parts:
        return None, None
    end_match = _DATE_PART_RE.match(parts[-1])
    if end_match is None or end_match.group(2) is None:
        return None, None
    end = date(
        int(end_match.group(3) or year), _MONTHS[end_match.group(2)], int(end_match.group(1))
    )
    if len(parts) == 1:
        return end, end
    start_match = _DATE_PART_RE.match(parts[0])
    if start_match is None or start_match.group(1) is None:
        return None, end
    start_day = int(start_match.group(1))
    start_month = start_match.group(2)
    if start_month is None:
        start = date(end.year, end.month, start_day)
    else:
        start_year = end.year - 1 if _MONTHS[start_month] > end.month else end.year
        start = date(start_year, _MONTHS[start_month], start_day)
    return start, end


def extract_tournaments_from_archive(html: str, year: int) -> list[dict[str, Any]]:
    """Parse the ATP results-archive page into tournament discovery rows.

    Each row carries ``slug``, ``tournament_id``, ``name``, ``tier`` (bronze
    vocabulary; None for non-tier or badge-less entries so the flow can
    skip/report), ``start_date``/``end_date`` (date or None) and ``year``.
    Entries without a profile link are ignored; a missing results link or badge
    never raises.
    """
    tournaments: list[dict[str, Any]] = []
    for block in _ARCHIVE_LI_RE.findall(html):
        if "tournament-info" not in block:
            continue
        profile = _PROFILE_LINK_RE.search(block)
        if profile is None:
            continue
        slug, tournament_id = profile.group(1), profile.group(2)
        results = RESULTS_LINK_RE.search(block)
        name_match = _NAME_SPAN_RE.search(block)
        date_match = _DATE_SPAN_RE.search(block)
        badge_match = _BADGE_RE.search(block)
        start_date, end_date = (
            _parse_archive_dates(date_match.group(1), year) if date_match else (None, None)
        )
        # The results link repeats the id and carries the entry's own year; use
        # it only when consistent with the authoritative profile link.
        entry_year = str(year)
        if results is not None and results.group(2) == tournament_id:
            entry_year = results.group(3)
        tournaments.append(
            {
                "slug": slug,
                "tournament_id": tournament_id,
                "name": name_match.group(1).strip() if name_match else None,
                "tier": _tier_from_badge(badge_match.group(1) if badge_match else None),
                "start_date": start_date,
                "end_date": end_date,
                "year": entry_year,
            }
        )
    return tournaments


def _as_date(value: date | str | None) -> date | None:
    """ISO string or date to date; None when unparseable."""
    if value is None or isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def in_window(
    tournaments: list[dict[str, Any]], start_date: date | str, end_date: date | str
) -> list[dict[str, Any]]:
    """Tournaments whose [start_date, end_date] intersects the run window.

    A tournament without reliable date metadata never intersects and is
    excluded (deterministic; unparseable metadata is reported upstream). The
    window compares inclusively on both ends.
    """
    start, end = _as_date(start_date), _as_date(end_date)
    if start is None or end is None:
        return []
    return [
        tournament
        for tournament in tournaments
        if tournament.get("start_date") is not None
        and tournament.get("end_date") is not None
        and tournament["start_date"] <= end
        and tournament["end_date"] >= start
    ]


def tier_for_tournament(
    tournament: dict[str, Any],
    existing_rows: list[dict[str, Any]] | None = None,
) -> str | None:
    """Bronze tier for an archive tournament row; None when unsupported.

    The archive badge tier (already bronze vocabulary) wins when present. A
    badge-less tournament is classified only from existing bronze match rows
    (``rankings.tier_from_bronze``) — never guessed, so an unknown tournament
    resolves to None and the flow skips/reports it instead of assuming an ATP
    250+ classification without evidence.
    """
    tier = tournament.get("tier")
    if tier:
        return str(tier)
    return rankings.tier_from_bronze(str(tournament.get("tournament_id") or ""), existing_rows)


# ── Match discovery parsers (fixture-testable, no network) ──

# One match card per played match: the results page renders each card as
# <div class="match"> holding .match-header (round + court), .match-content
# with two .stats-item/.player-info blocks and per-set .score-item cells, and
# .match-footer > .match-cta with the H2H and stats (msXXX) links.
_MATCH_CARD_RE = re.compile(r'<div class="match">')
_DAY_HEADER_RE = re.compile(r'<div class="tournament-day">')
_DAY_DATE_RE = re.compile(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})")
_ROUND_STRONG_RE = re.compile(r"<span><strong>([^<]+)</strong></span>")
_PLAYER_LINK_RE = re.compile(r'href="/en/players/([^/]+)/([^/]+)/overview"[^>]*>([^<]*)</a>')
_STATS_LINK_RE = re.compile(r"/en/scores/stats-centre/archive/(\d{4})/(\d+)/(ms\d+)\"")
_STATS_ITEM_RE = re.compile(r'class="stats-item"')
_SCORE_ITEM_RE = re.compile(r'<div class="score-item">(.*?)</div>', re.S)
_WINNER_DIV_RE = re.compile(r'<div class="winner">')
_MATCH_FOOTER_RE = re.compile(r'<div class="match-footer">')
_TEXT_TAG_RE = re.compile(r"<[^>]+>")

# Tournament-round strong text -> bronze round vocabulary (r128..f); anything
# unmapped (e.g. qualifying) keeps its raw long name for upstream reporting.
_ROUND_TO_BRONZE = {
    "Final": "f",
    "Semifinals": "sf",
    "Quarterfinals": "qf",
    "Round of 16": "r16",
    "Round of 32": "r32",
    "Round of 64": "r64",
    "Round of 128": "r128",
}


def _normalize_round(text: str) -> str:
    """Bronze round for a match-header strong text; raw name when unmapped."""
    name = re.split(r"\s*[-\u2013]\s*", text.strip(), maxsplit=1)[0]
    return _ROUND_TO_BRONZE.get(name, name or "")


def _score_items(block: str) -> list[str]:
    """Per-set score-item cells of one player's stats block (tags kept)."""
    return [re.sub(r"\s+", "", item) for item in _SCORE_ITEM_RE.findall(block)]


def _score_from_blocks(items1: list[str], items2: list[str]) -> str | None:
    """Winner-perspective set score; None when the sets do not align.

    The first score-item in each block is an empty spacer column and is
    dropped. Sets are zipped by index; a count mismatch (retired or incomplete
    set) and ambiguous tiebreak cells yield None so callers never store a
    misaligned score. Tiebreak points live on the set loser's cell ("6 4" in a
    "7-6(4)" set); bronze strips the parentheses later (see _score in ingest).
    """

    def without_spacer(items: list[str]) -> list[str]:
        return items[1:] if items and _TEXT_TAG_RE.sub("", items[0]) == "" else items

    player1, player2 = without_spacer(items1), without_spacer(items2)
    if not player1 or len(player1) != len(player2):
        return None
    sets: list[str] = []
    for cell1, cell2 in zip(player1, player2, strict=True):
        games1 = re.findall(r"\d+", cell1)
        games2 = re.findall(r"\d+", cell2)
        if len(games1) == len(games2) == 1:
            sets.append(f"{games1[0]}-{games2[0]}")
        elif len(games1) == 1 and len(games2) == 2 and int(games1[0]) > int(games2[0]):
            sets.append(f"{games1[0]}-{games2[0]}({games2[1]})")
        elif len(games2) == 1 and len(games1) == 2 and int(games2[0]) > int(games1[0]):
            sets.append(f"{games2[0]}-{games1[0]}({games1[1]})")
        else:
            return None
    return " ".join(sets)


def extract_matches_from_results(html: str, tournament_id: str, year: int) -> list[dict[str, Any]]:
    """Parse a tournament outcomes page into per-match discovery rows.

    One row per distinct ``msXXX`` stats link, bound to the card that carries it
    (the ``.match-footer``/``.match-cta`` block of that card, never paired with
    links from other cards): ``match_id`` (msXXX), ``round`` (bronze vocabulary),
    ``player1_id``/``player2_id`` (uppercased) plus their ``player1_slug``/
    ``player2_slug`` and ``player1_name``/``player2_name`` from the two
    ``.player-info`` overview links, ``winner_id`` (the player-info holding the
    winner checkmark), ``match_date`` from the enclosing ``.tournament-day``
    header, and ``score`` when the set cells align reliably (else None).

    Cards whose stats link is not an msXXX (qualifying qsXXX, doubles) are
    ignored; the tournament/year in the stats URL must match the arguments.
    """
    day_positions = [match.start() for match in _DAY_HEADER_RE.finditer(html)]
    card_positions = [match.start() for match in _MATCH_CARD_RE.finditer(html)]
    matches: list[dict[str, Any]] = []
    for index, card_start in enumerate(card_positions):
        card_end = card_positions[index + 1] if index + 1 < len(card_positions) else len(html)
        card = html[card_start:card_end]
        stats_link = _STATS_LINK_RE.search(card)
        if stats_link is None or stats_link.group(2) != tournament_id:
            continue
        if int(stats_link.group(1)) != year:
            continue

        day = next(
            (start for start in reversed(day_positions) if start < card_start),
            None,
        )
        match_date: date | None = None
        if day is not None:
            day_date = _DAY_DATE_RE.search(html[day:card_start])
            if day_date is not None:
                match_date = date(
                    int(day_date.group(3)), _MONTHS[day_date.group(2)], int(day_date.group(1))
                )

        round_match = _ROUND_STRONG_RE.search(card)
        players = list(_PLAYER_LINK_RE.finditer(card))
        player1 = players[0] if players else None
        player2 = players[1] if len(players) > 1 else None
        winner = _WINNER_DIV_RE.search(card)
        winner_is_player1 = winner is not None and (
            player2 is None or winner.start() < player2.start()
        )

        score: str | None = None
        stats = [m.start() for m in _STATS_ITEM_RE.finditer(card)]
        footer = _MATCH_FOOTER_RE.search(card)
        footer_start = footer.start() if footer is not None else len(card)
        if len(stats) >= 2:
            block1 = card[stats[0] : stats[1]]
            block2 = card[stats[1] : footer_start]
            score = _score_from_blocks(_score_items(block1), _score_items(block2))

        matches.append(
            {
                "match_id": stats_link.group(3),
                "round": _normalize_round(round_match.group(1)) if round_match else None,
                "player1_id": player1.group(2).upper() if player1 else None,
                "player1_slug": player1.group(1) if player1 else None,
                "player2_id": player2.group(2).upper() if player2 else None,
                "player2_slug": player2.group(1) if player2 else None,
                "player1_name": player1.group(3).strip() if player1 else None,
                "player2_name": player2.group(3).strip() if player2 else None,
                "winner_id": (
                    player1.group(2).upper()
                    if player1 and winner_is_player1
                    else player2.group(2).upper()
                    if player2
                    else None
                ),
                "match_date": match_date,
                "score": score,
            }
        )
    return matches


# Canonical physical-match key: (match_date, tournament_id, unordered canonical
# uppercase player ids). tournament_id scopes the player pair so the same
# players on different events/dates cannot collide; a frozenset (never a
# mutable dict order) keeps the key deterministic and orientation-free.
PhysicalKey = tuple[date, str, frozenset[str]]


def physical_key(match_date: date, tournament_id: str, player1: str, player2: str) -> PhysicalKey:
    """Canonical physical-match key: (match_date, tournament_id, {player ids}).

    Player ids are uppercased and folded into a frozenset, so a match is the
    same physical match regardless of page/winner-first orientation or id case.
    """
    return (match_date, tournament_id, frozenset({player1.upper(), player2.upper()}))


def ms_sequence(match_id: Any) -> int:
    """Positive numeric sequence from an ATP ``msNNN`` stats id; 0 when malformed.

    Only the strict ``msNNN`` shape is accepted — digits are never stripped out
    of a non-ms id — and a zero/non-positive sequence is rejected by callers.
    """
    raw = str(match_id or "").strip()
    digits = raw[2:] if raw.lower().startswith("ms") else ""
    return int(digits) if digits.isdigit() else 0


def build_match_id(year: int, tournament_id: str, sequence: int) -> str:
    """Canonical match id ``YYYY-TOURNAMENT_ID-NNN``, independent of match date.

    Shares the year-prefix rule with bronze/raw-CSV ingestion
    (``ingest.canonical_match_id``): a tournament id already starting with its
    edition year is embedded verbatim (``2026-418`` stays ``2026-418``), never
    a second prefix; otherwise the year is prepended once (``418`` + 2026 ->
    ``2026-418-026``, never ``2026-2026-418-026``). The id is opaque:
    dashed/nonstandard Davis Cup ids pass through untouched, never parsed as
    numeric. ``sequence`` is the zero-padded ATP ``msNNN`` numeric sequence
    (grows beyond 999 naturally; the caller rejects non-positive values). No
    date component: the same edition+tournament+sequence always names the same
    match.
    """
    return canonical_match_id(tournament_id, sequence, year)


def surface_for_tournament(
    tournament_id: str, existing_rows: list[dict[str, Any]] | None = None
) -> str | None:
    """Latest known surface for a tournament from existing bronze rows.

    Bronze has no tournament_id column, so a row belongs to the tournament when
    its ``match_id`` carries the id as a ``-``-separated segment (both the
    Sackmann ``YYYY-TOURNAMENT_ID-NNN`` and the match-stats
    ``YYYY-YYYY-TOURNAMENT_ID-NNN`` shapes embed it). Returns the surface of
    the newest matching row, None when the tournament is unknown — the caller
    skips/reports instead of inventing a surface. Purely over the supplied
    rows; nothing is scraped here.
    """
    if not existing_rows:
        return None
    token = f"-{tournament_id}-"
    known = [
        row for row in existing_rows if token in str(row.get("match_id", "")) and row.get("surface")
    ]
    if not known:
        return None
    newest = max(known, key=lambda row: str(row.get("match_date", "")))
    return str(newest["surface"])


def resolve_discovered_matches(
    matches: list[dict[str, Any]],
    rank_map: dict[str, str],
    canonical: dict[str, str] | None = None,
    tier: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve discovery rows to canonical bronze match rows; skip + report the rest.

    A row (``extract_matches_from_results`` output) resolves when the tournament
    ``tier`` is set and both player ids resolve via ``rankings.resolve_player_id``
    — exact reviewed-map id, exact canonical id, or a unique normalized-name
    variant. Resolved rows replace the page ids with canonical ids, keep the
    winner id consistent with the original page orientation, and stamp the
    bronze ``tournament`` tier. Every rejected row is returned with a ``reason``
    (unknown tournament tier, unresolved/ambiguous player) so the flow counts,
    skips, and reports — the reviewed identity map is never auto-updated here.
    """
    if not tier:
        return [], [{**match, "reason": "unknown tournament tier"} for match in matches]
    resolved: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for match in matches:
        player1 = rankings.resolve_player_id(
            str(match.get("player1_name") or ""),
            str(match.get("player1_id") or ""),
            rank_map,
            canonical,
        )
        player2 = rankings.resolve_player_id(
            str(match.get("player2_name") or ""),
            str(match.get("player2_id") or ""),
            rank_map,
            canonical,
        )
        if player1 is None or player2 is None:
            skipped.append(
                {
                    **match,
                    "reason": (
                        f"unresolved player: {match.get('player1_name')!r} "
                        f"({match.get('player1_id')}) / {match.get('player2_name')!r} "
                        f"({match.get('player2_id')})"
                    ),
                }
            )
            continue
        resolved.append(
            {
                **match,
                "player1_id": player1,
                "player2_id": player2,
                "winner_id": (
                    player1 if match.get("winner_id") == match.get("player1_id") else player2
                ),
                "tournament": tier,
            }
        )
    return resolved, skipped


# ── Hawkeye fetch + stats mapper ───────────────────────────────────

# Match-stats JSON endpoint (raw JSON body) and the two page URLs the flow
# navigates; later flow code formats all three from its discovery rows.
HAWKEYE_URL = (
    "https://www.atptour.com/-/Hawkeye/MatchStats/Complete/{year}/{tournament_id}/{match_id}"
)
RESULTS_ARCHIVE_URL = "https://www.atptour.com/en/scores/results-archive?year={year}"
TOURNAMENT_RESULTS_URL = (
    "https://www.atptour.com/en/scores/archive/{slug}/{tournament_id}/{year}/results"
)

# Per-request navigation budget for a Hawkeye fetch (same class of timeout the
# rankings flow uses for its pages); a timeout is a per-match skip, never an
# abort.
HAWKEYE_NAV_TIMEOUT_MS = 60_000
# Randomized human-like gap between Hawkeye requests (bot-detection hygiene).
HAWKEYE_SLEEP_MIN_S = 3.0
HAWKEYE_SLEEP_MAX_S = 8.0

# JSON is served raw, but CloakBrowser renders it wrapped in an HTML shell
# (<html>...<pre>{...}</pre>... — seen in the probes and the live run), so an
# HTML body is only a challenge when it carries no recoverable JSON object.
# _JSON_BODY_RE recovers that embedded object.
_HTML_BODY_RE = re.compile(r"<\s*(?:!doctype|html)\b", re.I)
_JSON_BODY_RE = re.compile(r"(\{.*\})", re.S)

# Match.Round.ShortName (F/SF/QF/...) -> bronze round vocabulary; unmapped
# rounds (qualifying etc.) resolve to None and the caller skips/reports.
_ROUND_SHORT_TO_BRONZE = {
    "F": "f",
    "SF": "sf",
    "QF": "qf",
    "R16": "r16",
    "R32": "r32",
    "R64": "r64",
    "R128": "r128",
    "RR": "rr",
}

# Tournament.EventType -> raw level code, then LEVEL_MAP to bronze tier (the
# archive badges use the same raw codes; slams use "GS", not "grandslam").
_EVENT_TYPE_TO_LEVEL = {"GS": "G", "1000": "M", "500": "500", "250": "250"}

# One team's nine bronze stat fields, each read from its ServiceStats holder:
# (payload key, subfield, bronze field suffix). Both teams' fields share the
# suffix; the mapper prefixes player1_/player2_.
_SIDE_STAT_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("Aces", "Number", "aces"),
    ("DoubleFaults", "Number", "double_faults"),
    ("FirstServe", "Dividend", "first_serves_made"),
    ("FirstServe", "Divisor", "total_serve_points"),
    ("FirstServePointsWon", "Dividend", "first_serve_points_won"),
    ("SecondServePointsWon", "Dividend", "second_serve_points_won"),
    ("ServiceGamesPlayed", "Number", "service_games"),
    ("BreakPointsSaved", "Dividend", "break_points_saved"),
    ("BreakPointsSaved", "Divisor", "break_points_faced"),
)


def _team_blocks(team: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-set stat blocks of one team, from either payload carrier.

    The 2026 payload renders them under ``Sets``, older payloads under
    ``SetScores``; both carry the identical per-block shape, so the mapper
    reads whichever is present.
    """
    blocks: list[dict[str, Any]] = []
    for carrier in ("Sets", "SetScores"):
        raw = team.get(carrier)
        if isinstance(raw, list):
            blocks.extend(block for block in raw if isinstance(block, dict))
    return blocks


def _side_stats(team: dict[str, Any]) -> tuple[dict[str, int] | None, str]:
    """Match-total stat fields for one team, or (None, reason) when unusable.

    The payload renders the whole-match totals on the block whose ``SetScore``
    is empty (older payloads carry stats only there); a payload without such a
    block sums its per-set blocks instead. Every required stat must be present
    and numeric — an absent stat is a structured skip, never a fabricated
    value. Explicit zeros are legal and kept (e.g. 0 break points faced).
    """
    blocks = _team_blocks(team)
    totals_block = next((block for block in blocks if block.get("SetScore") in (None, "")), None)
    blocks = [totals_block] if totals_block is not None else blocks
    if not blocks:
        return None, "no per-set stats blocks"
    totals: dict[str, int] = {}
    for block in blocks:
        service = block.get("Stats")
        if not isinstance(service, dict) or not isinstance(service.get("ServiceStats"), dict):
            return None, "stats block missing ServiceStats"
        service = service["ServiceStats"]
        for key, subfield, field in _SIDE_STAT_FIELDS:
            holder = service.get(key)
            value = holder.get(subfield) if isinstance(holder, dict) else None
            if value is None:
                return None, f"missing {key}.{subfield}"
            try:
                totals[field] = totals.get(field, 0) + int(value)
            except (TypeError, ValueError):
                return None, f"non-numeric {key}.{subfield}={value!r}"
    return totals, ""


def _score_from_teams(winner_team: dict[str, Any], loser_team: dict[str, Any]) -> str | None:
    """Winner-perspective set score from the payload; None when not derivable.

    The two teams' per-set blocks pair by index; blocks without a set score
    (the whole-match totals row) are skipped, and a set-count mismatch or
    non-numeric cell means the sets cannot be aligned reliably — None, never a
    fabricated score. Tiebreak digits are omitted, matching bronze's canonical
    score format (``_score`` in src/db/ingest.py).
    """
    winner_blocks = _team_blocks(winner_team)
    loser_blocks = _team_blocks(loser_team)
    if len(winner_blocks) != len(loser_blocks):
        return None
    sets: list[str] = []
    for winner_block, loser_block in zip(winner_blocks, loser_blocks, strict=True):
        winner_games, loser_games = (
            winner_block.get("SetScore"),
            loser_block.get("SetScore"),
        )
        if winner_games is None or loser_games is None:
            continue
        try:
            sets.append(f"{int(winner_games)}-{int(loser_games)}")
        except (TypeError, ValueError):
            return None
    return " ".join(sets) if sets else None


def _round_from_payload(match: dict[str, Any]) -> str | None:
    """Bronze round for Match.Round.ShortName; None when absent or unmapped."""
    round_info = match.get("Round")
    short = round_info.get("ShortName") if isinstance(round_info, dict) else None
    if not short:
        return None
    return _ROUND_SHORT_TO_BRONZE.get(str(short).upper())


def _tier_from_event_type(tournament: dict[str, Any]) -> str | None:
    """Bronze tier for Tournament.EventType; None when absent or non-tier."""
    level = _EVENT_TYPE_TO_LEVEL.get(str(tournament.get("EventType") or "").upper())
    return LEVEL_MAP.get(level) if level is not None else None


def _report(reason: str, discovered_match: dict[str, Any] | None) -> None:
    """Print a mapper skip reason (skip-and-report; the flow also counts Nones)."""
    match_id = discovered_match.get("match_id") if discovered_match else None
    prefix = f"  Hawkeye {match_id}: " if match_id else "  Hawkeye: "
    print(f"{prefix}skipped ({reason})")


def _lookup(values: dict[str, Any] | None, player_id: str, default: Any) -> Any:
    """rank_points/age lookup with the seed defaults (0 / 0.0) for unknowns."""
    if values is None:
        return default
    value = values.get(player_id)
    return default if value is None else value


def _field_text(value: Any) -> str:
    """Trimmed payload/scrape text; '' when absent."""
    if value is None:
        return ""
    return str(value).strip()


def _payload_int(value: Any) -> int | None:
    """Positive integer payload field (draw size, number of sets); None when absent."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _match_minutes(value: Any) -> int | None:
    """Total minutes from an ``HH:MM:SS`` payload duration; None when malformed."""
    parts = _field_text(value).split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = (int(part) for part in parts)
    except ValueError:
        return None
    return round(hours * 60 + minutes + seconds / 60)


def hawkeye_to_bronze(
    payload: dict[str, Any],
    discovered_match: dict[str, Any] | None = None,
    surface: str | None = None,
    rank_points: dict[str, int] | None = None,
    ages: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """Map a Hawkeye Complete stats payload into one winner-first bronze row.

    Side A = ``Match.PlayerTeam`` (identity ``Match.PlayerTeam1``), side B =
    ``Match.OpponentTeam`` (identity ``Match.PlayerTeam2``); the side whose id
    matches ``Match.WinningPlayerId``/``Match.Winner`` is the winner and
    supplies the player1 stats, because bronze enforces ``winner_id =
    player1_id``. The winner's canonical id comes from ``discovered_match``
    (a resolved discovery row) when present and consistent with the payload;
    otherwise the payload side ids are used verbatim.

    Returns None (after printing a reason) when the identities or winner are
    missing/inconsistent, or any of the 18 required stat fields is absent —
    the caller skips and reports instead of storing a plausible-but-invented
    row. The 18 stat columns always carry payload-derived ints (explicit
    zeros are legal). Metadata — match_id, match_date, tournament/tier,
    tournament_name, round, surface, score — comes from the payload or
    ``discovered_match`` when available and is None otherwise; ``surface``
    (bronze fallback), ``rank_points``, and ``ages`` are injected lookups
    with the seed defaults (0 / 0.0) for unknown players.

    The returned row also carries non-bronze extra keys for the raw CSV sink
    (never written to bronze): per-side seed/entry in winner/loser
    orientation, draw_size/best_of/minutes from the payload, and the
    discovered page names — each None when absent, never fabricated.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("Match"), dict):
        _report("payload has no Match object", discovered_match)
        return None
    match = payload["Match"]
    tournament_raw = payload.get("Tournament")
    tournament = tournament_raw if isinstance(tournament_raw, dict) else {}

    team_a, team_b = match.get("PlayerTeam"), match.get("OpponentTeam")
    identity_a, identity_b = match.get("PlayerTeam1"), match.get("PlayerTeam2")
    if not isinstance(team_a, dict) or not isinstance(team_b, dict):
        _report("payload missing PlayerTeam/OpponentTeam", discovered_match)
        return None
    if not isinstance(identity_a, dict) or not isinstance(identity_b, dict):
        _report("payload missing PlayerTeam1/PlayerTeam2 identities", discovered_match)
        return None

    a_id = str(identity_a.get("PlayerId") or "").upper()
    b_id = str(identity_b.get("PlayerId") or "").upper()
    if not a_id or not b_id:
        _report("side identity missing PlayerId", discovered_match)
        return None
    winner = str(match.get("WinningPlayerId") or match.get("Winner") or "").upper()
    if not winner:
        _report("payload has no WinningPlayerId/Winner", discovered_match)
        return None
    if winner not in (a_id, b_id):
        _report(
            f"winner {winner} matches no side identity ({a_id}/{b_id})",
            discovered_match,
        )
        return None
    winner_is_a = winner == a_id

    stats_a, reason_a = _side_stats(team_a)
    stats_b, reason_b = _side_stats(team_b)
    if stats_a is None:
        _report(f"side {a_id}: {reason_a}", discovered_match)
        return None
    if stats_b is None:
        _report(f"side {b_id}: {reason_b}", discovered_match)
        return None

    # Canonical player ids: the discovered (resolved) ids win when they agree
    # with the payload winner; otherwise the payload side ids are used. Any
    # disagreement is a structured skip, never an invented orientation.
    p1_canonical = str(discovered_match.get("player1_id") or "").upper() if discovered_match else ""
    p2_canonical = str(discovered_match.get("player2_id") or "").upper() if discovered_match else ""
    discovered_winner = (
        str(discovered_match.get("winner_id") or "").upper() if discovered_match else ""
    )
    if p1_canonical and p2_canonical:
        if p1_canonical == winner and (not discovered_winner or discovered_winner == winner):
            player1_id, player2_id = p1_canonical, p2_canonical
        elif p2_canonical == winner and (not discovered_winner or discovered_winner == winner):
            player1_id, player2_id = p2_canonical, p1_canonical
        else:
            _report(
                f"discovered players {p1_canonical}/{p2_canonical} inconsistent "
                f"with payload winner {winner}",
                discovered_match,
            )
            return None
    elif discovered_winner:
        if discovered_winner != winner:
            _report(
                f"discovered winner {discovered_winner} != payload winner {winner}",
                discovered_match,
            )
            return None
        player1_id, player2_id = winner, (b_id if winner_is_a else a_id)
    else:
        player1_id, player2_id = (a_id, b_id) if winner_is_a else (b_id, a_id)

    winner_stats = stats_a if winner_is_a else stats_b
    loser_stats = stats_b if winner_is_a else stats_a

    score = _score_from_teams(team_a if winner_is_a else team_b, team_b if winner_is_a else team_a)
    if score is None and discovered_match and discovered_match.get("score"):
        # Bronze canonical format strips tiebreak digits (_score in ingest).
        score = re.sub(r"\(\d+\)", "", str(discovered_match["score"])).strip() or None

    round_value = (
        str(discovered_match["round"])
        if discovered_match and discovered_match.get("round")
        else _round_from_payload(match)
    )
    tier = (
        str(discovered_match["tournament"])
        if discovered_match and discovered_match.get("tournament")
        else _tier_from_event_type(tournament)
    )
    tournament_name = tournament.get("EventDisplayName") or tournament.get("TournamentName")

    # Raw-CSV source metadata (extra row keys, never bronze columns): per-side
    # seed/entry follow the payload winner orientation, draw_size/best_of/
    # minutes come from the payload when present — absent values stay None.
    winner_identity, loser_identity = (
        (identity_a, identity_b)
        if winner_is_a
        else (
            identity_b,
            identity_a,
        )
    )
    source_metadata: dict[str, Any] = {
        "winner_seed": _field_text(winner_identity.get("SeedPlayerTeam")),
        "winner_entry": _field_text(winner_identity.get("EntryStatusPlayerTeam")),
        "loser_seed": _field_text(loser_identity.get("SeedPlayerTeam")),
        "loser_entry": _field_text(loser_identity.get("EntryStatusPlayerTeam")),
        "draw_size": _payload_int(tournament.get("Singles")),
        "best_of": _payload_int(match.get("NumberOfSets")),
        "minutes": _match_minutes(match.get("MatchTime") or match.get("MatchTimeTotal")),
    }
    if discovered_match:
        source_metadata["player1_name"] = _field_text(discovered_match.get("player1_name"))
        source_metadata["player2_name"] = _field_text(discovered_match.get("player2_name"))

    row = dict.fromkeys(BRONZE_COLUMNS)
    row.update(
        {
            "match_id": discovered_match.get("match_id") if discovered_match else None,
            "match_date": _as_date(discovered_match.get("match_date"))
            if discovered_match
            else None,
            "player1_id": player1_id,
            "player2_id": player2_id,
            "tournament": tier,
            "tournament_name": str(tournament_name) if tournament_name else None,
            "round": round_value,
            "surface": surface or _canonical_surface(tournament.get("Court")),
            "score": score,
        }
    )
    for side, stats in (("player1", winner_stats), ("player2", loser_stats)):
        for field, value in stats.items():
            row[f"{side}_{field}"] = value
    row["player1_rank_points"] = _lookup(rank_points, player1_id, 0)
    row["player2_rank_points"] = _lookup(rank_points, player2_id, 0)
    row["player1_age"] = _lookup(ages, player1_id, 0.0)
    row["player2_age"] = _lookup(ages, player2_id, 0.0)
    row["winner_id"] = player1_id
    row.update({key: (value or None) for key, value in source_metadata.items()})
    return row


def fetch_hawkeye_match(
    page: Any,
    year: int | str,
    tournament_id: str,
    match_id: str,
) -> tuple[dict[str, Any] | None, str]:
    """Fetch one Hawkeye Complete stats payload; (payload, "") or (None, reason).

    Never raises on a bad response: navigation failures/timeouts, challenge or
    HTML responses, invalid JSON, and payloads missing the stats/identity
    blocks are classified and returned as a skip reason, so the batch loop
    continues to the next match. Uses the run's shared page (the batch owns
    pacing: ``rankings._jitter()`` here, the 3-8s gap between requests).
    """
    url = HAWKEYE_URL.format(year=year, tournament_id=tournament_id, match_id=match_id)
    rankings._jitter()
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=HAWKEYE_NAV_TIMEOUT_MS)
    except Exception as exc:
        return None, f"navigation failed ({type(exc).__name__}: {exc})"
    try:
        body = page.content()
    except Exception as exc:
        return None, f"page content failed ({type(exc).__name__}: {exc})"
    if _HTML_BODY_RE.search(body[:2048]):
        # CloakBrowser wraps the raw JSON in an HTML shell; recover it before
        # classifying the response as a challenge.
        embedded = _JSON_BODY_RE.search(body)
        if embedded is None:
            return None, f"HTML/challenge response ({len(body)} bytes)"
        try:
            payload = json.loads(embedded.group(1))
        except ValueError:
            return None, f"HTML/challenge response ({len(body)} bytes)"
    else:
        try:
            payload = json.loads(body)
        except ValueError:
            return None, f"invalid JSON response ({len(body)} bytes)"
    if not isinstance(payload, dict):
        return None, f"unexpected JSON shape ({type(payload).__name__})"
    match = payload.get("Match")
    if (
        not isinstance(match, dict)
        or not isinstance(match.get("PlayerTeam"), dict)
        or not isinstance(match.get("OpponentTeam"), dict)
    ):
        return None, "payload missing Match/PlayerTeam/OpponentTeam"
    return payload, ""


def fetch_hawkeye_batch(
    matches: list[dict[str, Any]],
    *,
    year: int | str,
    tournament_id: str,
    page: Any | None = None,
) -> list[dict[str, Any]]:
    """Fetch Hawkeye stats for every discovered match, sequentially.

    Launches the shared ``rankings._launch_browser`` (persistent profile,
    ``humanize=True``) when no ``page`` is supplied, keeps one page for the
    whole batch, and closes the browser in ``finally`` — one request at a time
    with a randomized 3-8s gap plus ``_jitter`` before each navigation. Each
    input row comes back with ``payload`` (dict or None) and ``hawkeye_error``
    (skip reason or None); a bad match never aborts the batch.
    """
    browser = None
    owned_page = None
    if page is None:
        browser = rankings._launch_browser()
        owned_page = browser.new_page()
        page = owned_page
    results: list[dict[str, Any]] = []
    try:
        for match in matches:
            match_id = str(match.get("match_id") or "")
            time.sleep(random.uniform(HAWKEYE_SLEEP_MIN_S, HAWKEYE_SLEEP_MAX_S))
            payload, reason = fetch_hawkeye_match(page, year, tournament_id, match_id)
            if reason:
                print(f"  Hawkeye {match_id}: skipped ({reason})")
            else:
                print(f"  Hawkeye {match_id}: fetched")
            results.append({**match, "payload": payload, "hawkeye_error": reason or None})
    finally:
        if browser is not None:
            print("Closing browser session")
            if owned_page is not None:
                owned_page.close()
            browser.close()
    return results


# ── Bronze upsert, insert-or-force-replace ─────────────────────────
#
# An existing match_id is skipped by default — no write, no selective stat
# fills. Only an explicit force run replaces the stored row across every
# non-key column (ON CONFLICT DO UPDATE), so repeated force runs converge to
# the candidate values.


def _python_scalar(value: Any) -> Any:
    """Python scalar for reuse/copy: timestamps -> date, numpy scalars -> plain."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if type(value).__module__ == "numpy":
        return value.item()
    return value


def _missing_value(value: Any) -> bool:
    """Whether a bronze count cell carries the missing sentinel (0/NULL/blank)."""
    if value is None or pd.isna(value):
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip().lower() == "nan"
    try:
        return int(value) == 0
    except (TypeError, ValueError):
        return False


def _candidate_absent(value: Any) -> bool:
    """Whether a candidate row carries NO value for a column (only NULL/blank).

    Distinct from ``_missing_value``: an explicit zero is real observed data
    (e.g. 0 double faults), never absence, so candidate zeros are fillable and
    insert-valid. None/NaN/blank alone mean the candidate has no value.
    """
    if value is None or pd.isna(value):
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip().lower() == "nan"
    return False


def validate_new_bronze_row(row: dict[str, Any]) -> str | None:
    """Skip reason for a candidate insert; None when the row is insert-ready.

    A new bronze.match_events row must be complete and winner-first (the
    schema enforces ``winner_id = player1_id`` and NOT NULL on the identity,
    tier, surface, 18 stat, rank_points, and age columns): every required
    field present, distinct player ids, canonical surface, a parseable
    match_date, and numeric stats. An explicit zero in a stat/rank_points/age
    cell is valid data (the payload observed zero), not a missing field — only
    a NULL/blank cell is. Returns a human-readable reason instead of raising,
    so the flow can count, skip, and report the row.
    """
    required = (
        "match_id",
        "match_date",
        "player1_id",
        "player2_id",
        "tournament",
        "surface",
        "winner_id",
    )
    for column in required:
        if _missing_value(row.get(column)):
            return f"missing required field {column}"
    if row["player1_id"] == row["player2_id"]:
        return "player1_id equals player2_id"
    if row["winner_id"] != row["player1_id"]:
        return "winner_id != player1_id (bronze requires winner-first rows)"
    if row["surface"] not in CANONICAL_SURFACES:
        return f"non-canonical surface {row['surface']!r}"
    if _as_date(row["match_date"]) is None:
        return f"unparseable match_date {row['match_date']!r}"
    for column in BRONZE_COLUMNS_INT:
        value = row.get(column)
        if value is None or _candidate_absent(value):
            return f"missing stat field {column}"
        try:
            int(value)
        except (TypeError, ValueError):
            return f"non-numeric stat {column}={value!r}"
    for column in (*BRONZE_COLUMNS_INT32, *BRONZE_COLUMNS_FLOAT):
        if _candidate_absent(row.get(column)):
            return f"missing field {column}"
    return None


def _to_frame(row: dict[str, Any]) -> pd.DataFrame:
    """One bronze row as a DataFrame over BRONZE_COLUMNS, in schema order."""
    return pd.DataFrame(
        [{column: row.get(column) for column in BRONZE_COLUMNS}],
        columns=list(BRONZE_COLUMNS),
    )


def find_existing_match(
    match_id: str,
    query: Callable[[str, list[object]], Any] | None = None,
) -> dict[str, Any] | None:
    """Stored bronze.match_events row for match_id, or None when absent.

    ``query`` is injectable for hermetic tests (the plan's in-memory fixture):
    a callable ``(sql, params) -> DataFrame | sequence[Mapping]`` defaulting to
    ``src.db.client.execute_df``. Returned values are normalized to Python
    scalars (timestamps to ``date``, numpy scalars to plain int/float) so
    callers can compare and reuse them without pandas/numpy interop.
    """
    if query is None:
        query = execute_df
    result = query(f"SELECT * FROM {BRONZE_MATCHES_TABLE} WHERE match_id = %s", [match_id])
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return None
        row = first_row_dict(result)
    else:
        rows = list(result)
        if not rows:
            return None
        row = dict(rows[0])
    return {column: _python_scalar(row.get(column)) for column in BRONZE_COLUMNS}


def upsert_bronze_match(
    row: dict[str, Any],
    *,
    force: bool = False,
    query: Callable[[str, list[object]], Any] | None = None,
) -> dict[str, Any]:
    """Upsert one winner-first bronze row; returns a result record for the flow.

    Existing match_id: skipped by default (a ``noop`` record — no write, no
    selective stat updates). Only with ``force=True`` is the stored row
    replaced across every non-key column (``update_cols`` = all bronze columns
    except ``match_id``), so repeated force runs converge to the candidate
    values.

    New match_id: ``validate_new_bronze_row`` must pass (complete winner-first
    row). The insert uses ``_copy_df_into`` with ON CONFLICT DO NOTHING, so a
    repeated insert never duplicates and a concurrent insert is a noop.

    Record shape: {match_id, action: inserted|updated|noop|skipped, reason,
    update_cols, rows_affected}. Expected per-row failures (validation or DB
    write errors) become ``skipped`` records with a reason — never swallowed,
    never raised — so the flow can count and report each one.
    """
    match_id = str(row.get("match_id") or "")
    existing = find_existing_match(match_id, query=query)
    if existing is not None:
        if not force:
            return {
                "match_id": match_id,
                "action": "noop",
                "reason": None,
                "update_cols": [],
                "rows_affected": 0,
            }
        update_cols = [column for column in BRONZE_COLUMNS if column != "match_id"]
        try:
            affected = _copy_df_into(
                BRONZE_MATCHES_TABLE,
                _to_frame({**existing, **row}),
                conflict_col="match_id",
                update_cols=update_cols,
            )
        except Exception as exc:
            return {
                "match_id": match_id,
                "action": "skipped",
                "reason": f"update failed: {type(exc).__name__}: {exc}",
                "update_cols": update_cols,
                "rows_affected": 0,
            }
        return {
            "match_id": match_id,
            "action": "updated",
            "reason": None,
            "update_cols": update_cols,
            "rows_affected": affected,
        }

    reason = validate_new_bronze_row(row)
    if reason is not None:
        return {
            "match_id": match_id,
            "action": "skipped",
            "reason": reason,
            "update_cols": [],
            "rows_affected": 0,
        }
    try:
        affected = _copy_df_into(
            BRONZE_MATCHES_TABLE,
            _to_frame(row),
            conflict_col="match_id",
            update_cols=None,
        )
    except Exception as exc:
        return {
            "match_id": match_id,
            "action": "skipped",
            "reason": f"insert failed: {type(exc).__name__}: {exc}",
            "update_cols": [],
            "rows_affected": 0,
        }
    return {
        "match_id": match_id,
        "action": "inserted" if affected else "noop",
        "reason": None,
        "update_cols": [],
        "rows_affected": affected,
    }


def upsert_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts over upsert result records, for the flow's end-of-run report.

    Folds ``upsert_bronze_match`` records into {inserted, updated, skipped,
    noop, skipped_reasons}; the ``discovered``/``fetched`` counts come from the
    flow's discovery and fetch phases and are not part of this fold. Every skip
    reason is retained — nothing is swallowed.
    """
    summary: dict[str, Any] = {
        "inserted": 0,
        "updated": 0,
        "skipped": 0,
        "noop": 0,
        "skipped_reasons": [],
    }
    for record in results:
        action = record.get("action")
        if action in summary:
            summary[action] += 1
        if record.get("reason"):
            summary["skipped_reasons"].append(f"{record.get('match_id')}: {record['reason']}")
    return summary


# ── Sackmann raw CSV persistence (shared sink with the bronze upsert) ─

# Full Sackmann match-CSV header, in file order — the columns of
# data/raw/{year}.csv (seed.py convention). Appended rows keep this exact
# column/order/format; fields fill from the scrape (payload-derived source
# metadata), the per-run player profile map (names/hand/height/IOC), or stay
# empty when neither knows them.
RAW_MATCH_COLUMNS = [
    "tourney_id",
    "tourney_name",
    "surface",
    "draw_size",
    "tourney_level",
    "indoor",
    "tourney_date",
    "match_num",
    "winner_id",
    "winner_seed",
    "winner_entry",
    "winner_name",
    "winner_hand",
    "winner_ht",
    "winner_ioc",
    "winner_age",
    "winner_rank",
    "winner_rank_points",
    "loser_id",
    "loser_seed",
    "loser_entry",
    "loser_name",
    "loser_hand",
    "loser_ht",
    "loser_ioc",
    "loser_age",
    "loser_rank",
    "loser_rank_points",
    "score",
    "best_of",
    "round",
    "minutes",
    "w_ace",
    "w_df",
    "w_svpt",
    "w_1stIn",
    "w_1stWon",
    "w_2ndWon",
    "w_SvGms",
    "w_bpSaved",
    "w_bpFaced",
    "l_ace",
    "l_df",
    "l_svpt",
    "l_1stIn",
    "l_1stWon",
    "l_2ndWon",
    "l_SvGms",
    "l_bpSaved",
    "l_bpFaced",
]

# Bronze tier -> raw tourney_level code (raw CSV vocabulary; the scrape's
# EventType codes map through LEVEL_MAP to the same values).
_BRONZE_TIER_TO_LEVEL = {
    "grand_slam": "G",
    "masters": "M",
    "atp_500": "500",
    "atp_250": "250",
}

_INDOR_TO_RAW: dict[int | None, str] = {0: "O", 1: "I"}

# Bronze player1_*/player2_* stat suffix -> raw winner-side column.
_BRONZE_TO_RAW_STATS = {
    "aces": "w_ace",
    "double_faults": "w_df",
    "first_serves_made": "w_1stIn",
    "total_serve_points": "w_svpt",
    "first_serve_points_won": "w_1stWon",
    "second_serve_points_won": "w_2ndWon",
    "service_games": "w_SvGms",
    "break_points_saved": "w_bpSaved",
    "break_points_faced": "w_bpFaced",
}


def raw_match_path(year: int, raw_dir: Path | None = None) -> Path:
    """Sackmann CSV for a year: ``data/raw/{year}.csv`` (seed.py convention)."""
    return (raw_dir or ROOT / "data" / "raw") / f"{year}.csv"


def _fmt_num(value: Any) -> str:
    """Raw-cell string for a number; blank for the unknown marker (0/None)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number == 0 or number != number:  # 0 / NaN are the unknown markers
        return ""
    return str(int(number)) if number.is_integer() else f"{number:.3f}"


def bronze_row_to_raw_match(
    row: dict[str, Any], profiles: dict[str, dict[str, str]] | None = None
) -> dict[str, str]:
    """One Sackmann-format raw row from a bronze match_events row.

    The canonical match_id embeds the raw tourney_id and match sequence, so it
    round-trips: ``2026-418-026`` -> tourney_id ``2026-418``, match_num 26, and
    re-deriving the id through the shared rule reproduces ``2026-418-026``.
    ``profiles`` is the per-run in-memory player metadata map
    (``load_player_metadata``): it supplies display name, hand, height, and IOC
    for winner/loser when the player is known (the discovered page names on the
    row are the fallback). The payload-derived source fields (per-side seed and
    entry, draw_size, best_of, minutes) are stamped extra keys on the bronze row
    by ``hawkeye_to_bronze``; stats/ranks/ages map from the bronze columns. Any
    field the scrape cannot know stays empty — nothing is fabricated.
    """
    match_id = str(row.get("match_id") or "")
    tourney_id, sep, seq = match_id.rpartition("-")
    if not sep:
        return dict.fromkeys(RAW_MATCH_COLUMNS, "")
    match_date = _as_date(row.get("match_date"))
    winner_id = str(row.get("winner_id") or "")
    loser_id = str(row.get("player2_id") or "")
    winner_profile = (profiles or {}).get(winner_id.upper()) or {}
    loser_profile = (profiles or {}).get(loser_id.upper()) or {}
    raw = dict.fromkeys(RAW_MATCH_COLUMNS, "")
    raw.update(
        {
            "tourney_id": tourney_id,
            "tourney_name": str(row.get("tournament_name") or ""),
            "surface": str(row.get("surface") or "").capitalize(),
            "draw_size": _fmt_num(row.get("draw_size")),
            "tourney_level": _BRONZE_TIER_TO_LEVEL.get(str(row.get("tournament") or ""), ""),
            "indoor": _INDOR_TO_RAW.get(row.get("is_indoor"), ""),
            "tourney_date": match_date.strftime("%Y%m%d") if match_date else "",
            "match_num": str(int(seq)),
            "winner_id": winner_id,
            "winner_seed": str(row.get("winner_seed") or ""),
            "winner_entry": str(row.get("winner_entry") or ""),
            "winner_name": winner_profile.get("display_name") or str(row.get("player1_name") or ""),
            "winner_hand": winner_profile.get("hand") or "",
            "winner_ht": winner_profile.get("height") or "",
            "winner_ioc": winner_profile.get("ioc") or "",
            "winner_age": _fmt_num(row.get("player1_age")),
            "winner_rank": _fmt_num(row.get("player1_ranking")),
            "winner_rank_points": _fmt_num(row.get("player1_rank_points")),
            "loser_id": loser_id,
            "loser_seed": str(row.get("loser_seed") or ""),
            "loser_entry": str(row.get("loser_entry") or ""),
            "loser_name": loser_profile.get("display_name") or str(row.get("player2_name") or ""),
            "loser_hand": loser_profile.get("hand") or "",
            "loser_ht": loser_profile.get("height") or "",
            "loser_ioc": loser_profile.get("ioc") or "",
            "loser_age": _fmt_num(row.get("player2_age")),
            "loser_rank": _fmt_num(row.get("player2_ranking")),
            "loser_rank_points": _fmt_num(row.get("player2_rank_points")),
            "score": str(row.get("score") or ""),
            "best_of": _fmt_num(row.get("best_of")),
            "round": str(row.get("round") or "").upper(),
            "minutes": _fmt_num(row.get("minutes")),
        }
    )
    for suffix, winner_col in _BRONZE_TO_RAW_STATS.items():
        raw[winner_col] = str(int(row.get(f"player1_{suffix}") or 0))
        raw[winner_col.replace("w_", "l_", 1)] = str(int(row.get(f"player2_{suffix}") or 0))
    return raw


def _csv_row_match_id(row: dict[str, str], year: int | None = None) -> str | None:
    """Canonical match id of one raw CSV row (from its tourney_id + match_num).

    The same rule as bronze ingestion, so the CSV and DB dedup sets agree:
    ``2026-418`` + 26 -> ``2026-418-026``. ``year`` falls back to the leading
    four digits of the row's tourney_date when the id itself carries none.
    """
    tourney_id = str(row.get("tourney_id") or "")
    match_num = str(row.get("match_num") or "")
    if not tourney_id or not match_num:
        return None
    if year is None:
        date_text = str(row.get("tourney_date") or "")
        year = int(date_text[:4]) if date_text[:4].isdigit() else None
    return canonical_match_id(tourney_id, int(match_num), year)


def load_csv_match_ids(path: Path) -> set[str]:
    """Canonical match ids already present in a Sackmann CSV; empty for a new file.

    Read once per run per year; the flow consults this set before every append,
    so a rescrape never duplicates a row already on disk (or appended earlier
    in this run). Existing rows are only ever read, never rewritten.
    """
    if not path.exists() or path.stat().st_size == 0:
        return set()
    ids: set[str] = set()
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            match_id = _csv_row_match_id(row)
            if match_id is not None:
                ids.add(match_id)
    return ids


def append_raw_match_rows(
    rows: list[dict[str, str]],
    path: Path,
    *,
    existing: set[str] | None = None,
) -> tuple[int, set[str]]:
    """Append only genuinely new Sackmann rows to a CSV; (appended, all ids).

    ``existing`` is the pre-loaded id set when supplied (dedup guard), otherwise
    loaded from the file. Rows already present (same canonical match id) are
    skipped, including repeats earlier in the same batch; a brand-new file gets
    the header first; an existing file is only appended to, so its rows are
    byte-for-byte untouched. Returns the number of rows actually appended and
    the full (existing + appended) id set.
    """
    ids = set(existing) if existing is not None else load_csv_match_ids(path)
    new_rows: list[dict[str, str]] = []
    for row in rows:
        match_id = _csv_row_match_id(row)
        if match_id is not None and match_id in ids:
            continue
        new_rows.append(row)
        if match_id is not None:
            ids.add(match_id)
    if not new_rows:
        return 0, ids
    create = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RAW_MATCH_COLUMNS)
        if create:
            writer.writeheader()
        writer.writerows(new_rows)
    return len(new_rows), ids


# ── Flow orchestration: window, archive, tournaments, enrichment ──

# Tier eligibility: only tour-level singles events qualify; everything else
# (atp_finals, davis_cup, olympics, challengers, unknown badges) is skipped and
# reported, never guessed.
TIER_ELIGIBLE = frozenset({"grand_slam", "masters", "atp_500", "atp_250"})

# Per-page navigation budget for archive/results pages (same class of timeout as
# the Hawkeye requests; a slow page is a per-item skip, never an abort).
RESULTS_NAV_TIMEOUT_MS = HAWKEYE_NAV_TIMEOUT_MS

# Bronze.match_events projection loaded once per run: the flow's fallback
# lookups (physical-match id reuse, tier/surface for unknown tournaments, and
# per-player rank_points/age) all read from these columns.
BRONZE_LOOKUP_COLUMNS = (
    "match_id",
    "match_date",
    "player1_id",
    "player2_id",
    "tournament",
    "surface",
    "player1_rank_points",
    "player2_rank_points",
    "player1_age",
    "player2_age",
)


def matches_watermark() -> date | None:
    """Latest stored match_date in bronze.match_events, or None when empty."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT MAX(match_date) FROM {BRONZE_MATCHES_TABLE}")
        row = cur.fetchone()
    return row[0] if row is not None and row[0] is not None else None


def resolve_window(
    start_date: date | None,
    end_date: date | None,
    watermark: date | None,
    today: date | None = None,
) -> tuple[date, date] | None:
    """Inclusive [start, end] scrape window, or None when there is nothing.

    Explicit dates are used exactly: both -> [start, end]; only ``start_date``
    -> [start, today]; only ``end_date`` -> [watermark, end]. With no dates (a
    scheduled run) the window defaults to [watermark, today], so a bare run
    resumes from the last stored match and never crawls unbounded history. The
    start needs the watermark whenever it is not explicit: an empty bronze
    table with no ``start_date``, or an ``end_date`` at/before the watermark,
    resolves to None so the caller reports and skips. ``today`` is injectable
    for hermetic tests.
    """
    today = today or date.today()
    if start_date is not None and end_date is not None:
        return (start_date, end_date) if start_date <= end_date else None
    if start_date is not None:
        return (start_date, today)
    if end_date is not None:
        if watermark is None or end_date < watermark:
            return None
        return (watermark, end_date)
    if watermark is None or today < watermark:
        return None
    return (watermark, today)


def load_bronze_lookup_rows(
    query: Callable[[str, list[object]], Any] | None = None,
) -> list[dict[str, Any]]:
    """All bronze.match_events rows projected for the flow's lookups, once per run.

    Values are normalized to Python scalars (``_python_scalar``) so the
    physical-match, rank_points, and age indexes compare cleanly. ``query`` is
    injectable for hermetic tests (default ``src.db.client.execute_df``).
    """
    if query is None:
        query = execute_df
    columns = ", ".join(BRONZE_LOOKUP_COLUMNS)
    frame = query(f"SELECT {columns} FROM {BRONZE_MATCHES_TABLE}", [])
    return [
        {column: _python_scalar(record.get(column)) for column in BRONZE_LOOKUP_COLUMNS}
        for record in frame.to_dict(orient="records")
    ]


def _physical_match_index(
    rows: list[dict[str, Any]],
) -> dict[tuple[date, frozenset[str]], str]:
    """{ (match_date, {player1_id, player2_id}): match_id } over stored rows.

    Bronze-side projection of the canonical physical key (see
    ``physical_key``): bronze rows carry no tournament id, so the tournament
    id is fixed by the page being processed. Uppercased ids in a frozenset
    keep the lookup orientation-free and case-insensitive, so reusing a stored
    match_id makes enrichment update that row instead of inserting a
    duplicate.
    """
    index: dict[tuple[date, frozenset[str]], str] = {}
    for row in rows:
        match_date = row.get("match_date")
        player1, player2 = row.get("player1_id"), row.get("player2_id")
        if match_date is None or not player1 or not player2 or not row.get("match_id"):
            continue
        index[(match_date, frozenset({str(player1).upper(), str(player2).upper()}))] = str(
            row["match_id"]
        )
    return index


def _known_match_ids(
    rows: list[dict[str, Any]],
) -> dict[str, tuple[date, frozenset[str]]]:
    """{stored match_id: (match_date, {player ids})} — id ownership from bronze.

    Inverse projection of ``_physical_match_index``: a derived/reused id that
    already names a different physical match in bronze is a collision the flow
    must skip, never merge (``_id_collision``).
    """
    known: dict[str, tuple[date, frozenset[str]]] = {}
    for row in rows:
        match_id = str(row.get("match_id") or "")
        match_date = row.get("match_date")
        player1, player2 = row.get("player1_id"), row.get("player2_id")
        if not match_id or match_date is None or not player1 or not player2:
            continue
        known[match_id] = (
            match_date,
            frozenset({str(player1).upper(), str(player2).upper()}),
        )
    return known


def _latest_rank_points_and_ages(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, float]]:
    """Newest stored per-player rank_points/age from bronze rows.

    Match-time values from the raw ATP seed material (0/0.0 are the ingest
    missing markers, consistent with ``seed.py``); unknown players stay absent
    and the mapper's ``_lookup`` defaults them to 0 / 0.0.
    """
    rank_points: dict[str, int] = {}
    ages: dict[str, float] = {}
    ordered = sorted(rows, key=lambda row: str(row.get("match_date") or ""))
    for row in ordered:
        if row.get("match_date") is None:
            continue
        for side in ("player1", "player2"):
            player_id = row.get(f"{side}_id")
            if not player_id:
                continue
            points = row.get(f"{side}_rank_points")
            if points and float(points) > 0:
                rank_points[str(player_id)] = int(points)
            age = row.get(f"{side}_age")
            if age and float(age) > 0:
                ages[str(player_id)] = float(age)
    return rank_points, ages


def _bronze_match_id(
    item: dict[str, Any],
    physical: dict[tuple[date, frozenset[str]], str],
    year: int,
    tournament_id: str,
) -> tuple[str, str, PhysicalKey | None]:
    """(bronze match_id, skip reason or "", canonical physical key) for one row.

    Reuses the stored physical match's id when the canonical
    (date, tournament, {player ids}) key is already in bronze — orientation-free
    because the key's player set is unordered — so enrichment updates that row.
    Otherwise derives the deterministic Sackmann id (``build_match_id``) from
    the strictly-parsed ``msNNN`` sequence (``ms_sequence``), zero-padded. A
    missing date or a malformed/non-positive ms id is a skip, never a
    fabricated id. The key is None exactly when a reason is returned.
    """
    match_date = _as_date(item.get("match_date"))
    if match_date is None:
        return "", f"missing match_date for {item.get('match_id')!r}", None
    player1, player2 = str(item.get("player1_id") or ""), str(item.get("player2_id") or "")
    key = physical_key(match_date, tournament_id, player1, player2)
    existing = physical.get((match_date, frozenset({player1.upper(), player2.upper()})))
    if existing:
        return existing, "", key
    sequence = ms_sequence(item.get("match_id"))
    if sequence <= 0:
        return "", f"no positive ms sequence in {item.get('match_id')!r}", key
    return build_match_id(year, tournament_id, sequence), "", key


def dedupe_physical_matches(
    matches: list[dict[str, Any]], tournament_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop repeated physical keys (same date/tournament/player pair) pre-fetch.

    Keeps the first occurrence so the Hawkeye fetch and upsert run once per
    physical match; rows without a resolvable key (missing date or player) pass
    through untouched and are skipped later by the caller. Returns
    (unique matches, dropped duplicates).
    """
    seen: set[PhysicalKey] = set()
    unique: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for match in matches:
        match_date = _as_date(match.get("match_date"))
        player1, player2 = str(match.get("player1_id") or ""), str(match.get("player2_id") or "")
        if match_date is None or not player1 or not player2:
            unique.append(match)
            continue
        key = physical_key(match_date, tournament_id, player1, player2)
        if key in seen:
            dropped.append(match)
            continue
        seen.add(key)
        unique.append(match)
    return unique, dropped


def _id_collision(
    match_id: str,
    key: PhysicalKey,
    known_ids: dict[str, tuple[date, frozenset[str]]],
    claimed: dict[str, PhysicalKey],
) -> str:
    """Skip reason when ``match_id`` is already owned by a different match; "" otherwise.

    ``known_ids`` is bronze id ownership (date + unordered player set — bronze
    rows carry no tournament id, which is fixed by the page being processed);
    ``claimed`` tracks ids derived/reused earlier in this run. An id naming the
    same physical key is fine (idempotent reuse); an id naming a different one
    is a collision the caller skips instead of merging two matches.
    """
    match_date, _, players = key
    bronze_key = known_ids.get(match_id)
    if bronze_key is not None and bronze_key != (match_date, players):
        return f"match_id {match_id} collides with a different bronze match"
    prior = claimed.get(match_id)
    if prior is not None and prior != key:
        return f"match_id {match_id} collides with a different match this run"
    return ""


def _fetch_page(page: Any, url: str, label: str) -> tuple[str, str]:
    """(html, "") or ("", reason) — one shared page, jitter around navigation.

    The run's single browser page navigates every archive year, tournament
    results page, and Hawkeye URL, so the persistent profile's Cloudflare
    clearance carries across all of them. Humanize discipline matches the
    rankings flow: ``rankings._jitter()`` before and after the goto; the 3-8s
    tournament/request pacing lives in the callers. A navigation/content
    failure is classified into a skip reason — never raised, so one bad page
    does not abort the run.
    """
    rankings._jitter()
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=RESULTS_NAV_TIMEOUT_MS)
    except Exception as exc:
        return "", f"{label}: navigation failed ({type(exc).__name__}: {exc})"
    rankings._jitter()
    try:
        body = page.content()
    except Exception as exc:
        return "", f"{label}: page content failed ({type(exc).__name__}: {exc})"
    if not body or not body.strip():
        return "", f"{label}: empty page content"
    return body, ""


def _process_tournament(
    page: Any,
    tournament: dict[str, Any],
    year: int,
    *,
    tier: str,
    surface: str | None,
    physical: dict[tuple[date, frozenset[str]], str],
    known_ids: dict[str, tuple[date, frozenset[str]]],
    claimed: dict[str, PhysicalKey],
    rank_points: dict[str, int],
    ages: dict[str, float],
    rank_map: dict[str, str],
    canonical: dict[str, str] | None,
    profiles: dict[str, dict[str, str]] | None = None,
    csv_ids: dict[int, set[str]] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Discover, enrich, upsert, and CSV-append one tournament's matches.

    Fetches the results page after a generous 3-8s gap, parses per-match msXXX
    rows, resolves player identities (skip + report on failure), dedupes
    repeated physical keys before the Hawkeye fetch, then fetches Hawkeye stats
    sequentially, upserts each winner-first bronze row — reusing the stored
    physical match's id by the canonical (date, tournament, player-pair) key or
    deriving the deterministic Sackmann id from the strictly-parsed ``msNNN``
    sequence; with ``force`` an existing row is replaced by the candidate,
    otherwise it is skipped — and appends each new match's Sackmann-format row
    to the year's raw CSV (deduped against the file and this run; never a
    rewrite). A proposed id already owned by a different physical match
    (``_id_collision``) is a reported skip, never a merge. Every failure is a
    printed, counted skip — never an abort. Returns the per-tournament record
    the flow folds into its totals.
    """
    tournament_id = str(tournament.get("tournament_id") or "")
    name = tournament.get("name")
    result: dict[str, Any] = {
        "tournament_id": tournament_id,
        "discovered": 0,
        "fetched": 0,
        "inserted": 0,
        "updated": 0,
        "noop": 0,
        "skipped": 0,
        "csv_appended": 0,
        "results_page_ok": False,
    }
    year = int(tournament.get("year") or year)
    url = TOURNAMENT_RESULTS_URL.format(
        slug=tournament["slug"], tournament_id=tournament_id, year=year
    )
    print(f"Tournament {tournament_id} ({name}): fetching {url}")
    time.sleep(random.uniform(HAWKEYE_SLEEP_MIN_S, HAWKEYE_SLEEP_MAX_S))
    html, err = _fetch_page(page, url, f"results {tournament_id}")
    if err:
        print(f"  Tournament {tournament_id}: skipped ({err})")
        print(
            f"Tournament {tournament_id}: results_page_failed — "
            "discovered=0 fetched=0 inserted=0 updated=0 noop=0 skipped=0"
        )
        return result
    result["results_page_ok"] = True

    matches = extract_matches_from_results(html, tournament_id, year)
    result["discovered"] = len(matches)
    if not matches:
        print(f"  Tournament {tournament_id}: no msXXX matches parsed")
        print(
            f"Tournament {tournament_id}: discovered=0 fetched=0 inserted=0 updated=0 noop=0 skipped=0"
        )
        return result

    # The flow always passes the live maps; None callers (direct/fixture use)
    # fall back to the reference tables so discovery never crashes, refreshing
    # the local maps instead of the caller's.
    if canonical is None:
        canonical = canonical_players()
    if profiles is None:
        profiles = load_player_metadata()

    # All players appearing on the tournament page, deduplicated by id, so each
    # DB-missing player is discovered once on the run's shared page. Successful
    # discoveries refresh canonical/rank_map/profiles, making them resolvable
    # this run; failures become per-match unresolved reasons via resolve_player_id.
    page_candidates: dict[str, dict[str, Any]] = {}
    for match in matches:
        for key in ("player1", "player2"):
            pid = str(match.get(f"{key}_id") or "").upper()
            name = str(match.get(f"{key}_name") or "")
            slug = str(match.get(f"{key}_slug") or "")
            if pid and pid not in page_candidates:
                page_candidates[pid] = {"id": pid, "slug": slug, "player": name}
    discovered = rankings.discover_players(
        page,
        list(page_candidates.values()),
        canonical=canonical,
        rank_map=rank_map,
        profiles=profiles,
    )
    print(
        f"Tournament {tournament_id}: profile discovery "
        f"known={discovered['known']} discovered={discovered['discovered']} "
        f"failed={len(discovered['failed'])}"
    )
    if discovered["failed"]:
        for failed in discovered["failed"]:
            print(f"  Tournament {tournament_id}: profile discovery failed {failed}")

    resolved, unresolved = resolve_discovered_matches(matches, rank_map, canonical, tier)
    for skipped in unresolved:
        print(
            f"  Tournament {tournament_id} {skipped.get('match_id')}: skipped ({skipped['reason']})"
        )
    result["skipped"] += len(unresolved)

    resolved, duplicates = dedupe_physical_matches(resolved, tournament_id)
    for duplicate in duplicates:
        print(
            f"  Tournament {tournament_id} {duplicate.get('match_id')}: "
            "skipped (duplicate physical match)"
        )
    result["skipped"] += len(duplicates)

    hawkeye = fetch_hawkeye_batch(resolved, year=year, tournament_id=tournament_id, page=page)
    upsert_records: list[dict[str, Any]] = []
    for item in hawkeye:
        ms_id = str(item.get("match_id") or "")
        if item.get("hawkeye_error"):
            result["skipped"] += 1
            continue
        result["fetched"] += 1
        bronze_match_id, reason, key = _bronze_match_id(item, physical, year, tournament_id)
        if reason:
            print(f"  {ms_id}: skipped ({reason})")
            result["skipped"] += 1
            continue
        assert key is not None  # _bronze_match_id resolves a key whenever reason is ""
        reason = _id_collision(bronze_match_id, key, known_ids, claimed)
        if reason:
            print(f"  {ms_id}: skipped ({reason})")
            result["skipped"] += 1
            continue
        claimed[bronze_match_id] = key
        candidate = {**item, "match_id": bronze_match_id}
        row = hawkeye_to_bronze(
            item["payload"],
            candidate,
            surface=surface,
            rank_points=rank_points,
            ages=ages,
        )
        if row is None:
            # hawkeye_to_bronze printed the detailed skip reason.
            result["skipped"] += 1
            continue
        record = upsert_bronze_match(row, force=force)
        upsert_records.append(record)
        if record.get("action") != "skipped":
            # Same successful path writes both sinks: the CSV append is deduped
            # against the file (loaded once per year this run) and this run's
            # appends, so a rescrape never duplicates a row already on disk.
            csv_path = raw_match_path(year)
            ids = (csv_ids or {}).setdefault(year, load_csv_match_ids(csv_path))
            if bronze_match_id not in ids:
                appended, ids = append_raw_match_rows(
                    [bronze_row_to_raw_match(row, profiles)], csv_path, existing=ids
                )
                result["csv_appended"] += appended
                if csv_ids is not None:
                    csv_ids[year] = ids
        suffix = f" ({record['reason']})" if record.get("reason") else ""
        print(f"  {bronze_match_id}: {record['action']}{suffix}")
    summary = upsert_summary(upsert_records)
    result.update(
        {
            "inserted": summary["inserted"],
            "updated": summary["updated"],
            "noop": summary["noop"],
        }
    )
    result["skipped"] += summary["skipped"]
    print(
        f"Tournament {tournament_id}: discovered={result['discovered']} "
        f"fetched={result['fetched']} inserted={result['inserted']} "
        f"updated={result['updated']} noop={result['noop']} "
        f"skipped={result['skipped']} csv_appended={result['csv_appended']}"
    )
    return result


@flow(log_prints=True, retries=1)
def matches_flow(
    start_date: date | None = None,
    end_date: date | None = None,
    csv_ids: dict[int, set[str]] | None = None,
    force: bool = False,
):
    """Discover ATP 250+ tournaments in the window and enrich bronze rows.

    Window is inclusive on both ends: explicit dates are used exactly; with no
    dates the window is watermark-driven (last stored ``bronze.match_events``
    match_date through today), so a bare scheduled run never crawls unbounded
    history. Match ids are stable and idempotent: a canonical
    (match_date, tournament_id, unordered player pair) reuses the stored row's
    id, else a deterministic id derives from the ATP ``msNNN`` sequence.
    Existing rows are skipped unless ``force`` is set. Every page/match failure
    is a counted skip unless no requested page could be fetched/parsed, then
    the run fails so Prefect retries.
    """
    load_env()
    watermark = matches_watermark()
    window = resolve_window(start_date, end_date, watermark)
    if window is None:
        if start_date is not None and end_date is not None:
            print(f"start_date {start_date} is after end_date {end_date}: nothing to fetch.")
        elif watermark is None:
            print(
                "bronze.match_events is empty — no watermark to start from; "
                "run `just seed` first (or pass an explicit --start)."
            )
        else:
            print(
                f"Window ends before the bronze watermark {watermark.isoformat()}: nothing to fetch."
            )
        return
    start, end = window
    years = list(range(start.year, end.year + 1))
    print(
        f"Matches window {start.isoformat()} .. {end.isoformat()} (inclusive): "
        f"{len(years)} archive year(s) {years}"
    )

    rows = load_bronze_lookup_rows()
    physical = _physical_match_index(rows)
    known_ids = _known_match_ids(rows)
    claimed: dict[str, PhysicalKey] = {}
    rank_points, ages = _latest_rank_points_and_ages(rows)
    rank_map = load_ranking_player_map()
    canonical = canonical_players()
    profiles = load_player_metadata()

    totals: dict[str, int] = {
        "discovered": 0,
        "fetched": 0,
        "inserted": 0,
        "updated": 0,
        "noop": 0,
        "skipped": 0,
        "csv_appended": 0,
    }
    csv_ids = csv_ids or {}
    browser = rankings._launch_browser()
    page = None
    any_page_ok = False
    try:
        page = browser.new_page()
        print(
            "Browser session open: one page across archive years, tournaments, and Hawkeye fetches"
        )
        for year in years:
            url = RESULTS_ARCHIVE_URL.format(year=year)
            print(f"Archive {year}: fetching {url}")
            html, err = _fetch_page(page, url, f"results archive {year}")
            if err:
                print(f"  Archive {year}: skipped ({err})")
                continue
            tournaments = extract_tournaments_from_archive(html, year)
            if not tournaments:
                if year > date.today().year:
                    # A future archive legitimately lists nothing yet.
                    print(f"  Archive {year}: future year, no tournaments listed")
                    any_page_ok = True
                else:
                    print(
                        f"  Archive {year}: no tournaments parsed "
                        "(Cloudflare challenge or markup change?)"
                    )
                continue
            any_page_ok = True
            print(f"Archive {year}: {len(tournaments)} tournament(s) parsed")
            for tournament in in_window(tournaments, start, end):
                tournament_id = tournament.get("tournament_id")
                tier = tier_for_tournament(tournament, rows)
                if tier not in TIER_ELIGIBLE:
                    print(
                        f"  Skipped tournament {tournament_id} ({tournament.get('name')}): "
                        f"tier {tier!r} not in {sorted(TIER_ELIGIBLE)}"
                    )
                    continue
                surface = surface_for_tournament(str(tournament_id or ""), rows)
                result = _process_tournament(
                    page,
                    tournament,
                    year,
                    tier=tier,
                    surface=surface,
                    physical=physical,
                    known_ids=known_ids,
                    claimed=claimed,
                    rank_points=rank_points,
                    ages=ages,
                    rank_map=rank_map,
                    canonical=canonical,
                    profiles=profiles,
                    csv_ids=csv_ids,
                    force=force,
                )
                if result["results_page_ok"]:
                    any_page_ok = True
                for key in totals:
                    totals[key] += int(result.get(key, 0))
    finally:
        # CloakBrowser tracks sessions server-side; an exit without a clean
        # close permanently wedges the session id, so the browser (and any page
        # that was created) is always released here.
        print("Closing browser session")
        if page is not None:
            page.close()
        browser.close()

    if not any_page_ok:
        raise RuntimeError(
            f"Could not access or parse any ATP page for window "
            f"{start.isoformat()} .. {end.isoformat()} ({len(years)} archive year(s)) "
            "- see the skip lines above."
        )
    print(
        "Matches complete: "
        f"discovered={totals['discovered']} fetched={totals['fetched']} "
        f"inserted={totals['inserted']} updated={totals['updated']} "
        f"noop={totals['noop']} skipped={totals['skipped']}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        help="inclusive window start (YYYY-MM-DD); defaults to the bronze watermark",
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        help="inclusive window end (YYYY-MM-DD); defaults to today",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing bronze rows with the candidate row instead of skipping them",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    matches_flow(start_date=args.start, end_date=args.end, force=args.force)


if __name__ == "__main__":
    main()


def register_deployment() -> None:
    """Create/update the Monday-scheduled matches deployment (idempotent by name).

    Registers on the host ``tennis-pool`` work pool like the other host flows.
    Scheduled production runs use this independent deployment; rankings and
    matches are separate deployments — there is no combined scrape flow. No
    static parameter defaults: deployment
    parameters are frozen at registration, so a baked-in date would go stale
    for later cron runs. The flow defaults to the watermark (see
    ``matches_flow``); pass explicit ``--param start_date``/``--param end_date``
    to override for a manual backfill.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    # prefect's async_dispatch stubs from_source() as a Coroutine union, but in
    # a sync context it returns the Flow itself; cast to Any sidesteps that.
    deployment = cast(
        Any,
        matches_flow.from_source(
            source=str(repo_root),
            entrypoint="src/flows/matches.py:matches_flow",
        ),
    )
    deployment.deploy(
        name=MATCHES_DEPLOYMENT_NAME,
        work_pool_name=WORK_POOL_NAME,
        cron=MATCHES_CRON,
        build=False,
        ignore_warnings=True,
        print_next_steps=False,
    )
    print(f"Registered deployment {MATCHES_DEPLOYMENT_NAME!r} (cron {MATCHES_CRON})")
