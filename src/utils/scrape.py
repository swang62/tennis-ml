"""Shared ATP scraping helpers used by both the rankings and matches flows.

These cover player-profile discovery: navigating the run's single persistent
browser page to an ATP overview URL, parsing the identity fields, and persisting
newly discovered players through ``src.db.ingest.persist_atp_player``. They are
kept out of either flow module so both can import the same implementation.
"""

from __future__ import annotations

import random
import re
import time
from typing import Any, cast

import pandas as pd

from src.constants import BRONZE_PROFILES_TABLE
from src.db.client import execute_df
from src.db.ingest import persist_atp_player
from src.utils.countries import valid_ioc

# Per-navigation page-load budget for any ATP page goto (rankings week and player
# overview). Defined here so both flows share one value; rankings re-imports it.
PAGE_NAVIGATION_TIMEOUT_MS = 60_000

PLAYER_OVERVIEW_URL = "https://www.atptour.com/en/players/{slug}/{player_id}/overview"

# The overview page embeds the ATP player id in its own subnavigation/overview
# links as /en/players/<slug>/<ID>/overview; a mismatched or absent id means
# the page did not render the player we navigated to.
_OVERVIEW_ID_RE = re.compile(r"/en/players/[^/]+/([a-z0-9]+)/overview", re.I)

# Personal-details values (Age / Weight / Height / Turned pro / Plays / Country)
# render in the module-"...-details" block; labels are title-cased and values
# follow a colon.
_OVERVIEW_DOB_RE = re.compile(r"\bAge\s*\d{1,3}\s*\(\s*(\d{4})/(\d{2})/(\d{2})\s*\)")
_OVERVIEW_WEIGHT_RE = re.compile(r"\bWeight\b[^()]*?\((\d{2,3})\s*kg\)", re.I)
_OVERVIEW_HEIGHT_RE = re.compile(r"\bHeight\b[^()]*?\((\d{3})\s*cm\)", re.I)
_OVERVIEW_PRO_RE = re.compile(r"\bTurned\s+pro\b[^0-9]*(\d{4})", re.I)
_OVERVIEW_PLAYS_RE = re.compile(
    r"\bPlays?\b\s*([A-Za-z\-]+?)\s*,\s*([A-Za-z\-]+?)\s*Backhand",
    re.I,
)
# Three-letter IOC code drawn from the page's country flag sprite reference
# (<use href="...flags.svg#flag-<ioc>">), the strongest IOC signal on the page.
_FLAG_SRC_RE = re.compile(r"flags\.svg#flag-([a-z]{3})")


def _jitter() -> None:
    """Human-like random pause between navigation steps (bot-detection resistance)."""
    time.sleep(random.uniform(0.8, 2.5))


def _fetch_overview_html(page: Any, slug: str, player_id: str) -> tuple[str, str]:
    """Navigate the run's shared page to a player overview; (html, "") or ("", reason)."""
    url = PLAYER_OVERVIEW_URL.format(slug=slug, player_id=player_id)
    _jitter()
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=PAGE_NAVIGATION_TIMEOUT_MS)
    except Exception as exc:
        return "", f"navigation failed ({type(exc).__name__}: {exc})"
    try:
        body = page.content()
    except Exception as exc:
        return "", f"page content failed ({type(exc).__name__}: {exc})"
    if not body or not body.strip():
        return "", "empty page content"
    return body, ""


def parse_player_overview(
    html: str, profile_id: str, candidate: dict[str, Any]
) -> tuple[dict[str, str] | None, str]:
    """Parse a rendered ATP overview page into a validated profile candidate.

    Returns ``(profile_candidate, "")`` on success or ``(None, reason)`` on
    failure. The page's embedded profile id must equal ``profile_id`` (the id
    from the ATP profile link that brought us here), the player display name
    must be present, and an IOC must resolve from the page — otherwise the row
    is rejected with a structured reason and never persisted. Optional profile
    fields (birthdate, weight, height, turned pro, handedness, backhand) are
    extracted when the page has them and left empty otherwise.
    """
    raw = {
        "id": profile_id.upper(),
        "player": (candidate.get("player") and str(candidate["player"]).strip()) or "",
        "slug": (candidate.get("slug") and str(candidate["slug"]).strip()) or "",
        "birthdate": "",
        "weight": "",
        "height": "",
        "turnedpro": "",
        "hand": "",
        "backhand": "",
        "ioc": "",
    }
    if not raw["player"]:
        return None, "missing display name"

    # The page must render the very player we navigated to: an embedded profile
    # id that disagrees with the link id is a redirect/mismatch, not a discovery.
    ids = {m.group(1).upper() for m in _OVERVIEW_ID_RE.finditer(html)}
    if profile_id.upper() not in ids:
        return None, (f"page player id {sorted(ids)} does not match link id {profile_id.upper()}")

    dob = _OVERVIEW_DOB_RE.search(html)
    if dob:
        raw["birthdate"] = f"{dob.group(1)}{dob.group(2)}{dob.group(3)}"
    weight = _OVERVIEW_WEIGHT_RE.search(html)
    if weight:
        raw["weight"] = weight.group(1)
    height = _OVERVIEW_HEIGHT_RE.search(html)
    if height:
        raw["height"] = height.group(1)
    pro = _OVERVIEW_PRO_RE.search(html)
    if pro:
        raw["turnedpro"] = pro.group(1)
    plays = _OVERVIEW_PLAYS_RE.search(html)
    if plays:
        raw["hand"] = {
            "right-handed": "R",
            "left-handed": "L",
            "ambidextrous": "A",
        }.get(plays.group(1).lower(), plays.group(1).lower())
        raw["backhand"] = {
            "two-handed": "2H",
            "one-handed": "1H",
        }.get(plays.group(2).lower(), plays.group(2).lower())

    ioc = _ioc_from_overview(html)
    raw["ioc"] = ioc
    return raw, ""


def _ioc_from_overview(html: str) -> str:
    """Three-letter IOC from the page's country flag sprite, else UNK.

    The personal-details Country value is a full nationality name (e.g. "Great
    Britain"), not an IOC code, so it cannot map to a code reliably; the flag
    sprite reference (``flags.svg#flag-<ioc>``) is the page's only machine-
    readable IOC signal. Unverifiable identifiers resolve to UNK.
    """
    flag = _FLAG_SRC_RE.search(html)
    return valid_ioc(flag.group(1)) if flag else "UNK"


def _known_profile_ids() -> set[str]:
    """Every player id already present in bronze.player_profiles (uppercased)."""
    df = execute_df(f"SELECT player_id FROM {BRONZE_PROFILES_TABLE}", [])
    return {str(pid).upper() for pid in df.get("player_id", pd.Series(dtype=object))}


def _discover_player(
    page: Any, candidate: dict[str, Any], profile_id: str
) -> tuple[dict[str, str] | None, str]:
    """Fetch, parse, and persist one DB-missing player; ``(profile, "")`` or ``(None, reason)``.

    Raises on navigation/parse/persistence failures; the caller turns every
    exception into one structured failure. Never aborts the scrape.
    """
    html, err = _fetch_overview_html(page, str(candidate.get("slug") or ""), profile_id)
    if err:
        raise DiscoverError(err)
    parsed, err = parse_player_overview(html, profile_id, candidate)
    if err:
        raise DiscoverError(err)
    if parsed is None:
        raise DiscoverError("unparseable overview page")
    persist_atp_player(cast(dict[str, object], parsed), insert=True)
    return parsed, ""


def discover_players(
    page: Any,
    candidates: list[dict[str, Any]],
    *,
    canonical: dict[str, str],
    rank_map: dict[str, str],
    profiles: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Discover and persist ATP profiles for player ids missing from bronze.

    ``candidates`` is a per-run batch of unique ``{id, slug, player}`` rows
    gathered from rankings/matches pages. Existing bronze players are skipped
    (never navigated, never written); each DB-missing id is fetched once on the
    run's shared ``page``, validated (page id == link id, name and IOC present),
    and persisted through ``ingest.persist_atp_player``. Successful discoveries
    immediately refresh the caller's in-memory ``canonical``, ``rank_map``, and
    ``profiles`` so downstream resolution sees them this run. Failures are
    per-player and non-fatal: navigation, parsing, identity, and persistence
    errors become structured reasons that never abort the scrape.

    Returns ``{known, discovered, failed: [{id, player, reason}]}``.
    """
    known = _known_profile_ids()
    seen: set[str] = set()
    missing: list[dict[str, Any]] = []
    for candidate in candidates:
        player_id = str(candidate.get("id") or "").upper()
        if player_id and player_id in known:
            continue
        if player_id and player_id in seen:
            continue
        if player_id:
            seen.add(player_id)
        missing.append(candidate)
    result: dict[str, Any] = {
        "known": len(candidates) - len(missing),
        "discovered": 0,
        "failed": [],
    }
    for candidate in missing:
        player_id = str(candidate.get("id") or "").upper()
        name = str(candidate.get("player") or "")
        if not player_id or not name:
            result["failed"].append(
                {
                    "id": player_id or name or "?",
                    "player": name,
                    "reason": "missing id or display name",
                }
            )
            continue
        parsed: dict[str, str] | None = None
        try:
            parsed, _err = _discover_player(page, candidate, player_id)
        except Exception as exc:
            result["failed"].append(
                {"id": player_id, "player": name, "reason": f"{type(exc).__name__}: {exc}"}
            )
            continue
        assert parsed is not None  # _discover_player raises instead of returning None
        # Success: make the new identity resolvable for the rest of this run.
        canonical[player_id] = name
        rank_map[player_id] = player_id
        profiles.setdefault(player_id, {}).update(
            {
                "display_name": name,
                "hand": parsed.get("hand", ""),
                "height": parsed.get("height", ""),
                "ioc": parsed.get("ioc", ""),
            }
        )
        result["discovered"] += 1
    return result


class DiscoverError(RuntimeError):
    """Structured per-player discovery failure (non-fatal to the scrape)."""
