"""Shared ATP player-profile scraping helpers."""

from __future__ import annotations

import random
import re
import time
from html import unescape
from typing import Any, cast

import pandas as pd

from src.constants import BRONZE_PROFILES_TABLE
from src.db.client import execute_df
from src.db.ingest import persist_atp_player
from src.utils.countries import valid_ioc

# Shared page-load budget for rankings and player-overview navigation.
PAGE_NAVIGATION_TIMEOUT_MS = 60_000

PLAYER_OVERVIEW_URL = "https://www.atptour.com/en/players/{slug}/{player_id}/overview"

# Validate the embedded overview link to reject redirects and mismatches.
_OVERVIEW_ID_RE = re.compile(r"/en/players/[^/]+/([a-z0-9]+)/overview", re.I)
_OVERVIEW_TITLE_RE = re.compile(r"<title>\s*(.*?)\s*\|\s*Overview\b", re.I | re.S)
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Personal details render in the module-"...-details" block.
_OVERVIEW_DOB_RE = re.compile(r"\bAge\s*\d{1,3}\s*\(\s*(\d{4})/(\d{2})/(\d{2})\s*\)")
_OVERVIEW_WEIGHT_RE = re.compile(r"\bWeight\b[^()]*?\((\d{2,3})\s*kg\)", re.I)
_OVERVIEW_HEIGHT_RE = re.compile(r"\bHeight\b[^()]*?\((\d{3})\s*cm\)", re.I)
_OVERVIEW_PRO_RE = re.compile(r"\bTurned\s+pro\s*(\d{4})\b", re.I)
_OVERVIEW_PLAYS_RE = re.compile(
    r"\bPlays?\b\s*([A-Za-z\-]+?)\s*,\s*([A-Za-z\-]+?)\s*Backhand",
    re.I,
)
_OVERVIEW_BIRTHPLACE_RE = re.compile(
    r"\bBirthplace\s+(.+?)(?=\s+(?:Plays|Coach|Latest\s+news|Follow\s+player)\b)", re.I
)
# The flag sprite is the page's machine-readable IOC signal.
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
        page.wait_for_function(
            """playerId => {
                const id = `/${playerId}/overview`.toLowerCase();
                const hasPlayer = [...document.querySelectorAll("a[href]")]
                    .some(link => link.href.toLowerCase().includes(id));
                return hasPlayer && /\\bAge\\b/.test(document.body?.innerText || "");
            }""",
            arg=player_id,
            timeout=PAGE_NAVIGATION_TIMEOUT_MS,
        )
    except Exception as exc:
        return "", f"overview did not render ({type(exc).__name__}: {exc})"
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
    """Parse and validate an ATP overview into a profile candidate."""
    title = _OVERVIEW_TITLE_RE.search(html)
    display_name = unescape(_HTML_TAG_RE.sub(" ", title.group(1))).strip() if title else ""
    raw = {
        "id": profile_id.upper(),
        "player": display_name
        or ((candidate.get("player") and str(candidate["player"]).strip()) or ""),
        "atpname": display_name,
        "slug": (candidate.get("slug") and str(candidate["slug"]).strip()) or "",
        "birthdate": "",
        "weight": "",
        "height": "",
        "turnedpro": "",
        "hand": "",
        "backhand": "",
        "birthplace": "",
        "coaches": "",
        "ioc": "",
    }
    if not raw["player"]:
        return None, "missing display name"

    # Reject redirects and pages for a different player.
    ids = {m.group(1).upper() for m in _OVERVIEW_ID_RE.finditer(html)}
    if profile_id.upper() not in ids:
        return None, (f"page player id {sorted(ids)} does not match link id {profile_id.upper()}")

    text = re.sub(r"\s+", " ", unescape(_HTML_TAG_RE.sub(" ", html)))
    dob = _OVERVIEW_DOB_RE.search(text)
    if dob:
        raw["birthdate"] = f"{dob.group(1)}{dob.group(2)}{dob.group(3)}"
    weight = _OVERVIEW_WEIGHT_RE.search(text)
    if weight:
        raw["weight"] = weight.group(1)
    height = _OVERVIEW_HEIGHT_RE.search(text)
    if height:
        raw["height"] = height.group(1)
    pro = _OVERVIEW_PRO_RE.search(text)
    if pro:
        raw["turnedpro"] = pro.group(1)
    plays = _OVERVIEW_PLAYS_RE.search(text)
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

    birthplace = _OVERVIEW_BIRTHPLACE_RE.search(text)
    if birthplace:
        raw["birthplace"] = birthplace.group(1).strip()

    ioc = _ioc_from_overview(html)
    raw["ioc"] = ioc
    return raw, ""


def _ioc_from_overview(html: str) -> str:
    """Return the IOC from the flag sprite, or ``UNK`` when absent."""
    flag = _FLAG_SRC_RE.search(html)
    return valid_ioc(flag.group(1)) if flag else "UNK"


def _known_profile_ids() -> set[str]:
    """Every player id already present in bronze.player_profiles (uppercased)."""
    df = execute_df(f"SELECT player_id FROM {BRONZE_PROFILES_TABLE}", [])
    return {str(pid).upper() for pid in df.get("player_id", pd.Series(dtype=object))}


def _discover_player(
    page: Any, candidate: dict[str, Any], profile_id: str
) -> tuple[dict[str, str] | None, str]:
    """Fetch, validate, and persist one missing player profile."""
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
    profiles: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Discover missing profiles and return known, discovered, and failed counts."""
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
