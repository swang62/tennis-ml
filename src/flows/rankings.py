"""Prefect flow: weekly ATP rankings catch-up.

The database is the self-healing watermark (``MAX(bronze.rankings.ranking_date)``).
With no params it fetches every missing Monday from the watermark through the
most recent completed Monday; explicit ``--param start_date``/``end_date``
override the range. Guarantees for every path: a stored ranking week is never
re-scraped, the effective start never precedes the Monday after the watermark
nor Jan 1 of the current year, and the effective end never follows the most
recent completed Monday.

Weeks are fetched with the CloakBrowser Python library (never the MCP server),
using one persistent browser/page for the whole run so the profile's Cloudflare
clearance and humanized-jitter discipline carry across every week. A process
that exits without a clean ``browser.close()`` wedges that session server-side,
so the ``finally`` always closes page and browser. Identity resolves only
through the approved ranking identity map (``load_ranking_player_map``): raw
ranking source ids never reach the database and unmapped players are skipped
and reported.

A week that fails to load or parse is skipped so the backfill continues; but a
run that could not access or parse the site for any of its weeks fails (and is
retried) instead of succeeding on no data. Each successful week commits
independently.

Rankings and matches are separate deployments (``rankings-flow/rankings``
Monday 06:00 UTC, ``matches-flow/matches`` 06:30 UTC); a successful run of
either triggers the incremental ETL deployment. There is no combined scrape
flow; both modules are standalone commands (``just rankings`` / ``just matches``).
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import time
from contextlib import suppress
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd
from prefect import flow, task

from src.constants import WORK_POOL_NAME, load_env
from src.db.client import connection
from src.db.ingest import (
    BRONZE_RANKINGS_TABLE,
    RANKING_TARGET_COLUMNS,
    RANKINGS_COLUMNS,
    RANKINGS_DIR,
    _copy_df_into,
    canonical_players,
    load_player_metadata,
)
from src.utils.scrape import (
    PAGE_NAVIGATION_TIMEOUT_MS,
    PLAYER_OVERVIEW_URL,
    DiscoverError,
    _discover_player,
    _fetch_overview_html,
    _ioc_from_overview,
    _jitter,
    _known_profile_ids,
    discover_players,
    parse_player_overview,
)

RANKINGS_URL = "https://www.atptour.com/en/rankings/singles?rankRange=0-200&dateWeek={date}"
CURRENT_RANKINGS_CSV = RANKINGS_DIR / "atp_rankings_current.csv"

# Rankings table anchor, stable across ATP Tour page versions (both the mobile
# and the desktop rankings table carry this class; the parser uses only the
# first table containing player links). A Cloudflare challenge or markup change
# means this never matches: the week is logged and skipped, never a failure.
# Rows are targeted (not the table element) because the table is permanently
# hidden by the .non-live state — the element exists in the DOM immediately but
# the row links only appear once the server-side data has rendered.
RANKINGS_TABLE_SELECTOR = "table.mega-table tbody tr a[href*='/en/players/']"
# #dateWeek-filter: the SELECT listing every published ranking week. Its
# presence in the DOM marks the end of the Cloudflare/manual widget hand-off;
# options are matched by text in _week_in_filter.
FILTER_SELECTOR = "#dateWeek-filter"
# Per-week page render budget. Reapplied fresh on every wait (one wait per
# week), so a slow page never eats into the next week's budget. 30s per page is
# ample for a normal rankings render.
RANKINGS_TABLE_TIMEOUT_MS = 30_000
# Total budget for the rows to render (Cloudflare auto-clear or server-side
# render) before a week is skipped. 30s is the outer bound — anything slower
# than that is not going to resolve, and a genuine missing week has no rows to
# render and is skipped immediately.
CHALLENGE_RESOLVE_BUDGET_S = 30
# Readiness budget for the #dateWeek-filter element after navigation: while a
# Cloudflare/manual widget is up the filter is absent from the DOM, so element
# presence is the readiness signal. Once it appears, the requested week's
# option is checked exactly once — a missing option means the week was never
# published, with no further waiting.
FILTER_VERIFY_BUDGET_S = 15

CLOAKBROWSER_PROFILE_DIR = (
    Path.home() / ".local" / "share" / "tennis-prefect-worker" / "cloakbrowser"
)

# Player profile link inside each rankings row: /en/players/<slug>/<id>/overview
# where <id> is the canonical ATP_Database id (lowercased in the URL). The
# rank and points cells are matched by class prefix ("rank", "points") across
# the current mobile/desktop table variants; the empty <li class="rank"> move
# indicator inside the player cell never matches the rank regex because it
# carries no bare digits.
# The slug group is captured so each parsed row can feed profile discovery.
_SLUG_RE = r"[^/]+"
_PLAYER_LINK_RE = re.compile(rf"/en/players/({_SLUG_RE})/([^/]+)/overview")
_RANK_CELL_RE = re.compile(r'class="rank\b[^"]*"[^>]*>\s*(\d+)\s*<', re.S)
_POINTS_CELL_RE = re.compile(r'class="points\b[^"]*"[^>]*>(.*?)</td>', re.S)
# First table in the page whose rows carry player links is the rankings table.
_TABLE_SPLIT_RE = re.compile(r"<table\b", re.I)
_ROW_SPLIT_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.S)
_TEXT_TAG_RE = re.compile(r"<[^>]+>")


class RankingsParseError(ValueError):
    """Raised when a fetched rankings page has no parseable rankings table."""


# ── Date math ────────────────────────────────────────────────────


def latest_completed_monday(today: date) -> date:
    """Most recent Monday strictly before ``today``."""
    days_since_monday = today.weekday()
    if days_since_monday == 0:
        return today - timedelta(days=7)
    return today - timedelta(days=days_since_monday)


def ranking_mondays_after(watermark: date, as_of: date) -> list[date]:
    """Every ATP ranking Monday in (watermark, as_of] when as_of is Monday.

    The watermark is itself a ranking Monday, so the first candidate is
    ``watermark + 7 days`` and every further Monday is a 7-day step.
    """
    end = as_of if as_of.weekday() == 0 else latest_completed_monday(as_of)
    start = watermark + timedelta(days=7)
    if start > end:
        return []
    return [start + timedelta(days=7 * i) for i in range((end - start).days // 7 + 1)]


def current_watermark() -> date | None:
    """Latest stored ranking date in bronze.rankings, or None when empty."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT MAX(ranking_date) FROM {BRONZE_RANKINGS_TABLE}")
        row = cur.fetchone()
    return row[0] if row is not None and row[0] is not None else None


def stored_ranking_mondays() -> set[date]:
    """Every ranking Monday currently present in bronze.rankings."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT DISTINCT ranking_date FROM {BRONZE_RANKINGS_TABLE}")
        return {row[0] for row in cur.fetchall()}


# ── Date math ────────────────────────────────────────────────────


@task
def missing_ranking_mondays(
    start_date: date | None = None,
    end_date: date | None = None,
    force: bool = False,
) -> tuple[date | None, list[date]]:
    """Weeks to fetch, optionally ignoring stored ranking dates.

    Returns ``(watermark, weeks)``; ``watermark`` is None when the table is
    empty and ``weeks`` is then empty because the flow must not scrape history
    from scratch (run ``just seed`` first).

    Presence is the only completeness gate unless ``force`` is true. Forced
    runs schedule every Monday in the requested range and upsert the rows again.
    With no ``start_date`` the scan starts one week after the watermark (the
    max stored Monday); with an explicit ``start_date`` it starts there (snapped
    forward to the next Monday). The upper bound is ``end_date`` when given,
    else the most recent completed Monday. Weeks come back oldest first.

    Safety floors apply to every path (bare, start-only, end-only, both): the
    effective start never precedes the Monday after the watermark nor Jan 1 of
    the current year, and the effective end never follows the most recent
    completed Monday — so an explicit historical ``start_date`` can never
    backfill before this year and stored weeks are never touched.
    """
    today = date.today()
    last_completed = latest_completed_monday(today)
    end = min(end_date or last_completed, last_completed)
    if force:
        start = start_date or date(today.year, 1, 1)
        start += timedelta(days=(7 - start.weekday()) % 7)
        weeks = []
        monday = start
        while monday <= end:
            weeks.append(monday)
            monday += timedelta(days=7)
        return None, weeks

    stored = stored_ranking_mondays()
    if not stored:
        return None, []
    watermark = max(stored)
    if start_date is None:
        start = watermark + timedelta(days=7)
    else:
        start = start_date + timedelta(days=(7 - start_date.weekday()) % 7)
    # Safety floors: never earlier than the Monday after the watermark, never
    # earlier than Jan 1 of the current year, never later than the most recent
    # completed Monday. The start is snapped to a Monday to keep the weekly step.
    floor = max(watermark + timedelta(days=7), date(today.year, 1, 1))
    start = max(start, floor)
    start += timedelta(days=(7 - start.weekday()) % 7)
    weeks: list[date] = []
    monday = start
    while monday <= end:
        if monday not in stored:
            weeks.append(monday)
        monday += timedelta(days=7)
    return watermark, weeks


# ── HTML parsing (fixture-testable, no network) ──────────────────


def _cell_text(cell_html: str) -> str:
    """Strip tags/entities from a cell fragment; commas are thousands separators."""
    return _TEXT_TAG_RE.sub("", cell_html).replace("&nbsp;", " ").strip()


def _parse_row(row_html: str) -> dict[str, Any] | None:
    """One rankings row -> {rank, points, name, player_id, slug}; None when not a player row."""
    link = _PLAYER_LINK_RE.search(row_html)
    if link is None:
        return None
    slug = link.group(1)
    player_id = link.group(2).upper()
    anchor_start = row_html.rfind("<a", 0, link.start())
    text_start = row_html.find(">", anchor_start) + 1
    name = _cell_text(row_html[text_start : row_html.find("</a>", text_start)])
    rank_match = _RANK_CELL_RE.search(row_html)
    if rank_match is None:
        raise RankingsParseError(f"rankings row missing rank cell: {name!r}")
    rank = int(rank_match.group(1))
    points_match = _POINTS_CELL_RE.search(row_html)
    points_text = _cell_text(points_match.group(1)) if points_match else ""
    points = int(points_text.replace(",", "")) if points_text else None
    return {
        "rank": rank,
        "points": points,
        "name": name,
        "player_id": player_id,
        "slug": slug,
    }


def extract_rankings_from_html(html: str) -> list[dict[str, Any]]:
    """Parse the rendered ATP rankings page into rank/points/name/player_id rows.

    Uses only the first page table that contains player profile links (the
    current page renders a concise and a full table with identical data).
    Raises RankingsParseError when no table with player links is found, so a
    challenge page or a markup change fails visibly.
    """
    fragments = _TABLE_SPLIT_RE.split(html)
    for fragment in fragments[1:]:  # skip everything before the first <table>
        rows: list[dict[str, Any]] = []
        for row_html in _ROW_SPLIT_RE.findall(fragment):
            parsed = _parse_row(row_html)
            # Only the top 200 are requested (rankRange=0-200); ATP may render a
            # fuller table, and the page duplicates the table, so cap by rank.
            if parsed is not None and 1 <= parsed["rank"] <= 200:
                rows.append(parsed)
        if rows:
            return rows
    raise RankingsParseError("no rankings table with player links found in page HTML")


# ── Identity translation + upsert ────────────────────────────────


def translate_rank_rows(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Convert every parsed live ATP ranking row to a bronze row."""
    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        if row["player_id"] and 1 <= row["rank"] <= 200:
            kept.append(row)
        else:
            skipped.append(row)
    if not kept:
        return pd.DataFrame(columns=["player_id", "rank", "points"]), skipped
    frame = pd.DataFrame(kept)[["player_id", "rank", "points"]]
    frame["player_id"] = frame["player_id"].astype(str)
    return cast(pd.DataFrame, frame), skipped


def tier_from_bronze(
    tournament_id: str, existing_rows: list[dict[str, Any]] | None = None
) -> str | None:
    """Bronze tier for a tournament from existing match rows; None when unknown.

    Bronze has no tournament_id column, so a row belongs to the tournament when
    its ``match_id`` carries the id as a ``-``-separated segment (both the
    Sackmann ``YYYY-TOURNAMENT_ID-NNN`` and the match-stats
    ``YYYY-YYYY-TOURNAMENT_ID-NNN`` shapes embed it). Returns the ``tournament``
    value (already bronze vocabulary, set at ingest) of the newest matching row.
    Unknown tournaments resolve to None — the caller skips/reports instead of
    inventing a tier. Purely over the supplied rows; nothing is scraped here.
    """
    token = f"-{str(tournament_id).strip()}-"
    if not existing_rows or token == "--":
        return None
    known = [
        row
        for row in existing_rows
        if token in str(row.get("match_id", "")) and row.get("tournament")
    ]
    if not known:
        return None
    newest = max(known, key=lambda row: str(row.get("match_date", "")))
    return str(newest["tournament"])


def _launch_browser():
    """Open the shared persistent CloakBrowser profile for this scrape run.

    A persistent profile retains Cloudflare clearance cookies between runs. The
    first launch disables HTTP/2 as CloakBrowser recommends to warm the profile.
    The flow is responsible for closing the context in its finally block.
    """
    import os

    import cloakbrowser

    first_launch = not CLOAKBROWSER_PROFILE_DIR.exists()
    print(f"Starting persistent CloakBrowser profile at {CLOAKBROWSER_PROFILE_DIR}")
    license_key = os.getenv("CLOAKBROWSER_LICENSE_KEY")
    if license_key:
        print("Using CloakBrowser Pro license key")
    else:
        print("No CLOAKBROWSER_LICENSE_KEY set; falling back to the free binary")
    return cloakbrowser.launch_persistent_context(
        CLOAKBROWSER_PROFILE_DIR,
        headless=False,
        humanize=True,
        human_preset="careful",
        geoip=True,
        license_key=license_key,
        args=(["--disable-http2", "--start-minimized"] if first_launch else ["--start-minimized"]),
    )


def _week_in_filter(page, week: date) -> bool:
    """Whether ``week`` appears in the page's #dateWeek-filter.

    The filter lists every published ranking week; the requested week shows up
    as an option when it exists. Each option's ``value`` is the date in
    YYYY-MM-DD, except the most recent week, whose value is the literal
    "Current Week" — so the current week matches by value only when it is the
    latest completed Monday. A week that was never published (e.g. a future
    Monday) is absent from the filter entirely and returns False.
    """
    wanted = week.strftime("%Y-%m-%d")
    # The latest completed Monday bounds the request window, so a week beyond it
    # can never be a valid (published) "Current Week".
    latest = latest_completed_monday(date.today()).strftime("%Y-%m-%d")
    return page.evaluate(
        "([wanted, latest]) => {"
        # Normalize by dropping separators so YYYY-MM-DD and YYYY.MM.DD match,
        # and check both the option value and its visible text. The current
        # week's value is the literal "Current Week" but its text carries the
        # date, so the text normalization already covers it; the explicit
        # Current Week + latest clause is a safety net for a date-less label.
        "  const w = wanted.replace(/[^0-9]/g, '');"
        "  const latestN = latest.replace(/[^0-9]/g, '');"
        "  return Array.from("
        "    document.querySelectorAll('#dateWeek-filter option')"
        "  ).some(o => {"
        "    const fields = [o.value, o.textContent.trim()];"
        "    const norms = fields.map(f => f.replace(/[^0-9]/g, '')).filter(Boolean);"
        "    if (norms.includes(w)) return true;"
        "    if (fields.includes('Current Week') && w === latestN) return true;"
        "    return false;"
        "  });"
        "}",
        [wanted, latest],
    )


def _fetch_week_html(page, url: str, week: date) -> str:
    """Navigate the shared rankings page to one week and return its HTML.

    The page stays open for the entire flow so its navigation state and the
    persistent profile's Cloudflare clearance are reused for every week. Random
    jitter before/after navigation keeps the moves human-like.

    Verification is element-first, not option-first: the page may be sitting on
    a Cloudflare or manual widget right after navigation, so the #dateWeek-filter
    SELECT is absent while verification is still underway. The flow first waits
    up to FILTER_VERIFY_BUDGET_S for the filter element itself to appear (a
    page that renders it immediately proceeds at once), then checks the
    requested week's option exactly once. A missing option means the week was
    never published — rejected immediately. A filter that never appears within
    the budget is an unverifiable page, distinct from a missing week.

    The gate for real weeks is the captured HTML containing actual player links:
    the table element itself is hidden by the .non-live state, so "attached" on
    the table fires before the rows render, and "visible" never fires at all.
    The loop therefore waits until either the row links appear in
    ``page.content()`` or the CHALLENGE_RESOLVE_BUDGET_S deadline passes (a
    Cloudflare challenge auto-clears in between). The caller decides whether a
    final no-table page is a missing week.
    """
    _jitter()
    page.goto(url, wait_until="domcontentloaded", timeout=PAGE_NAVIGATION_TIMEOUT_MS)
    _jitter()
    try:
        page.wait_for_selector(
            FILTER_SELECTOR, state="attached", timeout=FILTER_VERIFY_BUDGET_S * 1000
        )
    except Exception as exc:
        raise RankingsParseError(
            f"week {week.isoformat()}: #dateWeek-filter unavailable "
            "(Cloudflare or widget verification failed)"
        ) from exc
    if not _week_in_filter(page, week):
        raise RankingsParseError(
            f"week {week.isoformat()} not present in #dateWeek-filter (never published)"
        )
    deadline = time.monotonic() + CHALLENGE_RESOLVE_BUDGET_S
    while True:
        # Reapplied fresh on every check, so a slow page never eats into the
        # next week's budget.
        with suppress(Exception):
            page.wait_for_selector(
                RANKINGS_TABLE_SELECTOR, state="attached", timeout=RANKINGS_TABLE_TIMEOUT_MS
            )
        html = page.content()
        if _PLAYER_LINK_RE.search(html) or time.monotonic() >= deadline:
            break
        print("Rankings rows not rendered yet (Cloudflare or slow server-side render); waiting...")
        _jitter()
    return html


def fetch_and_upsert_week(
    page,
    week: date,
    *,
    canonical: dict[str, str] | None = None,
    profiles: dict[str, dict[str, str]] | None = None,
) -> int | None:
    """Fetch one weekly ranking page and upsert its mapped rows.

    Uses the run's shared browser page (never launches its own — the flow owns
    the page and closes it in finally). Before the identity-map filter it
    discovers ATP profiles for ranking players missing from bronze, so a valid
    new player is ingestible this week; a discovery failure only skips that
    player's row (via the identity filter, since it never entered the maps).
    ``canonical``/``profiles`` default to the reference tables, so direct calls
    without them (e.g. fixtures) still discovery against bronze. Commits
    independently (``_copy_df_into`` runs in one transaction). Returns the
    number of rows written, or ``None`` when the page could not be accessed or
    parsed (Cloudflare challenge, missing rankings week, markup change) — a
    signal the caller reads to distinguish "found no data" (a failure) from
    "found data but nothing new to write" (``0``).
    """
    if canonical is None:
        canonical = canonical_players()
    if profiles is None:
        profiles = load_player_metadata()
    url = RANKINGS_URL.format(date=week.isoformat())
    print(f"Week {week.isoformat()}: fetching {url}")
    try:
        html = _fetch_week_html(page, url, week)
        rows = extract_rankings_from_html(html)
    except Exception as exc:
        print(f"Week {week.isoformat()}: skipped (could not load or parse): {exc}")
        return None

    frame, skipped = translate_rank_rows(rows)
    frame = (
        frame.assign(ranking_date=week)[RANKING_TARGET_COLUMNS]
        .drop_duplicates(subset=["ranking_date", "player_id"], keep="last")
        .reset_index(drop=True)
    )

    if not frame.empty:
        _copy_df_into(
            BRONZE_RANKINGS_TABLE,
            frame,
            conflict_col="ranking_date, player_id",
            update_cols=["rank", "points"],
        )
        _append_current_rankings(frame)
    candidates = [
        {"id": row["player_id"], "slug": row["slug"], "player": row["name"]} for row in rows
    ]
    discovered = discover_players(page, candidates, canonical=canonical, profiles=profiles)
    print(
        f"Week {week.isoformat()}: profile discovery "
        f"known={discovered['known']} discovered={discovered['discovered']} "
        f"failed={len(discovered['failed'])}"
    )
    for failed in discovered["failed"]:
        print(f"  profile discovery failed: {failed}")
    for s in skipped:
        print(
            f"  skipped invalid ranking row: player_id={s['player_id']} "
            f"name={s['name']!r} rank={s['rank']} points={s['points']}"
        )
    print(f"Week {week.isoformat()}: succeeded: {len(frame)} rows stored, {len(skipped)} skipped")
    return len(frame)


def _append_current_rankings(frame: pd.DataFrame) -> None:
    """Append unseen live ATP-id ranking rows to the current raw CSV."""
    with open(CURRENT_RANKINGS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != RANKINGS_COLUMNS:
            raise ValueError(f"{CURRENT_RANKINGS_CSV.name}: expected columns {RANKINGS_COLUMNS}")
        existing = {(row["ranking_date"], row["player"]) for row in reader}
    with open(CURRENT_RANKINGS_CSV, "a", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        for row in frame.to_dict(orient="records"):
            key = (pd.Timestamp(row["ranking_date"]).strftime("%Y%m%d"), str(row["player_id"]))
            if key not in existing:
                writer.writerow(
                    [
                        key[0],
                        int(row["rank"]),
                        key[1],
                        "" if pd.isna(row["points"]) else int(row["points"]),
                    ]
                )


@flow(log_prints=True, retries=1)
def rankings_flow(
    start_date: date | None = None,
    end_date: date | None = None,
    force: bool = False,
):
    """Scrape missing ranking weeks: watermark-to-today (no params) or a date range.

    With no params every missing Monday from the watermark through the most
    recent completed Monday is fetched — the same for scheduled cron runs,
    ``prefect deployment run`` without params, and direct local calls, so a
    bare run never silently backfills far history. With only ``end_date`` every
    missing Monday from the watermark through that date is fetched oldest
    first; any explicit ``start_date`` and/or ``end_date`` — both together
    fetch every Monday in that range regardless of the watermark (for
    historical backfills). Safety guarantees, enforced for every path: a week
    already present in ``bronze.rankings`` is never re-scraped (presence is
    the only completeness test — partial prior ingests are not refetched); the
    effective start never precedes the Monday after the stored watermark or
    Jan 1 of the current year (an explicit historical ``start_date`` is
    clamped, with a warning); and the effective end never follows the most
    recent completed Monday. An ``end_date`` at or before the watermark
    naturally fetches nothing. Launches exactly one CloakBrowser session, processes every missing
    week inside a try block, and closes the browser in finally — even when a
    week fails, the session is always released before the flow returns/raises.
    A run that had weeks to fetch but could not access or parse the rankings
    site for any of them (Cloudflare blocked every page, or the markup changed)
    is marked failed and retried; finding data that is already stored is a
    success, not a failure.
    """
    load_env()
    today = date.today()
    if start_date is not None and start_date < date(today.year, 1, 1):
        print(
            f"WARNING: start_date {start_date.isoformat()} is before Jan 1 {today.year}; "
            "the scan is clamped to this year."
        )
    watermark, weeks = missing_ranking_mondays(start_date, end_date, force)
    if watermark is None and not force:
        print(
            "bronze.rankings is empty — initial seed not complete; "
            "run `just seed` first. Skipping browser work."
        )
        return
    if not weeks:
        if force:
            print("No ranking weeks in the forced range.")
        else:
            assert watermark is not None
            print(f"Rankings are current through {watermark.isoformat()} — no missing weeks.")
        return
    if force:
        print(
            f"Force enabled: fetching {len(weeks)} ranking week(s), including stored weeks, "
            f"oldest first: {weeks[0].isoformat()} .. {weeks[-1].isoformat()}"
        )
    else:
        assert watermark is not None
        print(
            f"Watermark {watermark.isoformat()}: fetching {len(weeks)} missing week(s), "
            f"oldest first: {weeks[0].isoformat()} .. {weeks[-1].isoformat()}"
        )

    canonical = canonical_players()
    profiles = load_player_metadata()
    browser = _launch_browser()
    page = None
    stored = 0
    found_data = False
    try:
        page = browser.new_page()
        print("Browser session open: navigating one page across all weeks")
        for week in weeks:
            rows = fetch_and_upsert_week(
                page,
                week,
                canonical=canonical,
                profiles=profiles,
            )
            if rows is not None:
                found_data = True
                stored += rows
    finally:
        # CloakBrowser tracks sessions server-side; an exit without a clean
        # close permanently wedges the session id, so the browser (and any page
        # that was created) is always released here — even when a week or page
        # creation fails. Closing the context closes its pages too.
        print("Closing browser session")
        if page is not None:
            page.close()
        browser.close()
    _fail_if_no_data_found(found_data, weeks)
    print(f"Scrape complete: {stored} rows stored")


def _fail_if_no_data_found(found_data: bool, weeks: list[date]) -> None:
    """Fail the run when the scrape could not access or parse the site at all.

    Rankings post weekly, so if every week we tried failed to fetch or parse
    (Cloudflare block, missing page, or markup change) the site is effectively
    unreachable — a real failure worth surfacing, not a "nothing new" result.
    That legitimate case (an empty ``weeks`` because everything is current)
    returns before any browser work. Finding a parseable page counts as success
    even when it wrote no new rows (already-present data), so re-scraping stored
    weeks is never a failure.
    """
    if not found_data:
        raise RuntimeError(
            f"Scrape could not access or parse the rankings site for any of "
            f"{len(weeks)} expected week(s) "
            f"({weeks[0].isoformat()}..{weeks[-1].isoformat()})."
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        help="inclusive window start (YYYY-MM-DD); floored to Jan 1 of the current "
        "year and to the week after the stored watermark",
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        help="inclusive window end (YYYY-MM-DD); defaults to today",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-scrape and upsert every Monday in the requested range, ignoring DB presence",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rankings_flow(start_date=args.start, end_date=args.end, force=args.force)


if __name__ == "__main__":
    main()


# ── Deployment ─────────────────────────────────────────────────────

RANKINGS_DEPLOYMENT_NAME = "rankings"
RANKINGS_CRON = "0 22 * * 1"


def register_deployment() -> None:
    """Create/update the Monday-scheduled rankings deployment (idempotent by name).

    Scheduled production runs use this independent deployment; rankings and
    matches are separate deployments — there is no combined scrape flow.
    Runs on the host ``tennis-pool`` work pool via the host worker and stays
    manually triggerable from the Prefect UI or ``prefect deployment run``.
    Flow code is the local repo checkout (process work pool — no image or
    remote storage is required).
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    # No static parameter defaults: deployment parameters are frozen at
    # registration, so a baked-in date would go stale for later cron runs. The
    # flow defaults to the watermark (see rankings_flow); pass explicit
    # --param start_date/end_date to override for a manual backfill.
    deployment = cast(
        Any,
        rankings_flow.from_source(
            source=str(repo_root),
            entrypoint="src/flows/rankings.py:rankings_flow",
        ),
    )
    deployment.deploy(
        name=RANKINGS_DEPLOYMENT_NAME,
        work_pool_name=WORK_POOL_NAME,
        cron=RANKINGS_CRON,
        build=False,
        ignore_warnings=True,
        print_next_steps=False,
    )
    print(f"Registered deployment {RANKINGS_DEPLOYMENT_NAME!r} (cron {RANKINGS_CRON})")
