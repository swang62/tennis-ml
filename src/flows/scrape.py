"""Prefect flow: ATP site scraping (currently weekly rankings catch-up).

The database is the self-healing watermark (``MAX(bronze.rankings.ranking_date)``).
Runs with no params fetch only the most recent completed Monday (the previous
week). Manual backfills pass ``--param end_date=YYYY-MM-DD`` to fetch every
missing Monday after the watermark through that date, oldest first; an
``end_date`` at or before the watermark fetches nothing. Weeks are fetched from
the ATP Tour site with the CloakBrowser Python library (an interactive
stealth-Chromium Playwright wrapper) — never the CloakBrowser MCP server.

Identity is resolved only through the approved ranking identity map
(``load_ranking_player_map``): the page exposes each player's canonical
ATP_Database id in the profile URL slug (e.g. ``/en/players/jannik-sinner/s0ag/overview``),
which must be present as a value in the reviewed map to be stored. Raw
numeric ranking source ids never reach the database, and unmapped players are
skipped and reported.

Session discipline (CloakBrowser free tier tracks sessions server-side; a
process that exits without a clean ``browser.close()`` permanently wedges that
session id): the flow launches exactly ONE persistent browser for the whole
run, navigates a single page across every missing week inside a try block, and
the ``finally`` block always closes the page and browser before the process can
exit. The persistent profile retains Cloudflare clearance cookies between runs,
and humanized navigation inserts random jitter so page moves are not detected
as a bot.

A week whose page fails to load or carries no parseable rankings table is
logged and skipped, never treated as a failure — a Cloudflare challenge, a
missing rankings page for that week, or a markup change all move on to the next
week instead of aborting the backfill. Each successful week commits
independently.

Run once (manual/local):  ``just scrape``
Backfill to a date:       ``just scrape --param end_date=YYYY-MM-DD``
"""

from __future__ import annotations

import random
import re
import time
from contextlib import suppress
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd
from prefect import flow, task
from prefect.deployments import run_deployment

from src.db.client import get_conn
from src.db.ingest import (
    BRONZE_RANKINGS_TABLE,
    RANKING_TARGET_COLUMNS,
    _copy_df_into,
    load_ranking_player_map,
)
from src.utils import load_env

RANKINGS_URL = "https://www.atptour.com/en/rankings/singles?rankRange=0-200&dateWeek={date}"

# Rankings table anchor, stable across ATP Tour page versions (both the mobile
# and the desktop rankings table carry this class; the parser uses only the
# first table containing player links). A Cloudflare challenge or markup change
# means this never matches: the week is logged and skipped, never a failure.
# Rows are targeted (not the table element) because the table is permanently
# hidden by the .non-live state — the element exists in the DOM immediately but
# the row links only appear once the server-side data has rendered.
RANKINGS_TABLE_SELECTOR = "table.mega-table tbody tr a[href*='/en/players/']"
# Per-week page render budget. Reapplied fresh on every wait (one wait per
# week), so a slow page never eats into the next week's budget. 30s per page is
# ample for a normal rankings render.
RANKINGS_TABLE_TIMEOUT_MS = 30_000
# Total budget for the rows to render (Cloudflare auto-clear or server-side
# render) before a week is skipped. 30s is the outer bound — anything slower
# than that is not going to resolve, and a genuine missing week has no rows to
# render and is skipped immediately.
CHALLENGE_RESOLVE_BUDGET_S = 30

SCRAPE_DEPLOYMENT_NAME = "scrape"
SCRAPE_CRON = "0 6 * * 1"  # Monday 06:00 UTC
# The ETL deployment that this flow triggers when a scrape actually stored rows.
# run_deployment resolves it by "<flow-name>/<deployment-name>".
ETL_DEPLOYMENT_REF = "etl-flow/etl"
WORK_POOL_NAME = "tennis-pool"
CLOAKBROWSER_PROFILE_DIR = (
    Path.home() / ".local" / "share" / "tennis-prefect-worker" / "cloakbrowser"
)

# Player profile link inside each rankings row: /en/players/<slug>/<id>/overview
# where <id> is the canonical ATP_Database id (lowercased in the URL). The
# rank and points cells are matched by class prefix ("rank", "points") across
# the current mobile/desktop table variants; the empty <li class="rank"> move
# indicator inside the player cell never matches the rank regex because it
# carries no bare digits.
_PLAYER_LINK_RE = re.compile(r"/en/players/[^/]+/([^/]+)/overview")
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
    with get_conn().cursor() as cur:
        cur.execute(f"SELECT MAX(ranking_date) FROM {BRONZE_RANKINGS_TABLE}")
        row = cur.fetchone()
    return row[0] if row is not None and row[0] is not None else None


def stored_ranking_mondays() -> set[date]:
    """Every ranking Monday currently present in bronze.rankings."""
    with get_conn().cursor() as cur:
        cur.execute(f"SELECT DISTINCT ranking_date FROM {BRONZE_RANKINGS_TABLE}")
        return {row[0] for row in cur.fetchall()}


@task
def missing_ranking_mondays(end_date: date | None = None) -> tuple[date | None, list[date]]:
    """Weeks to fetch: every missing Monday through ``end_date`` (default previous week).

    Returns ``(watermark, weeks)``; ``watermark`` is None when the table is
    empty (initial historical backfill not complete) and ``weeks`` is then
    empty because the flow must not scrape history from scratch.

    The scan always starts at the watermark and moves one week into the future
    at a time. With no ``end_date`` it stops at the most recent completed
    Monday (the previous week); with ``end_date`` it is the inclusive upper
    bound, so a backfill naturally quits when ``end_date`` is at or before the
    watermark. Every Monday through that end that is not already stored is
    returned oldest first — already-present weeks and interior gaps are handled
    by the same stored-set scan.
    """
    stored = stored_ranking_mondays()
    if not stored:
        return None, []
    watermark = max(stored)
    end = end_date or latest_completed_monday(date.today())
    weeks: list[date] = []
    m = watermark + timedelta(days=7)
    while m <= end:
        if m not in stored:
            weeks.append(m)
        m += timedelta(days=7)
    return watermark, weeks


def _jitter() -> None:
    """Human-like random pause between navigation steps (bot-detection resistance)."""
    time.sleep(random.uniform(0.8, 2.5))


# ── HTML parsing (fixture-testable, no network) ──────────────────


def _cell_text(cell_html: str) -> str:
    """Strip tags/entities from a cell fragment; commas are thousands separators."""
    return _TEXT_TAG_RE.sub("", cell_html).replace("&nbsp;", " ").strip()


def _parse_row(row_html: str) -> dict[str, Any] | None:
    """One rankings row -> {rank, points, name, player_id}; None when not a player row."""
    link = _PLAYER_LINK_RE.search(row_html)
    if link is None:
        return None
    player_id = link.group(1).upper()
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
    return {"rank": rank, "points": points, "name": name, "player_id": player_id}


def extract_rankings_from_html(html: str) -> list[dict[str, Any]]:
    """Parse the rendered ATP rankings page into rank/points/name/player_id rows.

    Uses only the first page table that contains player profile links (the
    current page renders a concise and a full table with identical data).
    Raises RankingsParseError when no table with player links is found, so a
    challenge page or a markup change fails visibly.
    """
    fragments = _TABLE_SPLIT_RE.split(html)
    for fragment in fragments[1:]:  # skip everything before the first <table
        rows: list[dict[str, Any]] = []
        for row_html in _ROW_SPLIT_RE.findall(fragment):
            parsed = _parse_row(row_html)
            if parsed is not None:
                rows.append(parsed)
        if rows:
            return rows
    raise RankingsParseError("no rankings table with player links found in page HTML")


# ── Identity translation + upsert ────────────────────────────────


def translate_rank_rows(
    rows: list[dict[str, Any]], rank_map: dict[str, str]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Keep only page rows whose canonical id is approved by the identity map.

    The map validates identity: a player whose canonical id is not a map value
    is skipped (and returned for reporting), never silently ingested. Raw
    ranking source ids are irrelevant here — the page exposes canonical ids.
    """
    approved = set(rank_map.values())
    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        if row["player_id"] in approved and 1 <= row["rank"] <= 200:
            kept.append(row)
        else:
            skipped.append(row)
    if not kept:
        return pd.DataFrame(columns=["player_id", "rank", "points"]), skipped
    frame = pd.DataFrame(kept)[["player_id", "rank", "points"]]
    frame["player_id"] = frame["player_id"].astype(str)
    return cast(pd.DataFrame, frame), skipped


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
        args=["--disable-http2"] if first_launch else [],
    )


def _week_in_filter(page, week: date) -> bool:
    """Whether ``week`` appears in the page's #dateWeek-filter.

    The filter lists every published ranking week; the requested week shows up
    as an option when it exists. Option values use "Current Week" for the most
    recent week and YYYY-MM-DD for older ones, but the option TEXT is always
    YYYY.MM.DD — so matching on text covers both. A week that was never
    published (e.g. a future Monday) is absent from the filter entirely.
    """
    wanted = week.strftime("%Y.%m.%d")
    return page.evaluate(
        "(wanted) => Array.from("
        "document.querySelectorAll('#dateWeek-filter option')"
        ").some(o => o.textContent.trim() === wanted)",
        wanted,
    )


def _fetch_week_html(page, url: str, week: date) -> str:
    """Navigate the shared rankings page to one week and return its HTML.

    The page stays open for the entire flow so its navigation state and the
    persistent profile's Cloudflare clearance are reused for every week. Random
    jitter before/after navigation keeps the moves human-like.

    A week that was never published (the #dateWeek-filter has no option for it)
    is rejected immediately — no point waiting for rows that cannot render. The
    gate for real weeks is the captured HTML containing actual player links:
    the table element itself is hidden by the .non-live state, so "attached" on
    the table fires before the rows render, and "visible" never fires at all.
    The loop therefore waits until either the row links appear in
    ``page.content()`` or the CHALLENGE_RESOLVE_BUDGET_S deadline passes (a
    Cloudflare challenge auto-clears in between). The caller decides whether a
    final no-table page is a missing week.
    """
    _jitter()
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    _jitter()
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


def fetch_and_upsert_week(page, week: date, rank_map: dict[str, str]) -> int:
    """Fetch one weekly ranking page and upsert its mapped rows; returns count.

    Uses the run's shared browser page (never launches its own — the flow owns
    the page and closes it in finally). Commits independently
    (``_copy_df_into`` runs in one transaction). A page that cannot be loaded or
    parsed (Cloudflare challenge, missing rankings week, markup change) is
    logged and skipped, returning 0 — never a failure, so a full backfill moves
    past blocked weeks.
    """
    url = RANKINGS_URL.format(date=week.isoformat())
    print(f"Week {week.isoformat()}: fetching {url}")
    try:
        html = _fetch_week_html(page, url, week)
        rows = extract_rankings_from_html(html)
    except Exception as exc:
        print(f"Week {week.isoformat()}: skipped (could not load or parse): {exc}")
        return 0

    frame, skipped = translate_rank_rows(rows, rank_map)
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
    for s in skipped:
        print(
            f"  skipped: player_id={s['player_id']} name={s['name']!r} "
            f"rank={s['rank']} points={s['points']}"
        )
    print(f"Week {week.isoformat()}: succeeded: {len(frame)} rows stored, {len(skipped)} skipped")
    return len(frame)


@flow(log_prints=True)
def scrape_flow(end_date: date | None = None):
    """Scrape missing ranking weeks: previous week (no params) or backfill to a date.

    With no ``end_date`` only the most recent completed Monday (the previous
    week) is fetched — the same for scheduled cron runs, ``prefect deployment
    run`` without params, and direct local calls, so a bare run never silently
    backfills far history. With an explicit ``end_date`` every missing Monday
    from the watermark through that date is fetched oldest first; an
    ``end_date`` at or before the watermark naturally fetches nothing. Launches
    exactly one CloakBrowser session, processes every missing week inside a try
    block, and closes the browser in finally — even when a week fails, the
    session is always released before the flow returns/raises.
    """
    load_env()
    watermark, weeks = missing_ranking_mondays(end_date)
    if watermark is None:
        print(
            "bronze.rankings is empty — initial seed not complete; "
            "run `just db-seed` first. Skipping browser work."
        )
        return
    if not weeks:
        print(f"Rankings are current through {watermark.isoformat()} — no missing weeks.")
        return
    print(
        f"Watermark {watermark.isoformat()}: fetching {len(weeks)} missing week(s), "
        f"oldest first: {weeks[0].isoformat()} .. {weeks[-1].isoformat()}"
    )

    rank_map = load_ranking_player_map()
    browser = _launch_browser()
    page = None
    stored = 0
    try:
        page = browser.new_page()
        print("Browser session open: navigating one page across all weeks")
        for week in weeks:
            stored += fetch_and_upsert_week(page, week, rank_map)
    finally:
        # CloakBrowser tracks sessions server-side; an exit without a clean
        # close permanently wedges the session id, so the browser (and any page
        # that was created) is always released here — even when a week or page
        # creation fails. Closing the context closes its pages too.
        print("Closing browser session")
        if page is not None:
            page.close()
        browser.close()
    print(f"Scrape complete: {stored} rows stored")
    _trigger_etl_if_needed(stored)


def _trigger_etl_if_needed(stored: int) -> None:
    """Trigger the ETL deployment only when the scrape stored new rows."""
    if stored == 0:
        print("No new rows stored — skipping ETL trigger.")
        return
    print(f"New data stored ({stored} rows); triggering {ETL_DEPLOYMENT_REF}...")
    # Native in-flow trigger: submits a flow run for the ETL deployment on the
    # same Prefect server (no webhook/API-key plumbing needed). timeout=0 is
    # fire-and-forget — the scrape run finishes without waiting on the ETL.
    run_deployment(ETL_DEPLOYMENT_REF, timeout=0)


def register_deployment() -> None:
    """Create/update the Monday-scheduled scrape deployment (idempotent by name).

    The deployment runs on the host ``tennis-pool`` work pool via the existing
    host worker and stays manually triggerable from the Prefect UI or
    ``prefect deployment run``. Flow code is the local repo checkout (process
    work pool — no image/remote storage is required).
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    # No static parameter defaults here: deployment parameters are frozen at
    # registration, so a baked-in date would go stale for later cron runs. The
    # flow defaults to the previous week itself (see scrape_flow); pass an
    # explicit --param end_date to override for a manual backfill.
    # prefect's async_dispatch stubs from_source() as a Coroutine union, but in
    # a sync context it returns the Flow itself; cast to Any sidesteps that.
    deployment = cast(
        Any,
        scrape_flow.from_source(
            source=str(repo_root),
            entrypoint="src/flows/scrape.py:scrape_flow",
        ),
    )
    deployment.deploy(
        name=SCRAPE_DEPLOYMENT_NAME,
        work_pool_name=WORK_POOL_NAME,
        cron=SCRAPE_CRON,
        build=False,
        ignore_warnings=True,
        print_next_steps=False,
    )
    print(f"Registered deployment {SCRAPE_DEPLOYMENT_NAME!r} (cron {SCRAPE_CRON})")
