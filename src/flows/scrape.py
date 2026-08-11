"""Prefect flow: automated weekly ATP rankings catch-up via CloakBrowser.

The database is the self-healing watermark: each run queries
``MAX(bronze.rankings.ranking_date)`` and processes every later ATP ranking
Monday through the latest completed Monday, oldest first. Weeks are fetched
from the ATP Tour site with the CloakBrowser Python library (an interactive
stealth-Chromium Playwright wrapper) — never the CloakBrowser MCP server.

Identity is resolved only through the approved ranking identity map
(``load_ranking_player_map``): the page exposes each player's canonical
ATP_Database id in the profile URL slug (e.g. ``/en/players/jannik-sinner/s0ag/overview``),
which must be present as a value in the reviewed map to be stored. Raw
numeric ranking source ids never reach the database, and unmapped players are
skipped and reported.

Session discipline (CloakBrowser free tier tracks sessions server-side; a
process that exits without a clean ``browser.close()`` permanently wedges that
session id): the flow launches exactly ONE browser for the whole run, loops
every missing week inside a try block, and the ``finally`` block always calls
``browser.close()`` before the process can exit — including when a week fails,
so Prefect records the failure but the session is released.

Each week commits independently, so a failed week raises with URL/date
evidence, leaves all previously ingested weeks intact, and is retried on the
next run. A run with no missing weeks (or with an empty table, meaning the
initial historical backfill is not complete) logs and exits without launching
a browser.

Run once (manual/local):  ``just rankings-fetch``
Register the Monday deployment:  ``just rankings-fetch --deploy``
"""

from __future__ import annotations

import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd
from prefect import flow, task

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
# means this never matches: the fetch fails visibly with URL/date evidence
# instead of writing partial data.
RANKINGS_TABLE_SELECTOR = "table.mega-table"

RANKINGS_DEPLOYMENT_NAME = "rankings-catchup"
RANKINGS_CRON = "0 6 * * 1"  # Monday 06:00 UTC
WORK_POOL_NAME = "tennis-pool"

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
    """Every ATP ranking Monday in (watermark, latest_completed_monday(as_of)].

    The watermark is itself a ranking Monday, so the first candidate is
    ``watermark + 7 days`` and every further Monday is a 7-day step.
    """
    end = latest_completed_monday(as_of)
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


@task
def missing_ranking_mondays(as_of: date | None = None) -> tuple[date | None, list[date]]:
    """DB watermark plus every later ranking Monday, oldest first.

    Returns ``(watermark, weeks)``; ``watermark`` is None when the table is
    empty (initial historical backfill not complete) and ``weeks`` is then
    empty because the flow must not scrape history from scratch.
    """
    watermark = current_watermark()
    if watermark is None:
        return None, []
    as_of = as_of or date.today()
    return watermark, ranking_mondays_after(watermark, as_of)


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
    """Open the single CloakBrowser session used for the whole catch-up run.

    Headed matches the local interactive convention; the flow is responsible
    for releasing the session with ``browser.close()`` in its finally block.
    """
    import cloakbrowser

    return cloakbrowser.launch(headless=False)


def _fetch_week_html(browser, url: str) -> str:
    """Load one weekly rankings URL in the shared browser; returns rendered HTML.

    A fresh page per week, closed on the way out. Nothing is persisted — no
    challenge/session credentials survive a run. Normal browser-navigation
    blocks (challenges, network failures) surface as exceptions here.
    """
    page = browser.new_page()
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        # The rankings tables are server-rendered into the DOM but start hidden
        # (the .non-live state), so "attached" — not "visible" — is the gate.
        page.wait_for_selector(RANKINGS_TABLE_SELECTOR, state="attached", timeout=45_000)
        return page.content()
    finally:
        page.close()


def fetch_and_upsert_week(browser, week: date, rank_map: dict[str, str]) -> int:
    """Fetch one weekly ranking page and upsert its mapped rows; returns count.

    Uses the run's shared browser session (never launches its own — the flow
    owns the single session and closes it in finally). Commits independently
    (``_copy_df_into`` runs in one transaction), so a failure here leaves every
    previously ingested week intact.
    """
    url = RANKINGS_URL.format(date=week.isoformat())
    try:
        html = _fetch_week_html(browser, url)
        rows = extract_rankings_from_html(html)
    except Exception as exc:
        raise RuntimeError(
            f"failed to load or parse ATP rankings page for week {week.isoformat()} ({url}): {exc}"
        ) from exc

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
    print(f"Week {week.isoformat()}: {len(frame)} rows stored, {len(skipped)} skipped")
    return len(frame)


@flow(log_prints=True)
def rankings_catchup_flow(as_of_date: date | None = None):
    """Catch up bronze.rankings from the DB watermark to the latest Monday.

    Launches exactly one CloakBrowser session, processes every missing week
    inside a try block, and closes the browser in finally — even when a week
    fails, the session is always released before the flow returns/raises.
    """
    load_env()
    watermark, weeks = missing_ranking_mondays(as_of_date)
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
    stored = 0
    try:
        for week in weeks:
            stored += fetch_and_upsert_week(browser, week, rank_map)
    finally:
        # CloakBrowser free tier tracks sessions server-side; an exit without a
        # clean close permanently wedges the session id, so the single session
        # is always released here before the exception (if any) propagates.
        browser.close()
    print(f"Rankings catch-up complete: {stored} rows stored")


def register_deployment() -> None:
    """Create/update the Monday-scheduled catch-up deployment (idempotent by name).

    The deployment runs on the host ``tennis-pool`` work pool via the existing
    host worker and stays manually triggerable from the Prefect UI or
    ``prefect deployment run``. Flow code is the local repo checkout (process
    work pool — no image/remote storage is required).
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    # prefect's async_dispatch stubs from_source() as a Coroutine union, but in
    # a sync context it returns the Flow itself; cast to Any sidesteps that.
    deployment = cast(
        Any,
        rankings_catchup_flow.from_source(
            source=str(repo_root),
            entrypoint="src/flows/scrape.py:rankings_catchup_flow",
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


def main(argv: list[str] | None = None) -> None:
    """Console-script entry for `just rankings-fetch [--deploy]`."""
    args = sys.argv[1:] if argv is None else argv
    if "--deploy" in args:
        register_deployment()
    else:
        rankings_catchup_flow()


if __name__ == "__main__":
    main()
