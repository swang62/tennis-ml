"""Backfill missing ATP ranking weeks using the persistent browser session."""

from __future__ import annotations

import argparse
import csv
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
    _jitter,
    discover_players,
)

RANKINGS_URL = "https://www.atptour.com/en/rankings/singles?rankRange=0-200&dateWeek={date}"
CURRENT_RANKINGS_CSV = RANKINGS_DIR / "atp_rankings_current.csv"

RANKINGS_TABLE_SELECTOR = "table.mega-table tbody tr a[href*='/en/players/']"
FILTER_SELECTOR = "#dateWeek-filter"
RANKINGS_TABLE_TIMEOUT_MS = 30_000
CHALLENGE_RESOLVE_BUDGET_S = 30
FILTER_VERIFY_BUDGET_S = 15

CLOAKBROWSER_PROFILE_DIR = (
    Path.home() / ".local" / "share" / "tennis-prefect-worker" / "cloakbrowser"
)

_SLUG_RE = r"[^/]+"
_PLAYER_LINK_RE = re.compile(rf"/en/players/({_SLUG_RE})/([^/]+)/overview")
_ATP_PLAYER_ID_RE = re.compile(r"^[A-Za-z0-9]{4}$")
_RANK_CELL_RE = re.compile(r'class="rank\b[^"]*"[^>]*>\s*(\d+)\s*<', re.S)
_POINTS_CELL_RE = re.compile(r'class="points\b[^"]*"[^>]*>(.*?)</td>', re.S)
_RANKINGS_TABLE_RE = re.compile(
    r'<table\b[^>]*\bclass=["\'][^"\']*\bmega-table\b[^"\']*["\'][^>]*>(.*?)</table>',
    re.I | re.S,
)
_ROW_SPLIT_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.S)
_TEXT_TAG_RE = re.compile(r"<[^>]+>")


class RankingsParseError(ValueError):
    """Raised when a fetched rankings page has no parseable rankings table."""


class RankingsNotPublishedError(LookupError):
    """Raised when ATP has not published the requested ranking week."""


def latest_completed_monday(today: date) -> date:
    """Most recent Monday on or before ``today``."""
    days_since_monday = today.weekday()
    return today - timedelta(days=days_since_monday)


def ranking_mondays_after(watermark: date, as_of: date) -> list[date]:
    """Every ATP ranking Monday in (watermark, as_of] when as_of is Monday."""
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


@task
def missing_ranking_mondays(
    start_date: date | None = None,
    end_date: date | None = None,
    force: bool = False,
) -> tuple[date | None, list[date]]:
    """Return ``(watermark, weeks)`` while respecting stored-week and date floors."""
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
    # Resume strictly after the last successful Monday, including across years.
    start = max(start, watermark + timedelta(days=7))
    start += timedelta(days=(7 - start.weekday()) % 7)
    weeks: list[date] = []
    monday = start
    while monday <= end:
        if monday not in stored:
            weeks.append(monday)
        monday += timedelta(days=7)
    return watermark, weeks


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
    if _ATP_PLAYER_ID_RE.fullmatch(player_id) is None:
        return None
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
    """Parse the first rendered rankings table containing player links."""
    for fragment in _RANKINGS_TABLE_RE.findall(html):
        rows: list[dict[str, Any]] = []
        for row_html in _ROW_SPLIT_RE.findall(fragment):
            parsed = _parse_row(row_html)
            if parsed is not None and 1 <= parsed["rank"] <= 200:
                rows.append(parsed)
        if rows:
            return rows
    raise RankingsParseError("no rankings table with player links found in page HTML")


def translate_rank_rows(
    rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
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
    """Return the newest matching bronze tier, or ``None`` when unknown."""
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
    """Open the persistent CloakBrowser profile used by this scrape run."""
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
    """Whether ``week`` appears in the page's #dateWeek-filter."""
    wanted = week.strftime("%Y-%m-%d")
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
    """Navigate the shared page, verify publication, and return rendered HTML."""
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
        raise RankingsNotPublishedError(week.isoformat())
    deadline = time.monotonic() + CHALLENGE_RESOLVE_BUDGET_S
    while True:
        with suppress(Exception):
            page.wait_for_selector(
                RANKINGS_TABLE_SELECTOR,
                state="attached",
                timeout=RANKINGS_TABLE_TIMEOUT_MS,
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
    dry_run: bool = False,
) -> int | None:
    """Fetch one weekly ranking page and upsert its mapped rows.

    Uses the flow's shared page and returns ``None`` when the page cannot be
    accessed or parsed, distinguishing failure from a successful zero-row write.
    """
    url = RANKINGS_URL.format(date=week.isoformat())
    print(f"Week {week.isoformat()}: fetching {url}")
    try:
        html = _fetch_week_html(page, url, week)
    except RankingsNotPublishedError:
        print(f"Week {week.isoformat()}: no rankings published")
        return 0
    if not html:
        print(f"Week {week.isoformat()}: no rankings published")
        return 0
    rows = extract_rankings_from_html(html)

    frame, skipped = translate_rank_rows(rows)
    frame = (
        frame.assign(ranking_date=week)[RANKING_TARGET_COLUMNS]
        .drop_duplicates(subset=["ranking_date", "player_id"], keep="last")
        .reset_index(drop=True)
    )

    if dry_run:
        # Non-mutating preview: report what would be written, but skip the
        # reference-table loads, every DB upsert, CSV append, profile
        # discovery, and CSV sort/write.
        _report_dry_run(week, frame, rows, skipped)
        return len(frame)

    if canonical is None:
        canonical = canonical_players()
    if profiles is None:
        profiles = load_player_metadata()
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


def _report_dry_run(
    week: date,
    frame: pd.DataFrame,
    rows: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> None:
    """Read-only dry-run report of what fetch_and_upsert_week would write.

    Performs no database, CSV, or profile writes.
    """
    print(
        f"[dry-run] Week {week.isoformat()}: {len(rows)} row(s) parsed, "
        f"{len(frame)} would be upserted, {len(skipped)} would be skipped"
    )
    for row in frame.itertuples(index=False):
        print(
            f"[dry-run]   would upsert player_id={row.player_id} "
            f"rank={row.rank} points={row.points}"
        )
    for s in skipped:
        print(
            f"[dry-run]   would skip invalid row: "
            f"player_id={s['player_id']} name={s['name']!r} rank={s['rank']} points={s['points']}"
        )
    print(f"[dry-run] Week {week.isoformat()}: no writes performed")


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
            key = (
                pd.Timestamp(row["ranking_date"]).strftime("%Y%m%d"),
                str(row["player_id"]),
            )
            if key not in existing:
                writer.writerow(
                    [
                        key[0],
                        int(row["rank"]),
                        key[1],
                        "" if pd.isna(row["points"]) else int(row["points"]),
                    ]
                )


def sort_current_rankings_csv() -> None:
    """Prefer legacy ids for duplicate date/rank rows, then sort the CSV."""
    with open(CURRENT_RANKINGS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != RANKINGS_COLUMNS:
            raise ValueError(f"{CURRENT_RANKINGS_CSV.name}: expected columns {RANKINGS_COLUMNS}")
        rows = list(reader)
    numeric_keys = {(row["ranking_date"], row["rank"]) for row in rows if row["player"].isdigit()}
    rows = [
        row
        for row in rows
        if row["player"].isdigit() or (row["ranking_date"], row["rank"]) not in numeric_keys
    ]
    rows.sort(key=lambda row: (int(row["ranking_date"]), int(row["rank"])))
    with open(CURRENT_RANKINGS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RANKINGS_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def scrape_run_name(start_date: date | None, end_date: date | None) -> str:
    """Flow-run name for a scrape flow: a dated range when explicit, else 'latest'.

    Uses ISO dates for explicit bounds and ``latest`` for omitted bounds.
    """
    start = start_date.isoformat() if start_date is not None else "latest"
    end = end_date.isoformat() if end_date is not None else "latest"
    if start == "latest" and end == "latest":
        return "scrape-latest"
    return f"scrape-{start}-{end}"


def _scrape_flow_run_name() -> str:
    """Prefect ``flow_run_name`` callable: resolve the name from the run's params."""
    from prefect.runtime import flow_run as flow_run_runtime

    try:
        params = flow_run_runtime.get_parameters()
    except Exception:
        params = {}
    return scrape_run_name(params.get("start_date"), params.get("end_date"))


@flow(log_prints=True, retries=1, flow_run_name=_scrape_flow_run_name)
def rankings_flow(
    start_date: date | None = None,
    end_date: date | None = None,
    force: bool = False,
    dry_run: bool = False,
):
    """Fetch missing ranking weeks while preserving the watermark and browser invariants."""
    load_env()
    watermark, weeks = missing_ranking_mondays(start_date, end_date, force)
    if watermark is None and not force:
        print(
            "bronze.rankings is empty — initial seed not complete; "
            "run `just seed` first. Skipping browser work."
        )
        if not dry_run:
            sort_current_rankings_csv()
        return
    if not weeks:
        if force:
            print("No ranking weeks in the forced range.")
        else:
            assert watermark is not None
            print(f"Rankings are current through {watermark.isoformat()} — no missing weeks.")
        if not dry_run:
            sort_current_rankings_csv()
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
                dry_run=dry_run,
            )
            if rows is not None:
                found_data = True
                stored += rows
    finally:
        # Always release the server-side browser session.
        print("Closing browser session")
        if page is not None:
            page.close()
        browser.close()
    _fail_if_no_data_found(found_data, weeks)
    if not dry_run:
        sort_current_rankings_csv()
    print(f"Scrape complete: {stored} rows stored")


def _fail_if_no_data_found(found_data: bool, weeks: list[date]) -> None:
    """Fail when no requested week could be fetched or parsed."""
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch and parse every week but skip all DB upserts, CSV writes, and "
        "profile discovery; report what would be written",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rankings_flow(
        start_date=args.start,
        end_date=args.end,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()


# ── Deployment ─────────────────────────────────────────────────────

RANKINGS_DEPLOYMENT_NAME = "rankings"
RANKINGS_CRON = "0 22 * * 2"


def register_deployment() -> None:
    """Create or update the scheduled rankings deployment."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    # Leave dates unset so scheduled runs resolve the current watermark.
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
