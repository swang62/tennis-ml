"""Tests for scrape and ETL run-name helpers."""

from datetime import date

import src.flows.etl as etl
import src.flows.rankings as rankings

# ── Pure naming helpers ──────────────────────────────────────────


def test_scrape_run_name_both_dates():
    name = rankings.scrape_run_name(date(2024, 1, 1), date(2024, 2, 1))
    assert name == "scrape-2024-01-01-2024-02-01"


def test_scrape_run_name_neither_date_is_latest():
    name = rankings.scrape_run_name(None, None)
    assert name == "scrape-latest"


def test_scrape_run_name_start_only():
    name = rankings.scrape_run_name(date(2024, 1, 1), None)
    assert name == "scrape-2024-01-01-latest"


def test_scrape_run_name_end_only():
    name = rankings.scrape_run_name(None, date(2024, 2, 1))
    assert name == "scrape-latest-2024-02-01"


def test_scrape_run_name_distinguishes_omitted_from_explicit():
    # Explicit dates and omitted params must not produce the same name.
    assert rankings.scrape_run_name(None, None) != rankings.scrape_run_name(
        date(2024, 1, 1), date(2024, 2, 1)
    )


def test_etl_run_name_by_source():
    assert etl.etl_run_name("rankings") == "etl-rankings"
    assert etl.etl_run_name("matches") == "etl-matches"


def test_etl_run_name_manual_when_unset_or_unknown():
    assert etl.etl_run_name(None) == "etl-manual"
    assert etl.etl_run_name("drift") == "etl-manual"
