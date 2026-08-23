"""Tests for scrape and ETL run-name helpers and source validation."""

from datetime import date

import pytest

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


# ── ETL flow validates the source parameter ──────────────────────


def test_etl_flow_rejects_invalid_source():
    with pytest.raises(ValueError, match="source"):
        etl.etl_flow.fn(source="drift")


def test_etl_flow_accepts_known_sources_and_none(monkeypatch):
    # Patch the body so the guard is the only thing exercised (no DB/work).
    monkeypatch.setattr(etl, "load_env", lambda: None)
    monkeypatch.setattr(etl, "bronze_to_gold", lambda **_kwargs: 0)
    for source in ("rankings", "matches", None):
        etl.etl_flow.fn(source=source)  # must not raise
