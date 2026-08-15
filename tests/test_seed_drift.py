"""Hermetic tests for the dev-only drift fixture seeder (no live database)."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.constants import DRIFT_MIN_N_FOR_CHECK, ROOT
from src.db import seed_drift
from src.features.validate import run_ingestion_checks

ATP_DATABASE_CSV = ROOT / "data" / "ATP_player_database.csv"


def _canonical_ids() -> set[str]:
    with open(ATP_DATABASE_CSV) as f:
        return {str(row["id"]) for row in csv.DictReader(f)}


def _real_source_matches() -> dict[str, dict[Any, Any]]:
    """All bronze rows the raw corpus transforms to, keyed by source match_id."""
    source = seed_drift.atp_rows_to_bronze(seed_drift.load_raw_atp_rows(seed_drift.RAW_YEAR))
    return {str(r["match_id"]): dict(r) for r in source.to_dict(orient="records")}


def test_fixture_rows_pass_existing_bronze_ingestion_checks():
    df = seed_drift.build_fixture_rows()

    report = run_ingestion_checks(df)

    assert report["passed"] is True
    assert report["input_rows"] == seed_drift.N_MATCHES
    assert report["valid_rows"] == seed_drift.N_MATCHES
    assert report["dropped_rows"] == 0


def test_fixture_meets_drift_minimum_data_guard():
    df = seed_drift.build_fixture_rows()

    # 10 physical matches required; 12 -> 24 symmetric scored orientations.
    assert len(df) >= DRIFT_MIN_N_FOR_CHECK
    assert len(df) == seed_drift.N_MATCHES
    assert len(df["match_id"]) == len(set(df["match_id"]))


def test_fixture_rows_clone_real_source_matches():
    """Every fixture row is a real source match with only id/date overridden."""
    fixture = seed_drift.build_fixture_rows()
    source_by_id = _real_source_matches()
    copied = [c for c in seed_drift.BRONZE_COLUMNS if c not in ("match_id", "match_date")]

    matched_source_ids: set[str] = set()
    for _, row in fixture.iterrows():
        row_values = [row[c] for c in copied]
        candidates = {
            sid
            for sid, src in source_by_id.items()
            if pd.Series(row_values).equals(pd.Series([src[c] for c in copied]))
        }
        assert candidates, f"fixture row {row['match_id']} matches no real source row"
        matched_source_ids |= candidates

    # Exactly N_MATCHES distinct source matches, one per fixture row: no source
    # match identity is reused and no pairing is invented.
    assert len(matched_source_ids) == seed_drift.N_MATCHES


def test_fixture_uses_real_canonical_player_ids():
    df = seed_drift.build_fixture_rows()
    player_ids = set(df["player1_id"]) | set(df["player2_id"])

    assert player_ids
    assert player_ids <= _canonical_ids()


def test_fixture_respects_bronze_conventions():
    df = seed_drift.build_fixture_rows()

    # Winner is always player1 (schema CHECK) and the two sides are distinct.
    assert (df["winner_id"] == df["player1_id"]).all()
    assert (df["player1_id"] != df["player2_id"]).all()


def test_fixture_dates_are_after_the_seed_horizon():
    df = seed_drift.build_fixture_rows()

    assert df["match_date"].min() == seed_drift.DEFAULT_AFTER.isoformat()
    assert df["match_date"].max() > seed_drift.DEFAULT_AFTER.isoformat()

    rows = seed_drift.load_raw_atp_rows(seed_drift.RAW_YEAR)
    horizon = max(int(r["tourney_date"]) for r in rows)
    assert date(horizon // 10000, horizon % 10000 // 100, horizon % 100) < seed_drift.DEFAULT_AFTER


def test_fixture_is_deterministic():
    assert seed_drift.build_fixture_rows().equals(seed_drift.build_fixture_rows())


def test_validate_fixture_accepts_built_rows():
    df = seed_drift.build_fixture_rows()

    seed_drift.validate_fixture(df)  # must not raise


def test_validate_fixture_raises_on_invalid_rows():
    df = seed_drift.build_fixture_rows()
    df.loc[0, "winner_id"] = df.loc[0, "player2_id"]  # breaks winner = player1

    with pytest.raises(ValueError, match="failed bronze validation"):
        seed_drift.validate_fixture(df)


def test_main_dry_run_never_writes(monkeypatch, capsys):
    calls = []

    def fake_insert(_df, **kwargs):
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(seed_drift, "insert_bronze_rows", fake_insert)

    assert seed_drift.main(["--dry-run"]) == 0
    assert calls == []
    assert "no rows written" in capsys.readouterr().out


def test_main_inserts_idempotently_by_default(monkeypatch):
    calls = []

    def fake_insert(_df, **kwargs):
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(seed_drift, "insert_bronze_rows", fake_insert)

    seed_drift.main([])
    assert calls == [{"overwrite": False}]


def test_main_force_overwrites(monkeypatch):
    calls = []

    def fake_insert(_df, **kwargs):
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(seed_drift, "insert_bronze_rows", fake_insert)

    seed_drift.main(["--force"])
    assert calls == [{"overwrite": True}]


def test_main_rejects_after_before_seed_horizon(monkeypatch):
    monkeypatch.setattr(seed_drift, "insert_bronze_rows", lambda _df, **_kwargs: 0)

    with pytest.raises(ValueError, match="not after the seed horizon"):
        seed_drift.main(["--after", "2026-01-01"])


def test_main_prints_etl_and_removal_guidance(monkeypatch, capsys):
    monkeypatch.setattr(seed_drift, "insert_bronze_rows", lambda _df, **_kwargs: 0)

    seed_drift.main([])

    out = capsys.readouterr().out
    assert "just etl" in out
    assert "DELETE FROM bronze.match_events" in out
    assert "match_id LIKE 'drift-%'" in out


def test_parse_args_defaults():
    args = seed_drift.parse_args([])
    assert args.after == seed_drift.DEFAULT_AFTER
    assert args.force is False
    assert args.dry_run is False
