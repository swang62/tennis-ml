from datetime import datetime
from unittest.mock import MagicMock

import pytest

import src.flows.etl as etl
from src.flows.etl import BASE_PHASE_MODELS, FINAL_PHASE_MODELS, etl_flow

# Used by etl_flow to build gold; patched so the flow test never runs dbt.
FAKE_GOLD_COUNT = 1


def _patch_psycopg(monkeypatch):
    """Patch psycopg so _current_gold_count returns zeros without a live DB."""
    conn = MagicMock()
    direct_conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    conn.__enter__.return_value = direct_conn
    direct_conn.cursor.return_value.__enter__.return_value = cursor
    monkeypatch.setattr("src.flows.etl.psycopg.connect", lambda *_a, **_k: conn)


def _common_patches(monkeypatch, events, watermark_recorded):
    """Wire the dbt/elo/watermark boundaries to in-memory recorders."""
    monkeypatch.setattr(
        "src.flows.etl.run_dbt_build",
        lambda **kwargs: events.append(("dbt", kwargs.get("incremental"), kwargs.get("select"))),
    )
    monkeypatch.setattr("src.flows.etl.materialize_elo", lambda: events.append(("elo",)))
    monkeypatch.setattr("src.flows.etl._dbt_model_rows", lambda: {})
    monkeypatch.setattr("src.flows.etl._dbt_summary", lambda _log: "Done. PASS=1")
    monkeypatch.setattr(
        "src.flows.etl._record_incremental_watermark",
        lambda w: watermark_recorded.append(w),
    )
    _patch_psycopg(monkeypatch)


def test_etl_flow_maps_incremental_to_bronze_to_gold(monkeypatch):
    """etl_flow's user-facing incremental option flows through untouched."""
    received = []

    def fake_bronze_to_gold(**kwargs):
        received.append(kwargs)
        return FAKE_GOLD_COUNT

    monkeypatch.setattr("src.flows.etl.bronze_to_gold", fake_bronze_to_gold)
    monkeypatch.setattr("src.flows.etl.load_env", lambda: None)

    etl_flow.fn(incremental=True)
    etl_flow.fn()

    assert received == [{"incremental": True}, {"incremental": False}]


def test_bronze_to_gold_runs_base_elo_final_in_order(monkeypatch):
    """Bronze-to-gold runs base dbt, then Elo, then final dbt, then the watermark."""
    events: list = []
    watermark_recorded: list = []
    _common_patches(monkeypatch, events, watermark_recorded)
    monkeypatch.setattr(
        "src.flows.etl._incremental_watermarks",
        lambda: (datetime(2026, 1, 2), datetime(2026, 1, 1)),
    )

    etl.bronze_to_gold.fn(incremental=True)

    assert events[0] == ("dbt", True, BASE_PHASE_MODELS)
    assert events[1] == ("elo",)
    assert events[2] == ("dbt", True, FINAL_PHASE_MODELS)
    # Watermark advances only after every phase, with the source watermark.
    assert watermark_recorded == [datetime(2026, 1, 2)]


def test_bronze_to_gold_profile_only_skips_elo_and_match_features(monkeypatch):
    """No new bronze matches refreshes player_profiles only; Elo and final are skipped."""
    events: list = []
    watermark_recorded: list = []
    _common_patches(monkeypatch, events, watermark_recorded)
    # source_watermark == built_watermark triggers the profile-only shortcut.
    monkeypatch.setattr(
        "src.flows.etl._incremental_watermarks",
        lambda: (datetime(2026, 1, 1), datetime(2026, 1, 1)),
    )

    etl.bronze_to_gold.fn(incremental=True)

    assert events == [("dbt", True, ["player_profiles"])]
    assert ("elo",) not in events
    assert watermark_recorded == []


def test_elo_failure_blocks_watermark_and_final_phase(monkeypatch):
    """An Elo failure aborts before the final dbt phase and before the watermark."""
    events: list = []
    watermark_recorded: list = []
    _common_patches(monkeypatch, events, watermark_recorded)
    monkeypatch.setattr(
        "src.flows.etl._incremental_watermarks",
        lambda: (datetime(2026, 1, 2), datetime(2026, 1, 1)),
    )

    def boom():
        events.append(("elo",))
        raise RuntimeError("elo boom")

    monkeypatch.setattr("src.flows.etl.materialize_elo", boom)

    with pytest.raises(RuntimeError):
        etl.bronze_to_gold.fn(incremental=False)

    assert events[0][0] == "dbt"  # base phase ran
    assert events[1] == ("elo",)
    assert ("dbt", False, FINAL_PHASE_MODELS) not in events
    assert watermark_recorded == []


def test_final_phase_failure_blocks_watermark(monkeypatch):
    """A final-phase dbt failure aborts before the watermark advances."""
    events: list = []
    watermark_recorded: list = []
    _common_patches(monkeypatch, events, watermark_recorded)
    monkeypatch.setattr(
        "src.flows.etl._incremental_watermarks",
        lambda: (datetime(2026, 1, 2), datetime(2026, 1, 1)),
    )

    def fake_dbt(**kwargs):
        if kwargs.get("select") == FINAL_PHASE_MODELS:
            raise RuntimeError("dbt final boom")
        events.append(("dbt", kwargs.get("incremental"), kwargs.get("select")))

    monkeypatch.setattr("src.flows.etl.run_dbt_build", fake_dbt)

    with pytest.raises(RuntimeError):
        etl.bronze_to_gold.fn(incremental=False)

    assert events[0] == ("dbt", False, BASE_PHASE_MODELS)
    assert events[1] == ("elo",)
    assert watermark_recorded == []
