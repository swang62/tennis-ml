from datetime import datetime

import src.flows.etl as etl
from src.flows.etl import etl_flow

# Used by etl_flow to build gold; patched so the flow test never runs dbt.
FAKE_GOLD_COUNT = 1


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


def test_bronze_to_gold_maps_incremental_to_dbt_full_refresh(monkeypatch):
    """bronze_to_gold translates incremental=False (default) into dbt
    --full-refresh; incremental=True skips it. run_dbt_build keeps its
    full_refresh parameter for other callers (e.g. drift)."""
    from unittest.mock import MagicMock

    dbt_calls = []
    monkeypatch.setattr("src.flows.etl.run_dbt_build", lambda **kwargs: dbt_calls.append(kwargs))
    monkeypatch.setattr("src.flows.etl._dbt_model_rows", lambda: {})
    monkeypatch.setattr("src.flows.etl._dbt_summary", lambda _log: "Done. PASS=1")
    monkeypatch.setattr(
        "src.flows.etl._incremental_watermarks",
        lambda: (datetime(2026, 1, 2), datetime(2026, 1, 1)),
    )
    monkeypatch.setattr("src.flows.etl._record_incremental_watermark", lambda _watermark: None)

    conn = MagicMock()
    direct_conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    conn.__enter__.return_value = direct_conn
    direct_conn.cursor.return_value.__enter__.return_value = cursor
    monkeypatch.setattr("src.flows.etl.psycopg.connect", lambda *_args, **_kwargs: conn)

    etl.bronze_to_gold.fn(incremental=False)
    etl.bronze_to_gold.fn(incremental=True)

    assert [call["incremental"] for call in dbt_calls] == [False, True]


def test_bronze_to_gold_skips_expensive_models_without_new_matches(monkeypatch):
    from unittest.mock import MagicMock

    dbt_calls = []
    monkeypatch.setattr("src.flows.etl.run_dbt_build", lambda **kwargs: dbt_calls.append(kwargs))
    monkeypatch.setattr("src.flows.etl._dbt_model_rows", lambda: {})
    monkeypatch.setattr("src.flows.etl._dbt_summary", lambda _log: "Done. PASS=1")
    monkeypatch.setattr(
        "src.flows.etl._incremental_watermarks",
        lambda: (datetime(2026, 1, 1), datetime(2026, 1, 1)),
    )

    conn = MagicMock()
    direct_conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    conn.__enter__.return_value = direct_conn
    direct_conn.cursor.return_value.__enter__.return_value = cursor
    monkeypatch.setattr("src.flows.etl.psycopg.connect", lambda *_args, **_kwargs: conn)

    etl.bronze_to_gold.fn(incremental=True)

    assert dbt_calls[0]["select"] == ["player_profiles"]
