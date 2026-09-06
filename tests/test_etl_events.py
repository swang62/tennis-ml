"""Hermetic tests for ETL events: scrape event payloads, deployment automation
wiring, and the profile_only override."""

from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
from prefect.events.actions import SendNotification
from prefect.events.schemas.automations import Posture

import src.flows.etl as etl
import src.flows.matches as matches
import src.flows.rankings as rankings

# ── Empty-scrapes automation ───────────────────────────────────────


def test_empty_scrapes_automation_notifies_over_both_streams(monkeypatch):
    """register_automation creates one automation covering both scrape streams;
    its notification action references the saved ntfy block."""
    block_id = uuid4()
    saved_blocks: list[tuple[Any, dict[str, Any]]] = []
    created: list[Any] = []

    def fake_save(self, **kwargs):
        saved_blocks.append((self, kwargs))
        return block_id

    class _DummyAutomation:
        def delete(self):
            pass

    monkeypatch.setenv("NTFY_URL", "https://ntfy.sh/hermetic-topic")
    monkeypatch.setattr(etl, "_prefect_runs_url", lambda: None)
    monkeypatch.setattr(etl.CustomWebhookNotificationBlock, "save", fake_save)
    monkeypatch.setattr(etl.Automation, "read", classmethod(lambda *_a, **_k: _DummyAutomation()))
    monkeypatch.setattr(etl.Automation, "create", lambda self: created.append(self))

    etl.register_automation()

    assert len(created) == 1
    trigger = created[0].trigger
    assert trigger.posture is Posture.Proactive
    assert trigger.within == timedelta(days=8)
    action = created[0].actions[0]
    assert isinstance(action, SendNotification)
    assert action.block_document_id == block_id


def test_ntfy_block_userinfo_and_path_sets_endpoint_topic_and_click(monkeypatch):
    """NTFY_URL with userinfo and path: the block posts to scheme+userinfo+netloc,
    the topic is the path segment, and json_data carries ntfy priority/tags plus a
    Prefect click link."""
    captured: dict[str, Any] = {}

    def fake_save(self, **_kwargs):
        captured["block"] = self
        return uuid4()

    class _DummyAutomation:
        def delete(self):
            pass

    monkeypatch.setenv("NTFY_URL", "https://coco:yoyo@notify.stronglybrewed.dev/alerts")
    monkeypatch.setattr(etl, "_prefect_runs_url", lambda: "https://prefect.stronglybrewed.dev/runs")
    monkeypatch.setattr(etl.CustomWebhookNotificationBlock, "save", fake_save)
    monkeypatch.setattr(etl.Automation, "read", classmethod(lambda *_a, **_k: _DummyAutomation()))
    monkeypatch.setattr(etl.Automation, "create", lambda _self: None)

    etl.register_automation()

    assert captured["block"].url == "https://coco:yoyo@notify.stronglybrewed.dev"
    assert captured["block"].json_data["topic"] == "alerts"
    assert captured["block"].json_data["click"] == "https://prefect.stronglybrewed.dev/runs"


def test_ntfy_unset_registers_no_block_or_automation(monkeypatch):
    """Unset NTFY_URL: no notification block is saved and no automation is
    created; legacy cleanup still runs."""
    saved: list[Any] = []
    created: list[Any] = []

    class _DummyAutomation:
        def delete(self):
            pass

    monkeypatch.delenv("NTFY_URL", raising=False)
    monkeypatch.setattr(
        etl.CustomWebhookNotificationBlock,
        "save",
        lambda *_a, **_k: saved.append(True) or uuid4(),
    )
    monkeypatch.setattr(
        etl.Automation,
        "read",
        classmethod(lambda _cls, _name=None, **_k: _DummyAutomation()),
    )
    monkeypatch.setattr(etl.Automation, "create", lambda self: created.append(self))

    etl.register_automation()

    assert saved == []
    assert created == []


# ── Scrape event payload builders ──────────────────────────────────


def test_emit_rankings_scraped_shapes_event_with_watermark(monkeypatch):
    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(rankings, "emit_event", lambda **kw: emitted.append(kw))

    rankings._emit_rankings_scraped(12, datetime(2026, 1, 5).date())

    assert emitted[0]["event"] == "rankings.scraped"
    assert emitted[0]["payload"] == {"row_count": 12, "watermark": "2026-01-05"}


def test_emit_rankings_scraped_omits_watermark_when_unknown(monkeypatch):
    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(rankings, "emit_event", lambda **kw: emitted.append(kw))

    rankings._emit_rankings_scraped(7, None)

    assert emitted[0]["payload"] == {"row_count": 7}


def test_emit_rankings_scraped_swallows_emit_failure(monkeypatch):
    def boom(**_kwargs):
        raise RuntimeError("emit down")

    monkeypatch.setattr(rankings, "emit_event", boom)

    rankings._emit_rankings_scraped(3, None)  # must not raise


def test_emit_matches_scraped_shapes_event_with_window(monkeypatch):
    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(matches, "emit_event", lambda **kw: emitted.append(kw))

    matches._emit_matches_scraped(5, datetime(2026, 1, 1).date(), datetime(2026, 1, 31).date())

    assert emitted[0]["event"] == "matches.scraped"
    assert emitted[0]["payload"] == {
        "row_count": 5,
        "window_start": "2026-01-01",
        "window_end": "2026-01-31",
    }


def test_emit_matches_scraped_swallows_emit_failure(monkeypatch):
    def boom(**_kwargs):
        raise RuntimeError("emit down")

    monkeypatch.setattr(matches, "emit_event", boom)

    matches._emit_matches_scraped(2, None, None)  # must not raise


# ── profile_only override ──────────────────────────────────────────


class _EloResult:
    processed = 7
    snapshots = 3


@pytest.fixture
def etl_boundaries(monkeypatch):
    """Stub the dbt and DB boundaries so bronze_to_gold runs hermetically."""
    dbt_calls: list[dict[str, Any]] = []
    watermarks: list[datetime | None] = []

    def record_dbt(**kwargs):
        dbt_calls.append(kwargs)

    monkeypatch.setattr(etl, "run_dbt_build", record_dbt)
    monkeypatch.setattr(etl, "clear_etl_state", lambda: None)
    monkeypatch.setattr(etl, "_report_phase", lambda *_a, **_k: None)
    monkeypatch.setattr(etl, "_current_gold_count", lambda: 42)
    monkeypatch.setattr(etl, "_elo_counts", lambda: {"matches": 7, "snapshots": 3})
    monkeypatch.setattr(etl, "materialize_elo", lambda: _EloResult())
    monkeypatch.setattr(etl, "_record_incremental_watermark", watermarks.append)
    return {"dbt_calls": dbt_calls, "watermarks": watermarks}


def test_forced_profile_only_runs_only_player_profiles(etl_boundaries, monkeypatch):
    monkeypatch.setattr(etl, "_incremental_watermarks", lambda: (None, None))

    assert etl.bronze_to_gold.fn(incremental=True, profile_only=True) == (42, True)

    assert len(etl_boundaries["dbt_calls"]) == 1
    call = etl_boundaries["dbt_calls"][0]
    assert call["select"] == ["player_profiles"]
    assert call["subcommand"] == "run"
    assert call["incremental"] is True
    assert etl_boundaries["watermarks"] == []  # profiles-only never advances the watermark


def test_auto_derived_profile_only_when_no_new_matches(etl_boundaries, monkeypatch):
    source = datetime(2026, 1, 5)
    monkeypatch.setattr(etl, "_incremental_watermarks", lambda: (source, source))

    assert etl.bronze_to_gold.fn(incremental=True, profile_only=False) == (42, True)

    assert len(etl_boundaries["dbt_calls"]) == 1
    assert etl_boundaries["dbt_calls"][0]["select"] == ["player_profiles"]


def test_forced_profile_only_without_incremental_runs_full_refresh(etl_boundaries, monkeypatch):
    source = datetime(2026, 1, 5)
    monkeypatch.setattr(etl, "_incremental_watermarks", lambda: (source, None))

    assert etl.bronze_to_gold.fn(incremental=False, profile_only=True) == (42, False)

    calls = etl_boundaries["dbt_calls"]
    assert [call["select"] for call in calls] == [
        etl.BASE_PHASE_MODELS,
        etl.FINAL_PHASE_MODELS,
        ["test_type:data"],
    ]
    assert [call["subcommand"] for call in calls] == ["run", "run", "test"]
    assert etl_boundaries["watermarks"] == [source]  # watermark advances only after full success


def test_incremental_with_new_matches_runs_full_phases(etl_boundaries, monkeypatch):
    source = datetime(2026, 1, 5)
    built = datetime(2025, 12, 1)
    monkeypatch.setattr(etl, "_incremental_watermarks", lambda: (source, built))

    assert etl.bronze_to_gold.fn(incremental=True, profile_only=False) == (42, False)

    assert len(etl_boundaries["dbt_calls"]) == 3  # base, final, tests — no profiles-only branch
    assert etl_boundaries["watermarks"] == [source]
