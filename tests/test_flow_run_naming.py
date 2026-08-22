"""Hermetic tests for scrape/ETL flow-run naming and source-tagged automation.

No Prefect server, database, or network: the naming logic is pure, the
``flow_run_name`` callables are exercised by faking the runtime parameter
lookup, and the automations are built with only their trigger/action wiring.
"""

from datetime import date
from typing import cast
from uuid import uuid4

import pytest
from prefect.events.actions import RunDeployment
from prefect.events.schemas.automations import EventTrigger
from prefect.events.schemas.events import Resource, ResourceSpecification

import src.flows.etl as etl
import src.flows.matches as matches
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


# ── flow_run_name callables read runtime parameters (no server) ──


def test_scrape_flow_run_name_callable_reads_params(monkeypatch):
    monkeypatch.setattr(
        "prefect.runtime.flow_run.get_parameters",
        lambda: {"start_date": date(2024, 1, 1), "end_date": date(2024, 2, 1)},
    )
    assert rankings._scrape_flow_run_name() == "scrape-2024-01-01-2024-02-01"
    assert matches._scrape_flow_run_name() == "scrape-2024-01-01-2024-02-01"


def test_scrape_flow_run_name_callable_defaults_to_latest(monkeypatch):
    monkeypatch.setattr("prefect.runtime.flow_run.get_parameters", lambda: {})
    assert rankings._scrape_flow_run_name() == "scrape-latest"


def test_etl_flow_run_name_callable_reads_source(monkeypatch):
    monkeypatch.setattr(
        "prefect.runtime.flow_run.get_parameters",
        lambda: {"source": "matches", "incremental": True},
    )
    assert etl._etl_flow_run_name() == "etl-matches"


def test_etl_flow_run_name_callable_manual_when_no_source(monkeypatch):
    monkeypatch.setattr("prefect.runtime.flow_run.get_parameters", lambda: {"incremental": True})
    assert etl._etl_flow_run_name() == "etl-manual"


# ── ETL flow validates the source parameter (hermetic, no DB) ────


def test_etl_flow_rejects_invalid_source():
    with pytest.raises(ValueError, match="source"):
        etl.etl_flow.fn(source="drift")


def test_etl_flow_accepts_known_sources_and_none(monkeypatch):
    # Patch the body so the guard is the only thing exercised (no DB/work).
    monkeypatch.setattr(etl, "load_env", lambda: None)
    monkeypatch.setattr(etl, "bronze_to_gold", lambda **_kwargs: 0)
    for source in ("rankings", "matches", None):
        etl.etl_flow.fn(source=source)  # must not raise


# ── Source-tagged automations (one per scrape flow) ─────────────


def test_scrape_etl_automations_are_per_source():
    deployment_id = uuid4()
    for source, flow_name in (
        ("rankings", etl.RANKINGS_FLOW_NAME),
        ("matches", etl.MATCHES_FLOW_NAME),
    ):
        automation = etl.build_scrape_etl_automation(source, deployment_id)
        assert automation.name == f"{etl.SCRAPE_ETL_AUTOMATION_NAME}-{source}"

        trigger = cast(EventTrigger, automation.trigger)
        assert trigger.expect == {"prefect.flow-run.Completed"}
        match = cast(ResourceSpecification, trigger.match)
        assert match.root == {"prefect.resource.id": "prefect.flow-run.*"}
        match_related = cast(ResourceSpecification, trigger.match_related)
        assert match_related.root == {
            "prefect.resource.role": "flow",
            "prefect.resource.name": [flow_name],
        }

        assert len(automation.actions) == 1
        action = cast(RunDeployment, automation.actions[0])
        assert action.deployment_id == deployment_id
        # Action parameters replace (not merge with) deployment defaults, so
        # incremental must be passed explicitly alongside the source.
        assert action.parameters == {"source": source, "incremental": True}
        assert action.source == "selected"


def test_scrape_etl_automation_rejects_unknown_source():
    with pytest.raises(ValueError):
        etl.build_scrape_etl_automation("drift", uuid4())


def test_scrape_etl_automations_do_not_cross_match():
    """A rankings automation must not fire on a matches completion, and vice versa."""
    deployment_id = uuid4()
    rankings_auto = etl.build_scrape_etl_automation("rankings", deployment_id)
    matches_auto = etl.build_scrape_etl_automation("matches", deployment_id)

    r_trigger = cast(EventTrigger, rankings_auto.trigger)
    r_match_related = cast(ResourceSpecification, r_trigger.match_related)
    assert r_match_related.matches(
        cast(
            Resource,
            {"prefect.resource.role": "flow", "prefect.resource.name": etl.RANKINGS_FLOW_NAME},
        )
    )
    assert not r_match_related.matches(
        cast(
            Resource,
            {"prefect.resource.role": "flow", "prefect.resource.name": etl.MATCHES_FLOW_NAME},
        )
    )
    m_trigger = cast(EventTrigger, matches_auto.trigger)
    m_match_related = cast(ResourceSpecification, m_trigger.match_related)
    assert m_match_related.matches(
        cast(
            Resource,
            {"prefect.resource.role": "flow", "prefect.resource.name": etl.MATCHES_FLOW_NAME},
        )
    )
    assert not m_match_related.matches(
        cast(
            Resource,
            {"prefect.resource.role": "flow", "prefect.resource.name": etl.RANKINGS_FLOW_NAME},
        )
    )

    # Both still exclude unrelated flows and non-flow-run primary resources.
    for auto in (rankings_auto, matches_auto):
        trigger = cast(EventTrigger, auto.trigger)
        primary_match = cast(ResourceSpecification, trigger.match)
        assert primary_match.matches(
            cast(Resource, {"prefect.resource.id": "prefect.flow-run.<uuid>"})
        )
        assert not primary_match.matches(
            cast(Resource, {"prefect.resource.id": "prefect.task-run.<uuid>"})
        )
        assert not cast(ResourceSpecification, trigger.match_related).matches(
            cast(Resource, {"prefect.resource.role": "flow", "prefect.resource.name": "drift-flow"})
        )
