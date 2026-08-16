"""Offline drift-monitoring tests with mocked MLflow client and HTTP."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.flows import drift
from src.serving.service import PredictFromIdsRow


class _FakeModelVersion:
    def __init__(self, version="3", run_id="champ-run-id", creation_timestamp=1710000000000):
        self.version = version
        self.run_id = run_id
        self.creation_timestamp = creation_timestamp


class _FakeExperiment:
    def __init__(self, experiment_id="exp-1"):
        self.experiment_id = experiment_id


class _FakeMlflowClient:
    def __init__(self, champion=None, tags=None):
        self._champion = champion
        self._tags = tags or {}
        self.logged_params: dict[str, dict[str, object]] = {}
        self.logged_metrics: dict[str, dict[str, float]] = {}
        self.logged_texts: list[tuple[str, str, str]] = []
        self.logged_artifacts: list[tuple[str, str]] = []

    def get_model_version_by_alias(self, name, alias):
        assert name == "ensemble_lr_model"
        assert alias == "champion"
        if self._champion is None:
            from mlflow.exceptions import MlflowException

            raise MlflowException("Alias 'champion' not found")
        return self._champion

    def get_model_version(self, name, version):
        del name, version
        return SimpleNamespace(tags=self._tags)

    def get_experiment_by_name(self, _name):
        return _FakeExperiment()

    def log_param(self, run_id, key, value):
        self.logged_params.setdefault(run_id, {})[key] = value

    def log_metric(self, run_id, key, value):
        self.logged_metrics.setdefault(run_id, {})[key] = value

    def log_text(self, run_id, text, artifact_file):
        self.logged_texts.append((run_id, text, artifact_file))

    def log_artifact(self, run_id, local_path):
        self.logged_artifacts.append((run_id, local_path))


def _stub_batch_response(ctxs, probs=None):
    if probs is None:
        probs = [0.75 if i % 2 == 0 else 0.35 for i in range(len(ctxs))]
    return [
        {"player_id": c["player_id"], "opponent_id": c["opponent_id"], "p_win": probs[i]}
        for i, c in enumerate(ctxs)
    ]


def _validated_batch_rows(payload):
    rows = payload.get("rows") if isinstance(payload, dict) else None
    assert isinstance(rows, list)
    validated = [PredictFromIdsRow.model_validate(row) for row in rows]
    assert [row.model_dump(mode="json") for row in validated] == rows
    return rows


def _setup_model_info_stub(monkeypatch, mode="production", version="3", run_id="champ-run-id"):
    monkeypatch.setattr(drift, "BENTO_API_KEY", "")
    monkeypatch.setattr(drift, "PRODUCTION_BENTO_URL", "http://127.0.0.1:8187")
    monkeypatch.setattr(drift, "MODEL_INFO_ROUTE", "/api/model_info")
    fake_model_info = {
        "ok": True,
        "data": {
            "mode": mode,
            "manifest": {
                "champion": {
                    "registered_model_name": "ensemble_lr_model",
                    "version": version,
                    "run_id": run_id,
                }
            },
            "database": {"server_address": None, "server_port": None, "database_name": None},
        },
    }
    fake_resp = MagicMock()
    fake_resp.json.return_value = fake_model_info
    monkeypatch.setattr(drift.requests, "get", lambda _url, **__kwargs: fake_resp)


def test_no_champion_fails():
    client = _FakeMlflowClient(champion=None)
    with pytest.raises(RuntimeError, match="no champion found"):
        drift._validate_production(client)  # type: ignore[arg-type]


def test_champion_cutoff_prefers_training_data_tag():
    from datetime import date

    champion = _FakeModelVersion(
        version="3", run_id="champ-run-id", creation_timestamp=1700000000000
    )
    client = _FakeMlflowClient(
        champion=champion, tags={drift.TRAIN_DATA_MAX_DATE_KEY: "2025-01-10"}
    )
    assert drift._champion_cutoff_date(client) == date(2025, 1, 10)  # type: ignore[arg-type]


def test_champion_cutoff_falls_back_to_creation():
    from datetime import UTC, datetime

    champion = _FakeModelVersion(
        version="3", run_id="champ-run-id", creation_timestamp=1700000000000
    )
    client = _FakeMlflowClient(champion=champion, tags={})
    expected = datetime.fromtimestamp(1700000000000 / 1000, tz=UTC).date()
    assert drift._champion_cutoff_date(client) == expected  # type: ignore[arg-type]


def test_development_bento_is_valid_for_drift(monkeypatch):
    _setup_model_info_stub(monkeypatch, mode="development", version="2", run_id="other-run")

    client = _FakeMlflowClient(champion=_FakeModelVersion(version="3", run_id="champ-run-id"))

    champion = drift._validate_production(client)  # type: ignore[arg-type]
    assert champion.version == "3"


def test_empty_population_insufficient_data(monkeypatch, tmp_path):
    monkeypatch.setattr(drift, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(drift, "run_dbt_build", lambda **__kwargs: None)
    monkeypatch.setattr(drift, "load_env", lambda: None)
    _setup_model_info_stub(monkeypatch)

    champion = _FakeModelVersion(
        version="3", run_id="champ-run-id", creation_timestamp=1700000000000
    )
    client = _FakeMlflowClient(champion=champion)
    monkeypatch.setattr(drift, "MlflowClient", lambda: client)

    monkeypatch.setattr(drift, "execute_df", lambda _sql, _params=None: pd.DataFrame())

    mlflow_runs = []

    def fake_start_run(experiment_id=None, run_name=None, tags=None, log_system_metrics=False):
        del experiment_id, log_system_metrics
        run_id = f"run-{len(mlflow_runs)}"
        mlflow_runs.append({"name": run_name, "tags": tags})
        return MagicMock(
            info=SimpleNamespace(run_id=run_id), __enter__=MagicMock(), __exit__=MagicMock()
        )

    monkeypatch.setattr(drift.mlflow, "start_run", fake_start_run)

    result = drift.drift_flow.fn()
    assert result == 0
    assert any(
        r.get("tags") and r["tags"].get("status") == "insufficient_data" for r in mlflow_runs
    )


def test_small_population_insufficient_data(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(drift, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(drift, "load_env", lambda: None)
    _setup_model_info_stub(monkeypatch)

    champion = _FakeModelVersion()
    client = _FakeMlflowClient(champion=champion)
    monkeypatch.setattr(drift, "MlflowClient", lambda: client)
    monkeypatch.setattr(
        drift, "execute_df", lambda _sql, _params=None: _fake_bronze_window(2, seed=7)
    )
    # Fewer than DRIFT_MIN_N_FOR_CHECK physical matches must short-circuit
    # before any scoring HTTP call or Evidently report is produced.
    monkeypatch.setattr(
        drift.requests,
        "post",
        lambda *_args, **_kwargs: pytest.fail("HTTP scoring call for insufficient data"),
    )

    mlflow_runs = []

    def fake_start_run(experiment_id=None, run_name=None, tags=None, log_system_metrics=False):
        del experiment_id, log_system_metrics
        mlflow_runs.append({"name": run_name, "tags": tags})
        run = MagicMock(info=SimpleNamespace(run_id="small-window"))
        run.__enter__.return_value = run
        run.__exit__.return_value = False
        return run

    monkeypatch.setattr(drift.mlflow, "start_run", fake_start_run)

    # An explicit cutoff replaces the champion watermark even on the
    # insufficient-data path: it is what selects the (empty) current window,
    # what gets logged, and what the message reports.
    assert drift.drift_flow.fn(cutoff=date(2024, 11, 30)) == 0
    assert mlflow_runs[0]["tags"]["status"] == "insufficient_data"
    assert client.logged_params["small-window"]["cutoff_date"] == "2024-11-30"
    output = capsys.readouterr().out
    assert "2024-11-30" in output
    assert "insufficient_data" in output
    assert not list(tmp_path.glob("drift_report_*"))


def _champion_tags(pinned_metrics: dict[str, float]) -> dict[str, str]:
    """Champion model-version tags: training-data watermark + pinned metric tags."""
    tags = {
        drift.TRAIN_DATA_MAX_DATE_KEY: "2025-01-10",
        drift.EVAL_SPLIT_SIZE_KEY: "125",
        drift.EVAL_MAX_DATE_KEY: "2025-01-10",
    }
    tags.update(
        {f"{drift.METRIC_PREFIX}{name}": str(value) for name, value in pinned_metrics.items()}
    )
    return tags


_PINNED_METRICS = {
    "log_loss": 0.62,
    "roc_auc": 0.72,
    "accuracy": 0.63,
    "brier": 0.19,
}


def _fake_bronze_window(
    n: int,
    *,
    seed: int,
    surface: str = "hard",
    tournament: str = "atp_500",
    round_: str = "r64",
    indoor: int | None = None,
    rate_shift: float = 0.0,
) -> pd.DataFrame:
    """A bronze.match_events window: one row per physical match.

    Context values are the normalized public strings ingest writes. Winner is
    player1 by bronze convention (schema CHECK winner_id = player1_id). The
    match-stat rates mirror the drift SQL projection's output columns; a
    nonzero rate_shift moves every rate so two windows can be pushed apart for
    drift-verdict tests.
    """
    rng = np.random.default_rng(seed)
    is_indoor = [indoor] * n if indoor is not None else [int(x) for x in rng.integers(0, 2, n)]
    base_rates = {
        "player1_first_serve_pct": 0.62,
        "player1_serve_win_pct": 0.64,
        "player1_ace_rate": 0.08,
        "player1_df_rate": 0.06,
        "player2_first_serve_pct": 0.60,
        "player2_serve_win_pct": 0.58,
        "player2_ace_rate": 0.07,
        "player2_df_rate": 0.08,
    }
    rates = {
        col: np.clip(base + rng.uniform(-0.02, 0.02, n) + rate_shift, 0.0, 1.0)
        for col, base in base_rates.items()
    }
    return pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(n)],
            "match_date": pd.Timestamp("2025-01-15"),
            "player1_id": [str(i + 100) for i in range(n)],
            "player2_id": [str(i + 200) for i in range(n)],
            "winner_id": [str(i + 100) for i in range(n)],
            "surface": [surface] * n,
            "tournament": [tournament] * n,
            "round": [round_] * n,
            "is_indoor": is_indoor,
            **rates,
        }
    )


def test_five_bronze_matches_expand_to_ten_symmetric_scored_rows(monkeypatch):
    frame = _fake_bronze_window(5, seed=1)
    # Every second match is won by player2 to prove labels follow winner_id
    # equality, not bronze row position.
    frame.loc[1::2, "winner_id"] = frame.loc[1::2, "player2_id"].to_numpy()

    expanded = drift._expand_orientations(frame)
    assert len(expanded) == 10
    # Two adjacent rows per physical match, bronze player order preserved.
    assert list(expanded["match_id"]) == [mid for mid in frame["match_id"] for _ in range(2)]
    expected_players: list[str] = []
    expected_opponents: list[str] = []
    for p1, p2 in zip(frame["player1_id"], frame["player2_id"], strict=True):
        expected_players.extend([p1, p2])
        expected_opponents.extend([p2, p1])
    assert list(expanded["player_id"]) == expected_players
    assert list(expanded["opponent_id"]) == expected_opponents

    # Labels agree with the requested orientation and are complementary per
    # match: five physical matches yield ten rows with five wins and five losses.
    expected_labels: list[int] = []
    for i in range(len(frame)):
        winner = frame["winner_id"].iloc[i]
        expected_labels.extend(
            [
                int(winner == frame["player1_id"].iloc[i]),
                int(winner == frame["player2_id"].iloc[i]),
            ]
        )
    assert expanded["match_won"].tolist() == expected_labels
    assert set(expanded["match_won"]) == {0, 1}
    assert expanded.groupby("match_id")["match_won"].sum().tolist() == [1] * 5

    # Match-level numeric rates are identical across both orientations of each
    # physical match, and every rate column is numeric.
    for col in drift.DRIFT_FEATURE_COLS:
        assert expanded.groupby("match_id")[col].nunique().tolist() == [1] * 5
    assert set(expanded[drift.DRIFT_FEATURE_COLS].dtypes) <= {np.dtype("float64")}

    # Each observation round-trips the public bulk schema unchanged and never
    # carries internal encodings.
    contexts = drift._observation_contexts(expanded)
    assert len(contexts) == 10
    for context in contexts:
        assert "tournament_level" not in context
        assert "round_encoded" not in context
    validated = [PredictFromIdsRow.model_validate(context) for context in contexts]
    assert [row.model_dump(mode="json") for row in validated] == contexts
    # The drift boundary validates every context against the real request model.
    boundary = drift._validated_contexts(contexts)
    assert boundary == contexts
    assert set(boundary[0]) == {
        "player_id",
        "opponent_id",
        "surface",
        "as_of_date",
        "tournament",
        "round",
        "is_indoor",
    }

    # Scored rows: ten symmetric observations with p_win and the derived labels.
    # The analysis frame is exactly the numeric match-stat rates, the
    # orientation label, and the prediction — unique column names, no
    # categorical context or profile attributes.
    monkeypatch.setattr(drift, "_score_batches", lambda _ctxs: [0.9, 0.1] * 5)
    scored = drift._score_window(expanded)
    assert len(scored) == 10
    assert list(scored.columns) == drift.DRIFT_ANALYSIS_COLUMNS
    assert len(scored.columns) == len(set(scored.columns))
    assert set(scored.columns) == set(drift.DRIFT_ANALYSIS_COLUMNS)
    assert scored["p_win"].tolist() == [0.9, 0.1] * 5
    assert scored["match_won"].tolist() == expanded["match_won"].tolist()
    # No categorical match-context column enters the analysis frame.
    assert not {"surface", "tournament", "round", "is_indoor"} & set(scored.columns)


def _symmetric_batch_response(ctxs):
    """Order-preserving symmetric bulk response: forward p, then 1 - p."""
    return [
        {
            "player_id": c["player_id"],
            "opponent_id": c["opponent_id"],
            "p_win": 0.9 if i % 2 == 0 else 0.1,
        }
        for i, c in enumerate(ctxs)
    ]


def test_symmetric_orientation_invariants_and_accuracy_count_both_rows(monkeypatch):
    """One physical match (player1 wins) through the real scoring boundary.

    Each physical match yields two symmetric drift observations: forward
    player1→player2 with label y and reversed player2→player1 with label 1-y.
    The mocked bulk response is order-preserving and symmetric (p_reverse ==
    1 - p_forward), so the accuracy must count both correct predictions —
    a pair that collapses either orientation would land at 0.5, not 1.0.
    """
    frame = _fake_bronze_window(1, seed=1)  # winner_id == player1_id ("100")
    expanded = drift._expand_orientations(frame)

    # 1. Reversed request swaps player/opponent, bronze order preserved.
    assert expanded["player_id"].tolist() == ["100", "200"]
    assert expanded["opponent_id"].tolist() == ["200", "100"]

    # 2. Reversed label is 1 - forward label.
    assert expanded["match_won"].tolist() == [1, 0]

    contexts = drift._validated_contexts(drift._observation_contexts(expanded))
    assert contexts[0]["player_id"] == "100"
    assert contexts[0]["opponent_id"] == "200"
    assert contexts[1]["player_id"] == "200"
    assert contexts[1]["opponent_id"] == "100"

    monkeypatch.setattr(drift, "_post_batch", _symmetric_batch_response)
    probas = drift._score_batches(contexts)

    # 3. Returned p_win stays associated with the corresponding request row.
    assert probas == [0.9, 0.1]

    # 4. Symmetric service prediction: p_reverse == 1 - p_forward.
    assert probas[1] == pytest.approx(1.0 - probas[0])

    scored = drift._score_window(expanded)
    assert scored["p_win"].tolist() == [0.9, 0.1]
    assert scored["match_won"].tolist() == [1, 0]

    # 5. Accuracy counts both correct predictions, not one per match.
    metrics = drift._window_metrics(scored)
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_valid_winner_orientations_are_complementary_and_balanced():
    """A valid A/B winner yields labels [1,0] or [0,1] and an exactly 50/50 batch."""
    frame = _fake_bronze_window(4, seed=5)
    # Alternate the winner between A (player1) and B (player2) so both
    # orientation label patterns appear: [1,0] for A, [0,1] for B.
    frame.loc[1::2, "winner_id"] = frame.loc[1::2, "player2_id"].to_numpy()

    expanded = drift._expand_orientations(frame)
    assert len(expanded) == 8
    assert expanded["match_won"].tolist() == [1, 0, 0, 1, 1, 0, 0, 1]
    # Exactly one 1 and one 0 per physical match: labels are non-empty,
    # balanced 50/50, and every adjacent pair sums to 1.
    assert expanded["match_won"].sum() == len(expanded) // 2
    assert expanded["match_won"].mean() == 0.5
    assert expanded.groupby("match_id")["match_won"].sum().tolist() == [1] * 4


def test_invalid_winner_rejected_at_bronze_boundary():
    """Missing, unidentifiable, and ambiguous winner ids fail before expansion."""
    base = _fake_bronze_window(3, seed=6)

    # Missing winner.
    bad = base.copy()
    bad.loc[0, "winner_id"] = None
    with pytest.raises(ValueError, match="missing winner_id"):
        drift._expand_orientations(bad)

    # Winner equal to neither player.
    bad = base.copy()
    bad.loc[1, "winner_id"] = "999"
    with pytest.raises(ValueError, match="neither player"):
        drift._expand_orientations(bad)

    # Ambiguous winner: both sides share one id, so winner_id equals both.
    bad = base.copy()
    bad.loc[2, "player2_id"] = bad.loc[2, "player1_id"]
    with pytest.raises(ValueError, match="both players"):
        drift._expand_orientations(bad)


def test_invalid_player_ids_rejected_at_bronze_boundary():
    """A physical match with a missing player id is rejected before expansion."""
    base = _fake_bronze_window(2, seed=10)

    bad = base.copy()
    bad.loc[0, "player1_id"] = None
    with pytest.raises(ValueError, match="missing player id"):
        drift._expand_orientations(bad)

    bad = base.copy()
    bad.loc[1, "player2_id"] = None
    with pytest.raises(ValueError, match="missing player id"):
        drift._expand_orientations(bad)


def test_corrupted_expanded_batch_rejected_before_scoring(monkeypatch):
    """A non-complementary, odd, or empty expanded batch fails before any API call.

    `_score_window` re-validates the expanded frame on entry, so a pre-expanded
    (or externally corrupted) batch is refused even though it did not come from
    `_expand_orientations`. Labels are never silently repaired.
    """
    frame = _fake_bronze_window(2, seed=7)  # winner == player1 for both matches
    expanded = drift._expand_orientations(frame)
    assert expanded["match_won"].tolist() == [1, 0, 1, 0]

    monkeypatch.setattr(
        drift, "_score_batches", lambda _ctxs: pytest.fail("scoring called on invalid batch")
    )

    # Non-complementary pair: rows 0-1 become [1, 1] instead of [1, 0].
    corrupted = expanded.copy()
    corrupted.loc[1, "match_won"] = 1
    with pytest.raises(ValueError, match="do not sum to 1"):
        drift._score_window(corrupted)

    # Odd row count breaks the adjacent-pair structure.
    with pytest.raises(ValueError, match="odd row count"):
        drift._score_window(expanded.iloc[:-1])

    # Empty batch: no labels at all.
    with pytest.raises(ValueError, match="expanded drift frame is empty"):
        drift._score_window(pd.DataFrame())


def test_all_tie_p_win_batch_rejected(monkeypatch):
    """An all-0.5 p_win batch is rejected; a lone tie among varying ones passes.

    A symmetric order-preserving model ties (p == 1 - p == 0.5) only when it
    has no signal on a match, so a window whose predictions are ALL exactly
    0.5 indicates a broken serving path and must fail before metrics. A single
    tie in a small sample is a legitimate edge and must not fail.
    """
    frame = _fake_bronze_window(2, seed=9)
    expanded = drift._expand_orientations(frame)

    monkeypatch.setattr(drift, "_score_batches", lambda _ctxs: [0.5, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="constant-tie"):
        drift._score_window(expanded)

    # One tie among varying predictions is fine.
    monkeypatch.setattr(drift, "_score_batches", lambda _ctxs: [0.9, 0.5, 0.5, 0.1])
    scored = drift._score_window(expanded)
    assert scored["p_win"].tolist() == [0.9, 0.5, 0.5, 0.1]
    assert scored["match_won"].tolist() == expanded["match_won"].tolist()


def test_scored_frame_validator_direct():
    """`_validate_scored_frame` rejects non-finite, out-of-range, and all-tie p_win.

    The validator is the fail-fast gate between the scored rows and every
    downstream metric, so it is exercised directly: each violation must raise
    with row-level diagnostics embedded, and a valid frame must pass.
    """
    base = pd.DataFrame(
        {
            "match_id": ["m0", "m0"],
            "player_id": ["100", "200"],
            "opponent_id": ["200", "100"],
            "match_won": [1, 0],
        }
    )

    # Non-finite p_win (NaN) must be rejected, with diagnostics attached.
    bad = base.copy()
    bad["p_win"] = [0.9, np.nan]
    with pytest.raises(ValueError, match="non-finite") as excinfo:
        drift._validate_scored_frame(bad)
    assert "match_id" in str(excinfo.value) and "p_win" in str(excinfo.value)

    # Out-of-range p_win.
    bad = base.copy()
    bad["p_win"] = [1.2, -0.1]
    with pytest.raises(ValueError, match="outside"):
        drift._validate_scored_frame(bad)

    # Constant-tie batch.
    bad = base.copy()
    bad["p_win"] = [0.5, 0.5]
    with pytest.raises(ValueError, match="constant-tie"):
        drift._validate_scored_frame(bad)

    # Missing traceability columns.
    bad = base.copy()
    bad["p_win"] = [0.9, 0.1]
    with pytest.raises(ValueError, match="missing required columns"):
        drift._validate_scored_frame(bad.drop(columns=["match_id"]))

    # A valid frame passes without raising.
    good = base.copy()
    good["p_win"] = [0.9, 0.1]
    drift._validate_scored_frame(good)


def test_alternating_winners_symmetric_probas_score_accuracy_1(monkeypatch):
    """Correct symmetric mock p_win yields accuracy 1.0 when predictions match labels.

    Predictions are orientation-accurate (0.9 for the winner, 0.1 for the
    loser across both [1,0] and [0,1] patterns), so both rows of every pair
    count as correct — a design that collapses either orientation would land
    at 0.5, not 1.0.
    """
    frame = _fake_bronze_window(4, seed=8)
    frame.loc[1::2, "winner_id"] = frame.loc[1::2, "player2_id"].to_numpy()
    expanded = drift._expand_orientations(frame)
    assert expanded["match_won"].tolist() == [1, 0, 0, 1, 1, 0, 0, 1]

    monkeypatch.setattr(
        drift, "_score_batches", lambda _ctxs: [0.9, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.9]
    )
    scored = drift._score_window(expanded)
    assert scored["p_win"].tolist() == [0.9, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.9]
    assert scored["match_won"].tolist() == [1, 0, 0, 1, 1, 0, 0, 1]
    assert drift._window_metrics(scored)["accuracy"] == pytest.approx(1.0)


def test_scrambled_bulk_response_rejected_instead_of_silent_05(monkeypatch):
    """A bulk response whose row order/identity cannot be trusted must fail.

    Regression: `_score_batches` used to check only the row count and read
    p_win positionally, so a reordered (e.g. sorted/grouped) response silently
    paired each probability with the wrong orientation label — collapsing the
    symmetric accuracy to exactly 0.5. The boundary must refuse to score.
    """
    frame = _fake_bronze_window(1, seed=1)  # winner_id == player1_id ("100")
    expanded = drift._expand_orientations(frame)
    contexts = drift._validated_contexts(drift._observation_contexts(expanded))

    # Response records carry the ids but in reversed order: row 0 answers the
    # second request row and vice versa. Count matches, identity does not.
    scrambled = list(reversed(_symmetric_batch_response(contexts)))
    assert [r["player_id"] for r in scrambled] == ["200", "100"]

    monkeypatch.setattr(drift, "_post_batch", lambda _chunk: scrambled)

    with pytest.raises(RuntimeError, match="mismatch"):
        drift._score_batches(contexts)


def test_unsupported_bronze_values_resolved_at_boundary(monkeypatch):
    """Round ``rr`` (and other out-of-schema bronze values) become omitted fields.

    dbt maps unsupported rounds/tournaments to ordinal 0 and the public schema
    expresses that as an omitted field; unknown surfaces use the schema's own
    ``"0"`` marker. The raw bronze value is preserved in the observation
    contexts while the posted payload carries the resolved public value; the
    analysis frame carries only numeric match-stat rates, never the context.
    """
    frame = _fake_bronze_window(5, seed=2, round_="rr", tournament="masters")
    frame.loc[1::2, "winner_id"] = frame.loc[1::2, "player2_id"].to_numpy()
    # One match with an unknown-tier tournament and an out-of-schema surface.
    frame.loc[0, "tournament"] = "hopman_cup"
    frame.loc[0, "surface"] = "turf"

    expanded = drift._expand_orientations(frame)
    contexts = drift._observation_contexts(expanded)

    # Raw bronze values stay in the observation contexts (unresolved); only the
    # boundary resolves them for the endpoint.
    assert any(c["round"] == "rr" for c in contexts)
    assert any(c["tournament"] == "hopman_cup" for c in contexts)
    assert any(c["surface"] == "turf" for c in contexts)

    boundary = drift._validated_contexts(contexts)
    assert len(boundary) == 10
    assert all(c["round"] is None for c in boundary)
    # Match 0's two orientations carry the out-of-schema values; the others
    # keep their valid bronze values.
    assert boundary[0]["tournament"] is None
    assert boundary[1]["tournament"] is None
    assert all(c["tournament"] == "masters" for c in boundary[2:])
    assert boundary[0]["surface"] == "0"
    assert boundary[1]["surface"] == "0"
    assert all(c["surface"] == "hard" for c in boundary[2:])

    # The resolved payload round-trips the real request model unchanged.
    validated = [PredictFromIdsRow.model_validate(context) for context in boundary]
    assert [row.model_dump(mode="json") for row in validated] == boundary

    # Scoring still aligns rows: one p_win per context, first-supplied side.
    monkeypatch.setattr(drift, "_score_batches", lambda _ctxs: [0.9, 0.1] * 5)
    scored = drift._score_window(expanded)
    assert len(scored) == 10
    assert scored["p_win"].tolist() == [0.9, 0.1] * 5
    assert scored["match_won"].tolist() == expanded["match_won"].tolist()
    # The analysis frame stays numeric-only: the out-of-schema round value is
    # resolved at the boundary and never becomes a drift column.
    assert "round" not in scored.columns
    assert not {"surface", "tournament", "round", "is_indoor"} & set(scored.columns)


def test_unresolvable_context_raises_at_boundary():
    """A context missing a required field fails loudly before any HTTP post."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        drift._validated_contexts([{"player_id": "100", "opponent_id": "200"}])

    # Effective result columns of the bronze projection (after the AS alias);
    # each tuple entry is a full `expr AS alias` string.
    columns = [token.split(" AS ")[-1] for token in drift._BRONZE_WINDOW_COLUMNS]

    assert len(columns) == len(set(columns))
    assert set(drift.DRIFT_FEATURE_COLS) <= set(columns)


def test_evidently_drift_extracts_numeric_match_stats_and_p_win_psi(tmp_path, monkeypatch):
    """The DataDriftPreset input/report hold exactly features + p_win, never labels.

    The frames carry the orientation label plus traceability/context columns
    (like the real scored frames), but Evidently receives only the eight
    numeric match-stat rates and the prediction column: `match_won`, ids, and
    context are excluded from the report, per the DataDriftPreset contract.
    The per-feature verdict covers the match-stat rates only; p_win is
    reported separately via the prediction PSI.
    """
    base_rates = {
        "player1_first_serve_pct": 0.65,
        "player1_serve_win_pct": 0.67,
        "player1_ace_rate": 0.09,
        "player1_df_rate": 0.06,
        "player2_first_serve_pct": 0.63,
        "player2_serve_win_pct": 0.61,
        "player2_ace_rate": 0.08,
        "player2_df_rate": 0.07,
    }
    current = pd.DataFrame(
        {col: [value] * 60 for col, value in base_rates.items()}
        | {
            "match_won": [1, 0] * 30,
            "p_win": [0.9, 0.1] * 30,
            "player_id": ["100", "200"] * 30,
            "opponent_id": ["200", "100"] * 30,
            "surface": ["hard"] * 60,
            "tournament": ["atp_500"] * 60,
            "round": ["r64"] * 60,
            "is_indoor": [0] * 60,
        }
    )
    reference = pd.DataFrame(
        {col: [0.3] * 60 for col in base_rates}
        | {
            "match_won": [1, 0] * 30,
            "p_win": [0.5] * 60,
            "player_id": ["100", "200"] * 30,
            "opponent_id": ["200", "100"] * 30,
            "surface": ["hard"] * 60,
            "tournament": ["atp_500"] * 60,
            "round": ["r64"] * 60,
            "is_indoor": [0] * 60,
        }
    )
    json_path = tmp_path / "report.json"
    html_path = tmp_path / "report.html"

    # Record the exact frames handed to the report while still running the
    # real Evidently report underneath.
    recorded: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    real_run = drift.Report.run

    def record_run(self, *args, **kwargs):
        if "current_data" in kwargs:
            recorded.append((kwargs["current_data"], kwargs["reference_data"]))
        return real_run(self, *args, **kwargs)

    monkeypatch.setattr(drift.Report, "run", record_run)

    per_feature, drift_share, prediction_psi = drift._evidently_drift(
        current, reference, json_path, html_path
    )

    # Evidently sees exactly the monitored feature columns plus p_win — the
    # label, ids, and context columns never reach the report.
    evidently_columns = [*drift.DRIFT_FEATURE_COLS, "p_win"]
    assert len(recorded) == 1
    for frame in recorded[0]:
        assert list(frame.columns) == evidently_columns

    # Per-feature PSI covers exactly the numeric match-stat rates, all of which
    # are disjoint between the windows. The balanced label and p_win are not
    # part of the per-feature verdict.
    assert set(per_feature) == set(drift.DRIFT_FEATURE_COLS)
    assert all(psi > 0 for psi in per_feature.values())
    assert "match_won" not in per_feature
    assert "p_win" not in per_feature
    assert drift_share >= drift.DRIFT_SHARE_THRESHOLD

    # Prediction PSI comes from the report's p_win column.
    assert prediction_psi > 0

    # The saved report covers exactly the monitored features plus p_win, each
    # once; the label and ids/context never enter the report.
    payload = json.loads(json_path.read_text())
    drifted_columns = {
        metric["metric_name"][len("ValueDrift(column=") :].split(",", 1)[0]
        for metric in payload["metrics"]
        if metric["metric_name"].startswith("ValueDrift(column=")
    }
    assert drifted_columns == set(evidently_columns)
    assert len(drifted_columns) == len(evidently_columns)
    assert not {"match_won", "surface", "tournament", "round", "is_indoor"} & drifted_columns
    assert html_path.exists()


def test_normal_flow_runs_evidently_and_logs_drift_check(monkeypatch, tmp_path):
    monkeypatch.setattr(drift, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(drift, "run_dbt_build", lambda **__kwargs: None)
    monkeypatch.setattr(drift, "load_env", lambda: None)
    _setup_model_info_stub(monkeypatch)

    champion = _FakeModelVersion(
        version="3", run_id="champ-run-id", creation_timestamp=1700000000000
    )
    client = _FakeMlflowClient(champion=champion, tags=_champion_tags(_PINNED_METRICS))
    monkeypatch.setattr(drift, "MlflowClient", lambda: client)

    current_frame = _fake_bronze_window(60, seed=1)
    reference_frame = _fake_bronze_window(60, seed=2)
    sql_calls: list[tuple[str, list[object]]] = []

    def fake_execute_df(sql, params=None):
        sql_calls.append((sql, params or []))
        if "match_date > %s" in sql:
            return current_frame
        if "match_date < %s" in sql:
            return reference_frame
        raise AssertionError(f"unexpected drift SQL: {sql}")

    monkeypatch.setattr(drift, "execute_df", fake_execute_df)

    posts: list[list[dict[str, object]]] = []

    def fake_post_batch(url, json=None, headers=None, timeout=None):
        del url, headers, timeout
        ctxs = _validated_batch_rows(json or {})
        posts.append(ctxs)
        fake_resp = MagicMock()
        fake_resp.raise_for_status = lambda: None
        fake_resp.json.return_value = _stub_batch_response(ctxs)
        return fake_resp

    monkeypatch.setattr(drift.requests, "post", fake_post_batch)

    mlflow_runs = []

    def fake_start_run(experiment_id=None, run_name=None, tags=None, log_system_metrics=False):
        del experiment_id, log_system_metrics
        run_id = f"run-{len(mlflow_runs)}-{run_name}"
        mlflow_runs.append({"name": run_name, "tags": tags})
        return MagicMock(
            info=SimpleNamespace(run_id=run_id), __enter__=MagicMock(), __exit__=MagicMock()
        )

    monkeypatch.setattr(drift.mlflow, "start_run", fake_start_run)

    result = drift.drift_flow.fn()
    assert result == 0

    assert [r["name"] for r in mlflow_runs] == ["drift_check"]
    check_run = mlflow_runs[0]
    recommendation = check_run["tags"]["recommendation"]
    assert recommendation in {"healthy", "investigate", "retrain"}
    assert check_run["tags"]["retrain_required"] == str(recommendation == "retrain")

    # Both windows pulled from bronze physical matches, reference size-matched
    # to the current count.
    assert len(sql_calls) == 2
    current_sql, reference_sql = sql_calls
    assert "FROM bronze.match_events" in current_sql[0]
    assert "FROM bronze.match_events" in reference_sql[0]
    assert "match_date > %s" in current_sql[0]
    assert "match_date < %s" in reference_sql[0]
    assert reference_sql[1] == [date(2025, 1, 10), 60]

    # Both windows scored through the production Bento, two orientations each.
    assert len(posts) == 2
    assert len(posts[0]) == 120
    assert len(posts[1]) == 120

    summary_text = [text for _, text, name in client.logged_texts if name == "drift_summary.json"]
    assert summary_text
    summary = json.loads(summary_text[0])
    assert summary["recommendation"] == recommendation
    assert summary["retrain_required"] == (recommendation == "retrain")
    assert {"drift_share", "prediction_psi", "calibration_delta", "per_feature_drift"} <= set(
        summary
    )

    assert sorted(Path(path).name for _, path in client.logged_artifacts) == [
        "drift_report_2025-01-10_v3.html",
        "drift_report_2025-01-10_v3.json",
    ]
    assert (tmp_path / "drift_report_2025-01-10_v3.json").exists()
    assert (tmp_path / "drift_report_2025-01-10_v3.html").exists()


def test_cutoff_override_replaces_champion_watermark(monkeypatch, tmp_path, capsys):
    """An explicit cutoff wins over the pinned TRAIN_DATA_MAX_DATE watermark.

    The champion tag says 2025-01-10, the override says 2024-12-01; the
    override must drive window selection SQL, the report artifact names, the
    logged params, and the summary/messaging. Post-cutoff training-era matches
    are intentionally treated as current data — the override is never
    constrained relative to the watermark.
    """
    monkeypatch.setattr(drift, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(drift, "run_dbt_build", lambda **__kwargs: None)
    monkeypatch.setattr(drift, "load_env", lambda: None)
    _setup_model_info_stub(monkeypatch)

    champion = _FakeModelVersion(
        version="3", run_id="champ-run-id", creation_timestamp=1700000000000
    )
    client = _FakeMlflowClient(champion=champion, tags=_champion_tags(_PINNED_METRICS))
    monkeypatch.setattr(drift, "MlflowClient", lambda: client)

    current_frame = _fake_bronze_window(60, seed=1)
    reference_frame = _fake_bronze_window(60, seed=2)
    sql_calls: list[tuple[str, list[object]]] = []

    def fake_execute_df(sql, params=None):
        sql_calls.append((sql, params or []))
        if "match_date > %s" in sql:
            return current_frame
        if "match_date < %s" in sql:
            return reference_frame
        raise AssertionError(f"unexpected drift SQL: {sql}")

    monkeypatch.setattr(drift, "execute_df", fake_execute_df)

    posts: list[list[dict[str, object]]] = []

    def fake_post_batch(url, json=None, headers=None, timeout=None):
        del url, headers, timeout
        ctxs = _validated_batch_rows(json or {})
        posts.append(ctxs)
        fake_resp = MagicMock()
        fake_resp.raise_for_status = lambda: None
        fake_resp.json.return_value = _stub_batch_response(ctxs)
        return fake_resp

    monkeypatch.setattr(drift.requests, "post", fake_post_batch)

    mlflow_runs = []

    def fake_start_run(experiment_id=None, run_name=None, tags=None, log_system_metrics=False):
        del experiment_id, log_system_metrics
        run_id = f"run-{len(mlflow_runs)}-{run_name}"
        mlflow_runs.append({"name": run_name, "tags": tags})
        run = MagicMock(info=SimpleNamespace(run_id=run_id))
        run.__enter__.return_value = run
        run.__exit__.return_value = False
        return run

    monkeypatch.setattr(drift.mlflow, "start_run", fake_start_run)

    override = date(2024, 12, 1)
    result = drift.drift_flow.fn(cutoff=override)
    assert result == 0
    assert mlflow_runs[0]["name"] == "drift_check"

    # The override — not the 2025-01-10 watermark tag — selects both windows.
    current_sql, reference_sql = sql_calls
    assert current_sql[1] == [override]
    assert reference_sql[1] == [override, 60]

    # Report artifacts and logged params/summary carry the override date.
    assert sorted(Path(path).name for _, path in client.logged_artifacts) == [
        "drift_report_2024-12-01_v3.html",
        "drift_report_2024-12-01_v3.json",
    ]
    assert client.logged_params["run-0-drift_check"]["cutoff_date"] == "2024-12-01"
    summary = json.loads(
        next(text for _, text, name in client.logged_texts if name == "drift_summary.json")
    )
    assert summary["cutoff_date"] == "2024-12-01"
    assert "Cutoff date (override): 2024-12-01" in capsys.readouterr().out


def test_match_stat_drift_triggers_retrain_verdict(monkeypatch, tmp_path):
    monkeypatch.setattr(drift, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(drift, "run_dbt_build", lambda **__kwargs: None)
    monkeypatch.setattr(drift, "load_env", lambda: None)
    _setup_model_info_stub(monkeypatch)

    champion = _FakeModelVersion(
        version="3", run_id="champ-run-id", creation_timestamp=1700000000000
    )
    client = _FakeMlflowClient(champion=champion, tags=_champion_tags(_PINNED_METRICS))
    monkeypatch.setattr(drift, "MlflowClient", lambda: client)

    # The symmetric label design pins calibration at 0.5 (each physical match
    # yields one win and one loss), so the retrain verdict must come from
    # Evidently drift: every numeric match-stat rate is pushed apart between
    # the windows (rate_shift +-0.3), and the scored p_win distributions
    # differ (0.9/0.1 vs 0.6/0.4).
    current_frame = _fake_bronze_window(60, seed=3, rate_shift=0.3)
    reference_frame = _fake_bronze_window(60, seed=4, rate_shift=-0.3)

    def fake_execute_df(sql, _params=None):
        if "match_date > %s" in sql:
            return current_frame
        if "match_date < %s" in sql:
            return reference_frame
        raise AssertionError(f"unexpected drift SQL: {sql}")

    monkeypatch.setattr(drift, "execute_df", fake_execute_df)

    posts: list[list[dict[str, object]]] = []

    def fake_post_batch(url, json=None, headers=None, timeout=None):
        del url, headers, timeout
        ctxs = _validated_batch_rows(json or {})
        probs = [0.9, 0.1] if not posts else [0.6, 0.4]
        posts.append(ctxs)
        fake_resp = MagicMock()
        fake_resp.raise_for_status = lambda: None
        fake_resp.json.return_value = _stub_batch_response(ctxs, probs=probs * (len(ctxs) // 2))
        return fake_resp

    monkeypatch.setattr(drift.requests, "post", fake_post_batch)

    mlflow_runs = []

    def fake_start_run(experiment_id=None, run_name=None, tags=None, log_system_metrics=False):
        del experiment_id, log_system_metrics
        run_id = f"run-{len(mlflow_runs)}-{run_name}"
        mlflow_runs.append({"name": run_name, "tags": tags})
        return MagicMock(
            info=SimpleNamespace(run_id=run_id), __enter__=MagicMock(), __exit__=MagicMock()
        )

    monkeypatch.setattr(drift.mlflow, "start_run", fake_start_run)

    result = drift.drift_flow.fn()
    assert result == 0

    check_run = mlflow_runs[0]
    assert check_run["name"] == "drift_check"
    assert check_run["tags"]["recommendation"] == "retrain"
    assert check_run["tags"]["retrain_required"] == "True"

    summary_text = next(
        text for _, text, name in client.logged_texts if name == "drift_summary.json"
    )
    summary = json.loads(summary_text)
    assert summary["recommendation"] == "retrain"
    assert summary["drift_share"] >= drift.DRIFT_SHARE_THRESHOLD
