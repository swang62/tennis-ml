"""Local drift transformation, validation, and report tests."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.flows import drift
from src.serving.service import PredictFromIdsRow


def _fake_bronze_window(
    n: int,
    *,
    seed: int,
    surface: str = "hard",
    tournament: str = "atp_500",
    round_: str = "r64",
    indoor: int | None = None,
    best_of: int = 3,
    rate_shift: float = 0.0,
) -> pd.DataFrame:
    """Return one bronze-shaped row per physical match."""
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
            "best_of": [best_of] * n,
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
        "best_of",
        "is_indoor",
    }

    # Scored rows contain numeric rates, orientation labels, and predictions only.
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
    """Score one physical match through both symmetric orientations."""
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
    """Invalid expanded batches fail before any API call."""
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
    """An all-0.5 prediction batch is rejected, but an isolated tie is valid."""
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
    """Scored-frame validation rejects invalid probabilities with diagnostics."""
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
    """Symmetric orientation-aware predictions score every row correctly."""
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
    """A bulk response with reordered identities fails validation."""
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
    """Normalize out-of-schema context values before scoring and analysis."""
    frame = _fake_bronze_window(5, seed=2, round_="rr", tournament="masters")
    frame.loc[1::2, "winner_id"] = frame.loc[1::2, "player2_id"].to_numpy()
    # One match with an unknown-tier tournament and an out-of-schema surface,
    # one match with the legacy "0" surface marker.
    frame.loc[0, "tournament"] = "hopman_cup"
    frame.loc[0, "surface"] = "turf"
    frame.loc[1, "surface"] = "0"

    expanded = drift._expand_orientations(frame)
    contexts = drift._observation_contexts(expanded)

    # Raw bronze values stay in the observation contexts (unresolved); only the
    # boundary resolves them for the endpoint.
    assert any(c["round"] == "rr" for c in contexts)
    assert any(c["tournament"] == "hopman_cup" for c in contexts)
    assert any(c["surface"] == "turf" for c in contexts)
    assert any(c["surface"] == "0" for c in contexts)

    boundary = drift._validated_contexts(contexts)
    assert len(boundary) == 10
    assert all(c["round"] is None for c in boundary)
    # Matches 0/1 carry the out-of-schema values; the others keep their valid
    # bronze values. Both unresolvable surfaces normalize to hard.
    assert boundary[0]["tournament"] is None
    assert boundary[1]["tournament"] is None
    assert all(c["tournament"] == "masters" for c in boundary[2:])
    assert all(c["surface"] == "hard" for c in boundary[:4])
    assert all(c["surface"] == "hard" for c in boundary[4:])

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
    """Evidently receives only match-stat features and p_win."""
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

    # Per-feature PSI covers disjoint numeric rates, not the balanced label or p_win.
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
