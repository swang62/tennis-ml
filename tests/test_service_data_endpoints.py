"""Contract tests for /directory, /rank_history, /match_history, and /head_to_head."""

import logging
from datetime import date
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from starlette.testclient import TestClient

from src.constants import STACK_ORDER
from src.features import nn_inference as ni
from src.features.columns import FEATURE_COLS, TOUR_AVERAGES_FALLBACK_COLS
from src.serving.service import (
    DATA_APP,
    BestOf,
    PredictFromIdsRow,
    Round,
    Surface,
    TennisPredictor,
    TournamentLevel,
    _SuppressRequestValidationTraceback,
)

client = TestClient(DATA_APP)


def _directory_players_df() -> pd.DataFrame:
    """Shaped like the PLAYERS_SQL result (bronze profile joined to gold)."""
    return pd.DataFrame(
        [
            {
                "player_id": "p1",
                "display_name": "A Player",
                "ioc": "ESP",
                "backhand": "1H",
                "handedness": "R",
                "summary": "s1",
                "matches_played": np.int64(20),
                "current_rank": np.int64(1),
            },
            {
                "player_id": "p2",
                "display_name": "B Player",
                "ioc": "ARG",
                "backhand": "2H",
                "handedness": "L",
                "summary": "s2",
                "matches_played": np.int64(0),
                "current_rank": None,
            },
        ]
    )


def _directory_fake_execute_df(
    players_df: pd.DataFrame,
    summary_df: pd.DataFrame | None = None,
) -> object:
    """Keyed on the SQL text: PLAYERS_SQL -> players rows, else the summary."""

    def fake(sql: str, _params: list[object] | None = None) -> pd.DataFrame:
        return (
            players_df
            if "player_profiles" in sql
            else summary_df
            if summary_df is not None
            else _directory_summary_df()
        )

    return fake


def _directory_summary_df() -> pd.DataFrame:
    return pd.DataFrame({"latest_match_date": [date(2026, 8, 10)], "total_matches": [123456]})


def test_directory_returns_players_and_summary_in_one_envelope():
    fake = _directory_fake_execute_df(_directory_players_df())
    with patch("src.serving.service.execute_df", side_effect=fake):
        resp = client.get("/directory")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    data = resp.json()["data"]
    assert [p["player_id"] for p in data["players"]] == ["p1", "p2"]  # SQL row order
    assert data["players"][0]["iso2"] == "ES"
    assert data["players"][1]["matches_played"] == 0
    assert data["players"][1]["current_rank"] is None
    assert data["total_players"] == 2
    assert data["latest_match_date"] == "2026-08-10"
    assert data["total_matches"] == 123456


def test_directory_empty_players_and_summary():
    fake = _directory_fake_execute_df(pd.DataFrame(), summary_df=pd.DataFrame())
    with patch("src.serving.service.execute_df", side_effect=fake):
        resp = client.get("/directory")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["players"] == []
    assert data["total_players"] == 0
    assert data["latest_match_date"] is None
    assert data["total_matches"] == 0


def test_directory_database_error_returns_500():
    with patch("src.serving.service.execute_df", side_effect=RuntimeError("boom")):
        resp = client.get("/directory")
    assert resp.status_code == 500
    assert resp.json()["ok"] is False
    assert "directory query failed" in resp.json()["error"]


# ── /rank_history ───────────────────────────────────────────────────────────


def _rank_history_df() -> pd.DataFrame:
    # bronze.rankings rows, chronological (the query orders by ranking_date).
    return pd.DataFrame(
        [
            {"ranking_date": "2024-01-01", "rank": 100, "points": 400},
            {"ranking_date": "2024-02-05", "rank": 90, "points": 500},
            {"ranking_date": "2024-03-04", "rank": 88, "points": 530},
        ]
    )


def test_rank_history_shape_and_parameter_binding():
    with patch("src.serving.service.execute_df", return_value=_rank_history_df()) as exec:
        resp = client.get("/rank_history?player_id=p1")
    assert resp.status_code == 200
    sql, params = exec.call_args_list[0].args
    assert params == ["p1"]
    assert "%s" in sql
    data = resp.json()["data"]
    assert data["player_id"] == "p1"
    assert data["rank_history"] == [
        {"rank_date": "2024-01-01", "rank": 100},
        {"rank_date": "2024-02-05", "rank": 90},
        {"rank_date": "2024-03-04", "rank": 88},
    ]


def test_rank_history_empty_for_player_without_rankings():
    """A player with no bronze.rankings rows gets an empty history, not a 404."""
    with patch("src.serving.service.execute_df", return_value=pd.DataFrame()):
        resp = client.get("/rank_history?player_id=p1")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["player_id"] == "p1"
    assert data["rank_history"] == []


def test_rank_history_requires_player_id():
    resp = client.get("/rank_history")
    assert resp.status_code == 400
    assert "player_id" in resp.json()["error"]


# ── /match_history ──────────────────────────────────────────────────────────


def _match_history_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m2",
                "match_date": "2026-01-20",
                "tournament": "grand_slam",
                "tournament_name": "Australian Open",
                "surface": "hard",
                "round": "r32",
                "opponent_id": "p9",
                "opponent_name": "Rival",
                "opponent_ranking": 5,
                "match_won": 0,
                "score": "4-6 6-4 6-7",
                "aces": 12,
                "double_faults": 3,
                "first_serve_points_won": 50,
                "second_serve_points_won": 20,
                "total_serve_points": 100,
                "service_games": 14,
                "break_points_saved": 4,
                "break_points_faced": 6,
            },
            {
                "match_id": "m1",
                "match_date": "2026-01-10",
                "tournament": "masters",
                "tournament_name": None,
                "surface": "hard",
                "round": "qf",
                "opponent_id": "p8",
                "opponent_name": None,
                "opponent_ranking": 4,
                "match_won": 1,
                "score": "6-4 7-6",
                "aces": 8,
                "double_faults": 1,
                "first_serve_points_won": 40,
                "second_serve_points_won": 15,
                "total_serve_points": 80,
                "service_games": 12,
                "break_points_saved": 2,
                "break_points_faced": 3,
            },
        ]
    )


def test_match_history_shape_and_default_limit():
    with patch("src.serving.service.execute_df", return_value=_match_history_df()) as exec:
        resp = client.get("/match_history?player_id=p1")
    assert resp.status_code == 200
    sql, params = exec.call_args_list[0].args
    assert params == ["p1", 20]  # default limit
    assert "%s" in sql
    matches = resp.json()["data"]["matches"]
    assert len(matches) == 2
    assert matches[0]["match_id"] == "m2"
    assert matches[0]["result"] == "lost"
    assert matches[0]["opponent_name"] == "Rival"
    assert matches[0]["opponent_ranking"] == 5
    assert matches[0]["score"] == "4-6 6-4 6-7"
    assert matches[1]["result"] == "won"
    assert matches[1]["tournament_name"] is None
    assert matches[1]["opponent_name"] is None
    assert matches[1]["opponent_ranking"] == 4
    # The rank shown is the opponent's, never the profile player's, and the
    # ambiguous old `ranking` field is gone.
    assert "ranking" not in matches[0]
    assert "player_ranking" not in matches[0]
    # The limit is a bound parameter, never interpolated into the SQL string.
    assert "LIMIT %s" in sql


def test_match_history_returns_individual_matches_not_grouped_by_tournament():
    """Same-occurrence rounds (Rome final + earlier rounds) all come back as
    individual rows, newest first, up to the limit — no tournament dedup."""
    rows = [
        ("m4", "2026-05-25", "grand_slam", "Roland Garros", "r128", "p9"),
        ("m3", "2026-05-17", "masters", "Rome", "f", "p8"),
        ("m2", "2026-05-17", "masters", "Rome", "sf", "p7"),
        ("m1", "2026-05-17", "masters", "Rome", "r16", "p6"),
    ]
    df = pd.DataFrame(
        [
            {
                "match_id": mid,
                "match_date": date,
                "tournament": tier,
                "tournament_name": name,
                "surface": "clay",
                "round": rnd,
                "opponent_id": opp,
                "opponent_name": opp,
                "opponent_ranking": i,
                "match_won": 1,
                "score": "6-0 6-0",
                "aces": 0,
                "double_faults": 0,
                "first_serve_points_won": 0,
                "second_serve_points_won": 0,
                "total_serve_points": 0,
                "service_games": 0,
                "break_points_saved": 0,
                "break_points_faced": 0,
            }
            for i, (mid, date, tier, name, rnd, opp) in enumerate(rows, start=1)
        ]
    )
    with patch("src.serving.service.execute_df", return_value=df) as exec:
        resp = client.get("/match_history?player_id=p1&limit=20")
    assert resp.status_code == 200
    sql, params = exec.call_args_list[0].args
    assert params == ["p1", 20]
    assert "LIMIT %s" in sql
    # All three Rome rounds survive as individual matches, newest first.
    matches = resp.json()["data"]["matches"]
    assert [m["match_id"] for m in matches] == ["m4", "m3", "m2", "m1"]
    rome = [m for m in matches if m["tournament_name"] == "Rome"]
    assert [m["round"] for m in rome] == ["f", "sf", "r16"]


def test_match_history_limit_clamped():
    for raw, expected in (("500", 100), ("0", 1), ("7", 7)):
        with patch("src.serving.service.execute_df", return_value=pd.DataFrame()) as exec:
            resp = client.get(f"/match_history?player_id=p1&limit={raw}")
        assert resp.status_code == 200
        assert exec.call_args_list[0].args[1] == ["p1", expected]


def test_match_history_invalid_limit_returns_400():
    resp = client.get("/match_history?player_id=p1&limit=abc")
    assert resp.status_code == 400
    assert "limit" in resp.json()["error"]


def test_match_history_requires_player_id():
    resp = client.get("/match_history")
    assert resp.status_code == 400
    assert "player_id" in resp.json()["error"]


# ── /head_to_head ───────────────────────────────────────────────────────────


def _h2h_df() -> pd.DataFrame:
    # Meetings newest-first (the query orders match_date DESC, match_id DESC).
    rows = [
        ("m6", "2026-02-01", "a", "b", "a"),
        ("m5", "2026-01-01", "a", "b", "b"),
        ("m4", "2025-12-01", "a", "b", "a"),
        ("m3", "2025-11-01", "a", "b", "b"),
        ("m2", "2025-10-01", "a", "b", "b"),
        ("m1", "2025-09-01", "a", "b", "a"),
    ]
    return pd.DataFrame(
        [
            {
                "match_id": mid,
                "match_date": date,
                "tournament": "masters",
                "tournament_name": "Indian Wells Masters",
                "round": "qf",
                "surface": "hard",
                "player1_id": p1,
                "player2_id": p2,
                "winner_id": winner,
                "score": "6-4 6-4",
            }
            for mid, date, p1, p2, winner in rows
        ]
    )


def test_head_to_head_canonical_orientation_both_param_conventions():
    """Response echoes the supplied order: the first-supplied id is the
    player1 side — for both parameter conventions, in either request order."""
    for params in (
        {"player1_id": "a", "player2_id": "b"},
        {"player_id": "a", "opponent_id": "b"},
    ):
        with patch("src.serving.service.execute_df", return_value=_h2h_df()) as exec:
            resp = client.get("/head_to_head", params=params)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["player1_id"] == "a"
        assert data["player2_id"] == "b"
        # Newest meeting m6 was won by "a" = the first-supplied id.
        assert data["meetings"][0]["player1_won"] is True
        sql, bound = exec.call_args_list[0].args
        assert bound[:4] == ["a", "b", "b", "a"]
        assert bound[4] == 100  # default limit
        assert "LIMIT %s" in sql
    for params in (
        {"player1_id": "b", "player2_id": "a"},
        {"player_id": "b", "opponent_id": "a"},
    ):
        with patch("src.serving.service.execute_df", return_value=_h2h_df()) as exec:
            resp = client.get("/head_to_head", params=params)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["player1_id"] == "b"
        assert data["player2_id"] == "a"
        # Same meeting m6 (winner "a"): the first-supplied id "b" lost it.
        assert data["meetings"][0]["player1_won"] is False
        assert data["meetings"][0]["loser_id"] == "b"
        sql, bound = exec.call_args_list[0].args
        assert bound[:4] == ["b", "a", "a", "b"]
        assert bound[4] == 100  # default limit
        assert "LIMIT %s" in sql


def test_head_to_head_meetings_and_summary():
    with patch("src.serving.service.execute_df", return_value=_h2h_df()):
        resp = client.get("/head_to_head?player1_id=a&player2_id=b")
    assert resp.status_code == 200
    data = resp.json()["data"]
    meetings = data["meetings"]
    assert len(meetings) == 6
    assert [m["match_id"] for m in meetings] == ["m6", "m5", "m4", "m3", "m2", "m1"]
    assert meetings[0]["player1_won"] is True
    assert meetings[0]["winner_id"] == "a"
    assert meetings[0]["loser_id"] == "b"
    assert meetings[1]["player1_won"] is False
    assert meetings[1]["loser_id"] == "a"
    assert meetings[0]["score"] == "6-4 6-4"
    assert "match_id" in meetings[0]
    assert meetings[0]["tournament_name"] == "Indian Wells Masters"
    assert data["summary"] == {
        "meetings": 6,
        "player1_wins": 3,
        "player2_wins": 3,
        "player1_win_rate": 0.5,
        # last-5 mirror: rows m6..m2 -> a wins m6 and m4 -> 2/5
        "last5_player1_win_rate": 0.4,
    }


def test_head_to_head_zero_meetings():
    with patch("src.serving.service.execute_df", return_value=pd.DataFrame()):
        resp = client.get("/head_to_head?player1_id=a&player2_id=b")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["meetings"] == []
    assert data["summary"] == {
        "meetings": 0,
        "player1_wins": 0,
        "player2_wins": 0,
        "player1_win_rate": 0.5,
        "last5_player1_win_rate": 0.5,
    }


def test_head_to_head_limit_clamped():
    with patch("src.serving.service.execute_df", return_value=pd.DataFrame()) as exec:
        resp = client.get("/head_to_head?player1_id=a&player2_id=b&limit=500")
    assert resp.status_code == 200
    assert exec.call_args_list[0].args[1][4] == 100
    with patch("src.serving.service.execute_df", return_value=pd.DataFrame()) as exec:
        resp = client.get("/head_to_head?player1_id=a&player2_id=b&limit=0")
    assert resp.status_code == 200
    assert exec.call_args_list[0].args[1][4] == 1


def test_head_to_head_validation_errors():
    resp = client.get("/head_to_head")
    assert resp.status_code == 400
    resp = client.get("/head_to_head?player1_id=a")
    assert resp.status_code == 400
    resp = client.get("/head_to_head?player1_id=a&player2_id=b&limit=abc")
    assert resp.status_code == 400
    assert "limit" in resp.json()["error"]


def test_head_to_head_database_error_returns_500():
    with patch("src.serving.service.execute_df", side_effect=RuntimeError("boom")):
        resp = client.get("/head_to_head?player1_id=a&player2_id=b")
    assert resp.status_code == 500
    assert "head-to-head query failed" in resp.json()["error"]


# ── predict_from_ids order preservation ──────────────────────────────────────


class _ProbaModel:
    """Constant-probability sklearn-style classifier (class 1 = positive)."""

    def __init__(self, p: float) -> None:
        self.classes_ = np.array([0, 1])
        self._p = p

    def predict_proba(self, features) -> np.ndarray:
        return np.array([[1 - self._p, self._p]] * len(features))


class _ONNXSession:
    """Tiny ONNX-session stand-in returning a positive logit per batch row."""

    def run(self, _output_names: object, inputs: Any) -> list[np.ndarray]:
        batch = len(inputs["player_hist"])
        return [np.full((batch, 1), 2.0)]  # sigmoid(2) ~ 0.88 per row


def _fake_execute_df(sql: str, _params: list[object] | None = None) -> pd.DataFrame:
    """Hermetic execute_df stand-in: cold-start DB state keyed on the SQL text."""
    if "tour_averages" in sql:
        row: dict[str, object] = dict.fromkeys(TOUR_AVERAGES_FALLBACK_COLS, 1.0)
        row.update(
            singleton_id=1,
            pool_as_of_date=date.today(),
            snapshot_pool_rows=0,
            snapshot_pool_players=0,
            profile_rows=0,
            player_match_rows=0,
        )
        return pd.DataFrame([row])
    if "rolling_features" in sql:
        return pd.DataFrame()  # cold start: no snapshots
    if "player_matches" in sql:
        return pd.DataFrame([{"n": 0}])
    if "match_events" in sql:
        return pd.DataFrame(columns=["winner_id"])
    return pd.DataFrame()  # player_profiles: cold start


def test_predict_from_ids_preserves_caller_order():
    """predict_from_ids preserves caller order and swaps response sides with the ids."""
    # Use the decorated class's inner type to bypass DB/bootstrap initialization.
    pred = cast(Any, object.__new__(TennisPredictor.inner))  # type: ignore[attr-defined,arg-type]
    pred._stack_order = list(STACK_ORDER)
    pred.scaler = StandardScaler().fit(
        pd.DataFrame(np.zeros((1, len(FEATURE_COLS))), columns=FEATURE_COLS)
    )
    pred.linear = _ProbaModel(0.8)
    pred.gbdt = _ProbaModel(0.8)
    pred.production = _ProbaModel(0.9)
    pred.nn_session = _ONNXSession()
    # Trivial identity preprocessing so the GRU input builder runs hermetically.
    pred.nn_preprocessing = ni.GRUPreprocessing(
        fill_stats=np.zeros(ni.N_RAW, dtype=np.float32),
        scale_mean=np.zeros(ni.N_RAW, dtype=np.float32),
        scale_scale=np.ones(ni.N_RAW, dtype=np.float32),
    )

    def _empty_gru_history(_sql, _params=None):
        return pd.DataFrame()

    with (
        patch("src.features.inference.execute_df", side_effect=_fake_execute_df),
        patch("src.features.tour_averages.execute_df", side_effect=_fake_execute_df),
        patch("src.features.nn_inference.execute_df", side_effect=_empty_gru_history),
    ):
        ab = pred.predict_from_ids(
            PredictFromIdsRow(
                player_id="S0AG",
                opponent_id="Z355",
                surface=Surface.HARD,
                best_of=BestOf.BO3,
            )
        )
        ba = pred.predict_from_ids(
            PredictFromIdsRow(
                player_id="Z355",
                opponent_id="S0AG",
                surface=Surface.HARD,
                best_of=BestOf.BO3,
            )
        )

    assert ab["player_id"] == "S0AG"
    assert ab["opponent_id"] == "Z355"
    assert ab["predicted_winner"] == "S0AG"  # first-supplied id is the player side
    assert ba["player_id"] == "Z355"
    assert ba["opponent_id"] == "S0AG"
    assert ba["predicted_winner"] == "Z355"


def _make_predictor() -> Any:
    """Return a partially-initialized TennisPredictor for hermetic prediction tests."""
    pred = cast(Any, object.__new__(TennisPredictor.inner))  # type: ignore[attr-defined,arg-type]
    pred._stack_order = list(STACK_ORDER)
    pred.scaler = StandardScaler().fit(
        pd.DataFrame(np.zeros((1, len(FEATURE_COLS))), columns=FEATURE_COLS)
    )
    pred.linear = _ProbaModel(0.8)
    pred.gbdt = _ProbaModel(0.8)
    pred.production = _ProbaModel(0.9)
    pred.nn_session = _ONNXSession()
    pred.nn_preprocessing = ni.GRUPreprocessing(
        fill_stats=np.zeros(ni.N_RAW, dtype=np.float32),
        scale_mean=np.zeros(ni.N_RAW, dtype=np.float32),
        scale_scale=np.ones(ni.N_RAW, dtype=np.float32),
    )
    return pred


def test_predict_from_ids_bulk_preserves_caller_order_and_gru_path():
    """Bulk prediction builds paired GRU inputs per row and preserves input order."""
    pred = _make_predictor()

    def _empty_gru_history(_sql, _params=None):
        return pd.DataFrame()

    def _fake_bulk_build(rows):
        out = []
        for r in rows:
            pid = r["player_id"] if isinstance(r, dict) else r.player_id
            oid = r["opponent_id"] if isinstance(r, dict) else r.opponent_id
            out.append({**dict.fromkeys(FEATURE_COLS, 0.0), "player_id": pid, "opponent_id": oid})
        return pd.DataFrame(out)

    rows = [
        PredictFromIdsRow(
            player_id="S0AG", opponent_id="Z355", surface=Surface.HARD, best_of=BestOf.BO3
        ),
        PredictFromIdsRow(
            player_id="Z355", opponent_id="S0AG", surface=Surface.CLAY, best_of=BestOf.BO5
        ),
    ]
    with (
        patch("src.features.inference.build_inference_features_bulk", side_effect=_fake_bulk_build),
        patch("src.features.nn_inference.execute_df", side_effect=_empty_gru_history),
    ):
        out = pred.predict_from_ids_bulk(rows)

    assert [r["player_id"] for r in out] == ["S0AG", "Z355"]
    assert [r["opponent_id"] for r in out] == ["Z355", "S0AG"]
    for r in out:
        assert "p_nn" in r
        assert 0.0 <= float(r["p_nn"]) <= 1.0


def test_predict_from_ids_schema_derives_context_and_defaults():
    row = PredictFromIdsRow(
        player_id="S0AG",
        opponent_id="Z355",
        surface=Surface.HARD,
        best_of=BestOf.BO3,
        tournament=TournamentLevel.GRAND_SLAM,
        round=Round.F,
    )

    assert row.best_of == BestOf.BO3
    assert row.best_of.value == 3
    assert row.tournament == TournamentLevel.GRAND_SLAM
    assert row.round == Round.F
    assert row.is_indoor == 0
    assert row.as_of_date == date.today()


def test_predict_from_ids_schema_rejects_numeric_context_fields():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PredictFromIdsRow.model_validate(
            {
                "player_id": "S0AG",
                "opponent_id": "Z355",
                "surface": Surface.HARD,
                "best_of": 3,
                "tournament_level": 0,
            }
        )
    with pytest.raises(ValidationError):
        PredictFromIdsRow.model_validate(
            {
                "player_id": "S0AG",
                "opponent_id": "Z355",
                "surface": Surface.HARD,
                "best_of": 3,
                "round_encoded": 0,
            }
        )


# ── best_of contract (required, 1/3/5 only, bool + extras rejected) ─────────


def test_best_of_accepts_only_canonical_lengths():
    for value in (1, 3, 5):
        row = PredictFromIdsRow.model_validate(
            {"player_id": "S0AG", "opponent_id": "Z355", "surface": "hard", "best_of": value}
        )
        assert row.best_of == BestOf(value)
        assert row.best_of.value == value


def test_best_of_required_field_rejects_omitted():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PredictFromIdsRow.model_validate(
            {"player_id": "S0AG", "opponent_id": "Z355", "surface": "hard"}
        )


@pytest.mark.parametrize("bad", [0, 2, 4, 7, "3", 3.0, None])
def test_best_of_rejects_unsupported_values(bad):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PredictFromIdsRow.model_validate(
            {
                "player_id": "S0AG",
                "opponent_id": "Z355",
                "surface": "hard",
                "best_of": bad,
            }
        )


@pytest.mark.parametrize("flag", [True, False])
def test_best_of_rejects_bool(flag):
    from pydantic import ValidationError

    # bool is a subclass of int, but it is never a best_of length; reject it
    # explicitly so True/False cannot slip through as 1/0.
    with pytest.raises(ValidationError):
        PredictFromIdsRow.model_validate(
            {
                "player_id": "S0AG",
                "opponent_id": "Z355",
                "surface": "hard",
                "best_of": flag,
            }
        )


def test_best_of_rejects_extra_fields():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PredictFromIdsRow.model_validate(
            {
                "player_id": "S0AG",
                "opponent_id": "Z355",
                "surface": "hard",
                "best_of": 3,
                "tournament_level": 0,
            }
        )


# ── malformed request deserialization: filter BentoML client-error tracebacks ──


def _error_record(exc: Exception) -> logging.LogRecord:
    """Shaped like the record BentoML's log_exception emits per rejected request."""
    return logging.LogRecord(
        name="bentoml._internal.server.http_app",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="Exception on /predict_from_ids [POST]",
        args=(),
        exc_info=(type(exc), exc, exc.__traceback__),
    )


def test_validation_traceback_filter_drops_pydantic_errors(caplog):
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as ctx:
        PredictFromIdsRow.model_validate_json(
            '{"player_id": "S0AG", "opponent_id": "Z355", "surface": "sand"}'
        )
    with caplog.at_level(logging.WARNING, logger="tennis_ml.serving"):
        kept = _SuppressRequestValidationTraceback().filter(_error_record(ctx.value))

    assert kept is False  # traceback record dropped
    assert any(
        "request rejected with 400" in r.message and "Input should be" in r.message
        for r in caplog.records
    )


def test_surface_accepts_exactly_the_four_canonicals():
    from pydantic import ValidationError

    assert {s.value for s in Surface} == {"clay", "grass", "hard", "carpet"}
    for surface in ("clay", "grass", "hard", "carpet"):
        PredictFromIdsRow.model_validate(
            {"player_id": "S0AG", "opponent_id": "Z355", "surface": surface, "best_of": 3}
        )
    # The legacy "0" unknown-surface marker is no longer accepted.
    for surface in ("0", 0, None):
        with pytest.raises(ValidationError):
            PredictFromIdsRow.model_validate(
                {"player_id": "S0AG", "opponent_id": "Z355", "surface": surface, "best_of": 3}
            )


def test_validation_traceback_filter_keeps_server_errors():
    f = _SuppressRequestValidationTraceback()
    assert f.filter(_error_record(RuntimeError("boom"))) is True
    assert (
        f.filter(logging.LogRecord("x", logging.ERROR, __file__, 1, "msg", (), exc_info=None))
        is True
    )
