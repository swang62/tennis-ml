"""Hermetic behavioral tests for runtime GRU (``nn``) input construction.

No PostgreSQL, DuckDB, MLflow, or network access. History is supplied through
:func:`simulate_player_histories`, which replays the production set-wise filter
(``match_date < as_of``) over a local ``silver.player_matches`` frame so the
real offline transform path is exercised end to end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.features import nn_inference as ni
from src.training import nn_history as gh

HIST = ni.HISTORY_LEN
N_RAW = ni.N_RAW
CTX = ni.GRU_CONTEXT_NAMES


def _present(**over):
    row = {
        "match_id": "m",
        "match_date": "2024-01-01",
        "match_num": 1,
        "surface": "hard",
        "player_id": "P",
        "opponent_id": "O",
        "match_won": 1.0,
        "player_ranking": 20.0,
        "player_rank_points": 1200.0,
        "opponent_ranking": 50.0,
        "opponent_rank_points": 900.0,
        "aces": 10.0,
        "double_faults": 2.0,
        "first_serves_made": 60.0,
        "total_serve_points": 100.0,
        "first_serve_points_won": 45.0,
        "second_serve_points_won": 20.0,
        "service_games": 20.0,
        "return_points_won": 30.0,
        "return_points_available": 60.0,
        "break_points_saved": 3.0,
        "break_points_faced": 5.0,
    }
    row.update(over)
    return row


def _frame(rows):
    cols = ni._HISTORY_COLS
    df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    df["match_date"] = pd.to_datetime(df["match_date"])
    for c in ni._NUMERIC_COLS:
        df[c] = df[c].astype("float64")
    return df


def _ctx_row(**over):
    row = {
        "is_clay": 0.0,
        "is_grass": 0.0,
        "is_hard": 1.0,
        "is_indoor": 0.0,
        "best_of": 3.0,
        "tournament_level": 1.0,
        "round_encoded": 4.0,
        "elo_diff": 12.0,
        "age_diff": 2.0,
        "h2h_exposure": 3.0,
        "h2h_advantage": 1.0,
        "h2h_surface_advantage": 0.5,
    }
    row.update(over)
    return row


def _features_df(player_id, opponent_id, **ctx_over):
    return pd.DataFrame(
        [{**_ctx_row(**ctx_over), "player_id": player_id, "opponent_id": opponent_id}]
    )


def _preproc_from(store, idx):
    return ni.gru_preprocessing_from_store(store, np.asarray(idx))


# --- cold start -------------------------------------------------------------


def test_cold_start_is_finite_zero_sequence():
    pm = _frame([])  # no matches anywhere
    preproc = ni.GRUPreprocessing(
        fill_stats=np.zeros(N_RAW, dtype=np.float32),
        scale_mean=np.zeros(N_RAW, dtype=np.float32),
        scale_scale=np.ones(N_RAW, dtype=np.float32),
    )
    keys = {
        ("P", __import__("datetime").date(2024, 6, 1)),
        ("O", __import__("datetime").date(2024, 6, 1)),
    }
    lookup = ni.simulate_player_histories(pm, keys)
    batch = ni.build_gru_inputs_bulk(
        _features_df("P", "O"),
        as_of_dates=[__import__("datetime").date(2024, 6, 1)],
        preprocessing=preproc,
        history_lookup=lookup,
    )
    assert np.isfinite(batch.player_hist).all()
    assert np.isfinite(batch.opponent_hist).all()
    assert batch.player_len[0] == 0
    assert batch.opponent_len[0] == 0
    assert not batch.player_mask.any()
    assert not batch.opponent_mask.any()
    # All-zero padding survives untouched for a cold start.
    assert np.abs(batch.player_hist).max() == 0.0


# --- established player ------------------------------------------------------


def test_established_player_right_justified_and_finite():
    rows = [
        _present(match_id=f"m{k}", match_date=f"2024-01-{k + 1:02d}", aces=10.0 + k)
        for k in range(5)
    ]
    pm = _frame(rows)
    # Training-side store for the same data (full history, strict as-of).
    store = gh.build_history_store(pm, strict_as_of=True)
    preproc = _preproc_from(store, np.arange(len(pm)))
    as_of = __import__("datetime").date(2024, 2, 10)
    keys = {("P", as_of)}
    lookup = ni.simulate_player_histories(pm, keys)
    batch = ni.build_gru_inputs_bulk(
        _features_df("P", "O"),
        as_of_dates=[as_of],
        preprocessing=preproc,
        history_lookup=lookup,
    )
    assert batch.player_len[0] == 5
    # Right-justified: padding (if any) lives on the left, valid slots on the right.
    assert not batch.player_mask[0, : HIST - 5].any()
    assert batch.player_mask[0, HIST - 5 :].all()
    assert np.isfinite(batch.player_hist).all()
    assert batch.context.shape == (1, len(CTX))


# --- strict as-of boundary --------------------------------------------------


def test_strict_as_of_excludes_same_day_and_future():
    as_of = __import__("datetime").date(2024, 1, 3)
    rows = [
        _present(match_id="early", match_date="2024-01-01"),
        _present(match_id="sameday", match_date="2024-01-03"),
        _present(match_id="future", match_date="2024-01-05"),
    ]
    pm = _frame(rows)
    store = gh.build_history_store(pm, strict_as_of=True)
    preproc = _preproc_from(store, np.arange(len(pm)))
    keys = {("P", as_of)}
    lookup = ni.simulate_player_histories(pm, keys)
    batch = ni.build_gru_inputs_bulk(
        _features_df("P", "O"),
        as_of_dates=[as_of],
        preprocessing=preproc,
        history_lookup=lookup,
    )
    # Only the strictly-earlier match contributes.
    assert batch.player_len[0] == 1
    # Offline gold (strict_as_of) keeps the same single prior for P at this date.
    idx_early = store.index[("P", "early")]
    assert store.valid_mask[idx_early].sum() == 0  # 'early' has no prior


# --- offline / online equivalence -------------------------------------------


def test_offline_online_sequence_equivalence():
    # Player P has 12 prior matches; target T on a later date. Offline history
    # store (strict_as_of) and the runtime fetch (match_date < as_of) must build
    # identical preprocessed sequences for P at T's date.
    rows = [
        _present(match_id=f"m{k}", match_date=f"2024-01-{k + 1:02d}", aces=10.0 + k)
        for k in range(12)
    ]
    # The target match itself, also for P, on a later date.
    rows.append(_present(match_id="T", match_date="2024-02-01", player_id="P", opponent_id="O"))
    pm = _frame(rows)
    store = gh.build_history_store(pm, strict_as_of=True)
    preproc = _preproc_from(store, np.arange(len(pm)))

    as_of = __import__("datetime").date(2024, 2, 1)
    keys = {("P", as_of), ("O", as_of)}
    lookup = ni.simulate_player_histories(pm, keys)
    batch = ni.build_gru_inputs_bulk(
        _features_df("P", "O"),
        as_of_dates=[as_of],
        preprocessing=preproc,
        history_lookup=lookup,
    )

    # Offline sequence for target T (strict before its date).
    t_idx = store.index[("P", "T")]
    offline = store.impute_and_scale(np.arange(len(pm)))[t_idx]  # [HIST, N_RAW]
    online = batch.player_hist[0]

    # Same valid slots, right-justified, with the cap at HIST.
    assert offline.shape == online.shape == (HIST, N_RAW)
    assert np.array_equal(store.valid_mask[t_idx], batch.player_mask[0])
    # Preprocessed values match within tolerance on valid timesteps.
    mask = batch.player_mask[0]
    assert np.allclose(offline[mask], online[mask], atol=1e-4)
    # Padding slots are identical zeros.
    assert np.allclose(offline[~mask], online[~mask], atol=0.0)
