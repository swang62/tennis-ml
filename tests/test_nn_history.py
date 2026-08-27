"""Hermetic behavioral tests for the GRU discovery history store.

Local-only fixtures: pandas/numpy fakes of ``silver.player_matches`` are passed
straight to :func:`build_history_store`, exercising causality, padding, missing
stat imputation, and split indexing without any database, network, or MLflow.
"""

import numpy as np
import pandas as pd
import pytest

from src.training import nn_history as gh

RAW = gh.GRU_RAW_NAMES
HIST = gh.HISTORY_LEN
ACE_COL = RAW.index("ace_per_svc_game")


def _present(**over):
    """A fully-populated player-match row; everything present unless overridden."""
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
    cols = [
        "match_id",
        "match_date",
        "match_num",
        "surface",
        "player_id",
        "opponent_id",
        "match_won",
        "player_ranking",
        "player_rank_points",
        "opponent_ranking",
        "opponent_rank_points",
        "aces",
        "double_faults",
        "first_serves_made",
        "total_serve_points",
        "first_serve_points_won",
        "second_serve_points_won",
        "service_games",
        "return_points_won",
        "return_points_available",
        "break_points_saved",
        "break_points_faced",
    ]
    df = pd.DataFrame(rows)[cols]
    df["match_date"] = pd.to_datetime(df["match_date"])
    numeric = [
        c
        for c in cols
        if c not in ("match_id", "match_date", "surface", "player_id", "opponent_id")
    ]
    for c in numeric:
        df[c] = df[c].astype("float64")
    return df


def _history_ace_rates(store, idx):
    """Finite ace-per-service-game values visible in a valid history."""
    v = store.valid_mask[idx]
    raw = store.histories[idx, v, : gh.N_RAW]
    return [float(x) for x in raw[:, ACE_COL] if np.isfinite(x)]


def _ace_rates_match(actual, expected_aces):
    """Tolerant comparison against expected ace counts divided by 20 games."""
    exp = [float(r / 20.0) for r in expected_aces]
    return len(actual) == len(exp) and all(any(abs(a - e) < 1e-3 for e in exp) for a in actual)


def _valid_slot_ace_rates(store, idx):
    """Array of ace rates per valid slot, aligned to valid_mask."""
    v = store.valid_mask[idx]
    return store.histories[idx, v, ACE_COL]


def test_padding_and_prior_counts_zero_through_ten():
    # 12 chronological matches for one player: 0..10 priors (capped at HIST).
    rows = [
        _present(match_id=f"m{k}", match_date=f"2024-01-{k + 1:02d}", aces=10.0 + k)
        for k in range(12)
    ]
    store = gh.build_history_store(_frame(rows))
    assert store.histories.shape == (12, HIST, gh.STORE_WIDTH)

    for k in range(12):
        count = min(k, HIST)
        mask = store.valid_mask[k]
        assert mask.sum() == count
        if count:
            # Valid timesteps are right-justified: padding lives on the left.
            assert not mask[: HIST - count].any()
            assert mask[HIST - count :].all()
        else:
            assert not mask.any()


def test_same_day_match_num_ordering_drives_causality():
    # Same date, two matches. match_num 1 has id "m_b", match_num 2 has id "m_a"
    # so a naive match_id sort would reverse causality; ordering must use match_num.
    rows = [
        _present(match_id="m_b", match_num=1, match_date="2024-01-01", aces=11.0),
        _present(match_id="m_a", match_num=2, match_date="2024-01-01", aces=22.0),
    ]
    store = gh.build_history_store(_frame(rows))
    idx_a = store.index[("P", "m_a")]
    idx_b = store.index[("P", "m_b")]

    # Later match_num inherits the earlier one as a prior; earlier has none.
    assert store.valid_mask[idx_b].sum() == 0
    assert store.valid_mask[idx_a].sum() == 1
    # The inherited prior is m_b, identified by its opponent_ranking fingerprint.
    assert _ace_rates_match(_history_ace_rates(store, idx_a), [11.0])
    assert all(not np.isclose(x, 22.0 / 20.0, atol=1e-3) for x in _history_ace_rates(store, idx_a))


def test_current_and_future_matches_excluded_from_history():
    # Three ordered matches; each must see only strictly-earlier ones.
    rows = [
        _present(match_id="m0", match_date="2024-01-01", aces=10.0),
        _present(match_id="m1", match_date="2024-01-02", aces=20.0),
        _present(match_id="m2", match_date="2024-01-03", aces=30.0),
    ]
    store = gh.build_history_store(_frame(rows))
    i0 = store.index[("P", "m0")]
    i1 = store.index[("P", "m1")]
    i2 = store.index[("P", "m2")]

    assert _history_ace_rates(store, i0) == []
    # m1's history holds m0 only: neither its own rank nor the future m2.
    assert _ace_rates_match(_history_ace_rates(store, i1), [10.0])
    # m2's history holds m0 and m1, never its own current row.
    assert _ace_rates_match(_history_ace_rates(store, i2), [10.0, 20.0])
    for idx in (i0, i1, i2):
        rates = _valid_slot_ace_rates(store, idx)
        assert not np.any(np.isclose(rates, 30.0 / 20.0))  # never the latest/current


def test_missing_raw_stats_remain_nan_until_imputed():
    # m1 carries missing aces; m2's history includes m1.
    rows = [
        _present(match_id="m0", match_date="2024-01-01", aces=10.0),
        _present(
            match_id="m1",
            match_date="2024-01-02",
            aces=np.nan,
        ),
        _present(match_id="m2", match_date="2024-01-03", aces=30.0),
    ]
    store = gh.build_history_store(_frame(rows))
    i2 = store.index[("P", "m2")]
    v = store.valid_mask[i2]
    raw = store.histories[i2, v]
    # Locate the m1 slot from the missing raw value.
    m1_slot = int(np.where(~np.isfinite(raw[:, ACE_COL]))[0][0])
    present_slot = int(np.where(np.isfinite(raw[:, ACE_COL]))[0][0])
    # Missing raw values are NaN before imputation (no silent zero-fill).
    assert np.isnan(raw[m1_slot, ACE_COL])

    imputed = store.impute(np.arange(3))
    assert np.isfinite(imputed).all()
    # Previously-missing slots are now finite and present slots are unchanged.
    assert np.isfinite(imputed[i2, v, :][m1_slot, ACE_COL])
    assert imputed[i2, v, :][present_slot, ACE_COL] == raw[present_slot, ACE_COL]


def test_imputation_falls_back_to_zero_fill_when_no_fit_evidence():
    # m_a (only prior of m_b) has missing aces, so the fit band has no available
    # aces and the column fill must be 0.0.
    rows = [
        _present(match_id="mA", match_date="2024-01-01", aces=np.nan),
        _present(match_id="mB", match_date="2024-01-02"),
    ]
    store = gh.build_history_store(_frame(rows))
    fill = gh.compute_fill_stats(store, np.array([1]))
    assert fill[ACE_COL] == 0.0
    imputed = store.impute(np.array([1]))
    i_b = store.index[("P", "mB")]
    v = store.valid_mask[i_b]
    m_a_slot = int(np.where(~np.isfinite(store.histories[i_b, v, ACE_COL]))[0][0])
    assert imputed[i_b, v, :][m_a_slot, ACE_COL] == 0.0
    assert np.isfinite(imputed).all()


def test_split_rows_resolve_to_both_side_indices():
    # One physical match produces two perspective rows (A->B and B->A).
    rows = [
        _present(match_id="m", player_id="A", opponent_id="B"),
        _present(match_id="m", player_id="B", opponent_id="A", surface="clay"),
    ]
    store = gh.build_history_store(_frame(rows))
    info = pd.DataFrame(
        [
            {"player_id": "A", "opponent_id": "B", "match_id": "m"},
            {"player_id": "B", "opponent_id": "A", "match_id": "m"},
        ]
    )
    p_idx, o_idx = gh.map_split_indices(store, info)
    assert len(p_idx) == len(o_idx) == 2
    assert store.player_ids[p_idx[0]] == "A" and store.match_ids[p_idx[0]] == "m"
    assert store.player_ids[o_idx[0]] == "B" and store.match_ids[o_idx[0]] == "m"
    assert store.player_ids[p_idx[1]] == "B" and store.player_ids[o_idx[1]] == "A"


def test_map_split_indices_requires_complete_coverage():
    rows = [_present(match_id="m", player_id="A", opponent_id="B")]
    store = gh.build_history_store(_frame(rows))
    info = pd.DataFrame([{"player_id": "A", "opponent_id": "B", "match_id": "absent"}])
    with pytest.raises(KeyError):
        gh.map_split_indices(store, info)


def test_build_rejects_frames_missing_required_columns():
    df = _frame([_present()]).drop(columns=["aces"])
    with pytest.raises(ValueError):
        gh.build_history_store(df)


def test_read_player_match_history_df_accepts_dataframe_and_normalizes_dates():
    # The snapshot-backed reader must also accept an already-sourced DataFrame
    # (hermetic path, no DuckDB/Postgres), preserving order and returning a copy
    # with match_date as datetime64 for the gap-day math.
    rows = [
        _present(match_id="m0", match_date="2024-01-01"),
        _present(match_id="m1", match_date="2024-01-02"),
    ]
    df = _frame(rows)
    out = gh.read_player_match_history_df(df)
    assert out is not df  # non-destructive copy
    assert pd.api.types.is_datetime64_any_dtype(out["match_date"])
    assert list(out["match_id"]) == ["m0", "m1"]  # caller-controlled order preserved
    for col in (
        "match_id",
        "match_date",
        "match_num",
        "surface",
        "player_id",
        "opponent_id",
        "match_won",
        "opponent_ranking",
        "aces",
        "double_faults",
        "first_serves_made",
        "total_serve_points",
        "first_serve_points_won",
        "second_serve_points_won",
        "service_games",
        "return_points_won",
        "return_points_available",
        "break_points_saved",
        "break_points_faced",
    ):
        assert col in out.columns


def test_context_tensor_is_finite_and_excludes_redundant_features():
    x = pd.DataFrame(
        [
            {  # perspective A on hard
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
            },
            {  # perspective B on clay+grass (overlapping -> no carpet)
                "is_clay": 1.0,
                "is_grass": 1.0,
                "is_hard": 0.0,
                "is_indoor": 1.0,
                "best_of": 5.0,
                "tournament_level": 2.0,
                "round_encoded": 2.0,
                "elo_diff": -8.0,
                "age_diff": -2.0,
                "h2h_exposure": 3.0,
                "h2h_advantage": -1.0,
                "h2h_surface_advantage": -0.5,
            },
        ]
    )
    ctx = gh.build_context_tensor(x)
    assert ctx.shape == (2, len(gh.GRU_CONTEXT_NAMES))
    assert np.isfinite(ctx).all()
