"""Hermetic behavioral tests for the GRU discovery model (SymmetricGRU).

No training, MLflow, or database: random finite tensors and locally gathered
sequences exercise the antisymmetric scorer and the empty-history fallback.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.training import gru_history as gh
from src.training.nn import SymmetricGRU

HIST_DIM = gh.STORE_WIDTH  # 15
SEQ_LEN = gh.HISTORY_LEN  # 10
CTX_DIM = len(gh.GRU_CONTEXT_NAMES)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _random_batch(batch=2):
    ph = torch.randn(batch, SEQ_LEN, HIST_DIM)
    oh = torch.randn(batch, SEQ_LEN, HIST_DIM)
    pv = torch.randint(0, SEQ_LEN + 1, (batch,))
    ov = torch.randint(0, SEQ_LEN + 1, (batch,))
    ctx = torch.randn(batch, CTX_DIM)
    return ph, oh, pv, ov, ctx


def test_swapped_sequence_sides_negate_logit():
    model = SymmetricGRU(hist_dim=HIST_DIM, context_dim=len(gh.GRU_CONTEXT_NAMES))
    model.eval()
    ph, oh, pv, ov, ctx = _random_batch()
    fwd = model(ph, oh, pv, ov, ctx)
    swapped = model(oh, ph, ov, pv, ctx)
    assert fwd.shape == (2,)
    assert torch.isfinite(fwd).all()
    # score(player) - score(opponent) flips sign exactly when sides swap.
    assert torch.allclose(fwd, -swapped, atol=1e-5)


def test_empty_history_is_finite_and_antisymmetric():
    model = SymmetricGRU(hist_dim=HIST_DIM, context_dim=len(gh.GRU_CONTEXT_NAMES))
    model.eval()
    ph, oh, pv, ov, ctx = _random_batch()
    ph = ph.clone()
    oh = oh.clone()
    pv = torch.zeros(2, dtype=torch.long)
    ov = torch.zeros(2, dtype=torch.long)
    fwd = model(ph, oh, pv, ov, ctx)
    swapped = model(oh, ph, ov, pv, ctx)
    assert torch.isfinite(fwd).all()
    assert torch.allclose(fwd, -swapped, atol=1e-5)


def test_gathered_sequences_negate_end_to_end():
    # Wire the history store into the model with real (imputed) sequences.
    rows = [
        {
            "match_id": "m",
            "match_date": "2024-01-01",
            "match_num": 1,
            "surface": "hard",
            "player_id": "A",
            "opponent_id": "B",
            "match_won": 1.0,
            "player_ranking": 20.0,
            "player_rank_points": 1200.0,
            "opponent_ranking": 10.0,
            "opponent_rank_points": 1500.0,
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
        },
        {
            "match_id": "m",
            "match_date": "2024-01-01",
            "match_num": 1,
            "surface": "clay",
            "player_id": "B",
            "opponent_id": "A",
            "match_won": 0.0,
            "player_ranking": 10.0,
            "player_rank_points": 1500.0,
            "opponent_ranking": 20.0,
            "opponent_rank_points": 1200.0,
            "aces": 8.0,
            "double_faults": 3.0,
            "first_serves_made": 55.0,
            "total_serve_points": 90.0,
            "first_serve_points_won": 40.0,
            "second_serve_points_won": 18.0,
            "service_games": 18.0,
            "return_points_won": 25.0,
            "return_points_available": 55.0,
            "break_points_saved": 2.0,
            "break_points_faced": 4.0,
        },
    ]
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
    for c in cols[6:]:
        df[c] = df[c].astype("float64")

    store = gh.build_history_store(df)
    info = pd.DataFrame(
        [
            {"player_id": "A", "opponent_id": "B", "match_id": "m"},
            {"player_id": "B", "opponent_id": "A", "match_id": "m"},
        ]
    )
    p_idx, o_idx = gh.map_split_indices(store, info)
    imputed = store.impute(np.arange(len(df)))
    ph, oh, pv, ov = store.gather(imputed, p_idx, o_idx)
    x = pd.DataFrame(
        [
            {k: 0.0 for k in gh.GRU_CONTEXT_NAMES if k != "is_carpet"},
            {k: 0.0 for k in gh.GRU_CONTEXT_NAMES if k != "is_carpet"},
        ]
    )
    x["is_hard"] = 1.0
    x["is_clay"] = 0.0
    ctx = torch.from_numpy(gh.build_context_tensor(x))

    model = SymmetricGRU(hist_dim=HIST_DIM, context_dim=len(gh.GRU_CONTEXT_NAMES))
    model.eval()
    ph_t = torch.from_numpy(ph)
    oh_t = torch.from_numpy(oh)
    pv_t = torch.from_numpy(pv)
    ov_t = torch.from_numpy(ov)
    fwd = model(ph_t, oh_t, pv_t, ov_t, ctx)
    swapped = model(oh_t, ph_t, ov_t, pv_t, ctx)
    assert torch.isfinite(fwd).all()
    assert torch.allclose(fwd, -swapped, atol=1e-5)
