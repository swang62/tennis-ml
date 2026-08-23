"""Neural-network fit uses sample weights; validation and prediction do not.

The candidate fit path passes recency-aligned weights to ``training_step``
(normalized weighted BCE). Validation loss and inference stay unweighted so the
selection metrics are not distorted by the recency weighting.
"""

from __future__ import annotations

import torch
import pytest

from src.training.nn import TabularMLP


def test_training_step_unweighted_is_plain_bce():
    torch.manual_seed(0)
    model = TabularMLP(tab_dim=4)
    model.log = lambda name, value, **_kw: None  # type: ignore[method-assign]
    tab = torch.randn(8, 4)
    labels = torch.randint(0, 2, (8,)).float()
    out = model(tab)
    loss = model.training_step((tab, labels), 0)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(out, labels)
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)


def test_training_step_weighted_is_per_sample_bce_mean():
    torch.manual_seed(0)
    model = TabularMLP(tab_dim=4)
    model.log = lambda name, value, **_kw: None  # type: ignore[method-assign]
    tab = torch.randn(8, 4)
    labels = torch.randint(0, 2, (8,)).float()
    weights = torch.tensor([3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    out = model(tab)
    per = torch.nn.functional.binary_cross_entropy_with_logits(out, labels, reduction="none")
    expected = (per * weights).mean()
    loss = model.training_step((tab, labels, weights), 0)
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)


def test_validation_loss_is_unweighted_bce():
    torch.manual_seed(0)
    model = TabularMLP(tab_dim=4)
    captured: dict[str, torch.Tensor] = {}
    model.log = lambda name, value, **_kw: captured.update({name: value})  # type: ignore[method-assign]
    tab = torch.randn(8, 4)
    labels = torch.randint(0, 2, (8,)).float()
    out = model(tab)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(out, labels)
    model.validation_step((tab, labels), 0)
    assert captured["val_loss"].detach().item() == pytest.approx(expected.detach().item(), rel=1e-4)


def test_validation_step_ignores_weights():
    # The validation batch is a 2-tuple; a weighted 3-tuple is rejected,
    # confirming validation never consumes recency weights.
    torch.manual_seed(0)
    model = TabularMLP(tab_dim=4)
    tab = torch.randn(8, 4)
    labels = torch.randint(0, 2, (8,)).float()
    weights = torch.ones(8)
    with pytest.raises(ValueError):
        model.validation_step((tab, labels, weights), 0)
    # The unweighted 2-tuple still runs cleanly.
    model.log = lambda name, value, **_kw: None  # type: ignore[method-assign]
    model.validation_step((tab, labels), 0)


def test_predict_step_returns_sigmoid_probabilities():
    torch.manual_seed(0)
    model = TabularMLP(tab_dim=4)
    tab = torch.randn(8, 4)
    probs = model.predict_step((tab, torch.zeros(8)), 0)
    assert torch.all((probs >= 0) & (probs <= 1))
