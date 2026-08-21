import torch

from src.training.nn import TabularBioMLP


def _small_model(**kwargs) -> TabularBioMLP:
    torch.manual_seed(0)
    return TabularBioMLP(tab_dim=54, bio_dim=32, hidden_dim=16, **kwargs)


def _inputs():
    return torch.randn(2, 54), torch.randn(2, 32), torch.randn(2, 32)


def test_forward_output_shape():
    model = _small_model()
    out = model(*_inputs())
    assert out.shape == (2,)


def test_sigmoid_logits_in_unit_interval():
    model = _small_model()
    probs = torch.sigmoid(model(*_inputs()))
    assert probs.min() > 0.0
    assert probs.max() < 1.0


def test_deterministic_forward_with_dropout_zero():
    model = _small_model(dropout=0.0)
    tab, bio_p, bio_o = _inputs()
    assert torch.allclose(model(tab, bio_p, bio_o), model(tab, bio_p, bio_o), atol=0.0)
