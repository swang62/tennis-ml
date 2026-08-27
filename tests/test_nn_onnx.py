"""Hermetic PyTorch/ONNX Runtime parity and antisymmetry for the GRU (nn).

Builds an in-memory ``SymmetricGRU``, exports it to ONNX with the named
dynamic-batch inputs the service consumes, and verifies ORT logits match the
PyTorch logits for both single-row and multi-row batches. It also confirms the
serving antisymmetric combine yields complementary ``p_nn`` probabilities.
"""

import numpy as np
import pytest

from src.serving.service import _antisymmetric_nn_probs
from src.training import nn_history as gh
from src.training.nn import SymmetricGRU

torch = pytest.importorskip("torch")

H, R, C = gh.HISTORY_LEN, gh.N_RAW, len(gh.GRU_CONTEXT_NAMES)


@pytest.fixture(scope="module")
def onnx_path(tmp_path_factory):
    model = SymmetricGRU(hist_dim=R, context_dim=C, hidden_dim=32, dropout=0.0).eval()
    path = tmp_path_factory.mktemp("onnx") / "nn_best.onnx"
    dummy = (
        torch.zeros(2, H, R),
        torch.zeros(2, H, R),
        torch.zeros(2, H),
        torch.zeros(2, H),
        torch.zeros(2, C),
    )
    batch_dim = torch.export.Dim("batch")
    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy,
            str(path),
            input_names=[
                "player_hist",
                "opponent_hist",
                "player_valid",
                "opponent_valid",
                "context",
            ],
            output_names=["logit"],
            opset_version=18,
            dynamic_shapes={
                "player_hist": {0: batch_dim},
                "opponent_hist": {0: batch_dim},
                "player_valid": {0: batch_dim},
                "opponent_valid": {0: batch_dim},
                "context": {0: batch_dim},
            },
            external_data=False,
        )
    return model, path


def _ort_logits(sess, ph, oh, pv, ov, ctx):
    return sess.run(
        None,
        {
            "player_hist": ph.numpy().astype(np.float32),
            "opponent_hist": oh.numpy().astype(np.float32),
            "player_valid": pv.numpy().astype(np.float32),
            "opponent_valid": ov.numpy().astype(np.float32),
            "context": ctx.numpy().astype(np.float32),
        },
    )[0].reshape(-1)


def test_onnx_parity_batch_and_single(onnx_path):
    import onnxruntime as ort

    model, path = onnx_path
    sess = ort.InferenceSession(str(path))
    torch.manual_seed(0)
    ph = torch.randn(4, H, R)
    oh = torch.randn(4, H, R)
    pv = (torch.rand(4, H) > 0.4).float()
    ov = (torch.rand(4, H) > 0.4).float()
    ctx = torch.randn(4, C)

    with torch.inference_mode():
        pt = model(ph, oh, pv, ov, ctx).detach().numpy()
    ort_logits = _ort_logits(sess, ph, oh, pv, ov, ctx)
    assert np.max(np.abs(pt - ort_logits)) < 1e-4

    # single-row batch
    with torch.inference_mode():
        pt1 = model(ph[:1], oh[:1], pv[:1], ov[:1], ctx[:1]).detach().numpy()
    ort1 = _ort_logits(sess, ph[:1], oh[:1], pv[:1], ov[:1], ctx[:1])
    assert np.max(np.abs(pt1 - ort1)) < 1e-4


def test_ort_gru_antisymmetry_and_complementary_probs(onnx_path):
    """Swapping player/opponent with the same context negates the ORT logit, so
    the antisymmetric combine yields exact complementary p_nn values."""
    import onnxruntime as ort

    _model, path = onnx_path
    sess = ort.InferenceSession(str(path))
    torch.manual_seed(1)
    ph = torch.randn(3, H, R)
    oh = torch.randn(3, H, R)
    pv = (torch.rand(3, H) > 0.4).float()
    ov = (torch.rand(3, H) > 0.4).float()
    ctx = torch.randn(3, C)

    ab = _ort_logits(sess, ph, oh, pv, ov, ctx)
    # BA orientation: swap player/opponent, keep the (same) context.
    ba = _ort_logits(sess, oh, ph, ov, pv, ctx)
    p_nn_ab, p_nn_ba = _antisymmetric_nn_probs(ab, ba)

    assert np.allclose(ab + ba, 0.0, atol=1e-5)  # raw antisymmetry
    assert np.allclose(p_nn_ab + p_nn_ba, 1.0, atol=1e-9)
    # p_nn_ab should match the standard sigmoid of the AB logit here.
    assert np.allclose(p_nn_ab, 1.0 / (1.0 + np.exp(-ab)), atol=1e-9)
