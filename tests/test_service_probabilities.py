import numpy as np
import pytest

from src.serving.service import _positive_class_probability


class _Model:
    def __init__(self, classes):
        self.classes_ = np.array(classes)

    def predict_proba(self, _features):
        return np.array([[0.2, 0.8]])


def test_positive_probability_uses_class_label_not_column_position():
    assert _positive_class_probability(_Model([1, 0]), None).tolist() == [0.2]


def test_positive_probability_rejects_models_without_positive_label():
    with pytest.raises(ValueError, match="including 1"):
        _positive_class_probability(_Model([0, 2]), None)
