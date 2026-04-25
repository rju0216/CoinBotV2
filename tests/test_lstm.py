"""src/ml/models.py LSTMClassifier 단위 테스트.

forward shape, 단일/다중 레이어, eval 모드 등 검증.
"""

from __future__ import annotations

import pytest

try:
    import torch
    from src.ml.models import LSTMClassifier
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch 미설치")
class TestLSTMClassifier:
    def test_forward_shape(self):
        model = LSTMClassifier(n_features=27, hidden_size=64)
        x = torch.randn(2, 10, 27)  # batch=2, seq=10, features=27
        out = model(x)
        assert out.shape == (2, 3)

    def test_single_sample(self):
        model = LSTMClassifier(n_features=27)
        x = torch.randn(1, 60, 27)
        out = model(x)
        assert out.shape == (1, 3)

    def test_multi_layer(self):
        model = LSTMClassifier(n_features=27, num_layers=2)
        x = torch.randn(4, 30, 27)
        out = model(x)
        assert out.shape == (4, 3)

    def test_eval_mode_no_error(self):
        model = LSTMClassifier(n_features=27)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 10, 27))
        assert out.shape == (1, 3)
