"""src/ml/models.py TransformerClassifier + PositionalEncoding 단위 테스트.

forward shape, 홀수 d_model 대응(I-B003 회귀), eval 모드 등 검증.
"""

from __future__ import annotations

import pytest

try:
    import torch
    from src.ml.models import PositionalEncoding, TransformerClassifier
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch 미설치")
class TestTransformerClassifier:
    def test_forward_shape(self):
        model = TransformerClassifier(n_features=27, d_model=64)
        x = torch.randn(2, 10, 27)
        out = model(x)
        assert out.shape == (2, 3)

    def test_single_sample(self):
        model = TransformerClassifier(n_features=27)
        x = torch.randn(1, 60, 27)
        out = model(x)
        assert out.shape == (1, 3)

    def test_odd_d_model(self):
        """홀수 d_model에서도 crash 없이 동작."""
        model = TransformerClassifier(n_features=27, d_model=65, nhead=5)
        x = torch.randn(2, 10, 27)
        out = model(x)
        assert out.shape == (2, 3)

    def test_eval_mode_no_error(self):
        model = TransformerClassifier(n_features=27)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 10, 27))
        assert out.shape == (1, 3)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch 미설치")
class TestPositionalEncoding:
    def test_output_shape(self):
        pe = PositionalEncoding(d_model=64)
        x = torch.randn(2, 10, 64)
        out = pe(x)
        assert out.shape == x.shape

    def test_odd_d_model(self):
        pe = PositionalEncoding(d_model=65)
        x = torch.randn(1, 5, 65)
        out = pe(x)
        assert out.shape == x.shape
