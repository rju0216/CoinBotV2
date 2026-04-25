"""Phase 0 공통 인프라 단위 테스트.

features.py, label_generator.py, walk_forward.py, models.py 검증.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.strategy.features import (
    BASE_FEATURE_NAMES,
    compute_features,
    compute_multi_tf_features,
    get_feature_names,
)
from src.ml.label_generator import generate_direction_labels
from src.ml.walk_forward import (
    WalkForwardFold,
    apply_embargo,
    generate_walk_forward_splits,
)


# ─── 합성 캔들 헬퍼 ───


def _make_candles(n: int = 300, start_price: float = 67000.0) -> pd.DataFrame:
    """합성 OHLCV DataFrame 생성."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = start_price + np.cumsum(rng.normal(0, 50, n))
    high = close + rng.uniform(10, 100, n)
    low = close - rng.uniform(10, 100, n)
    open_ = close + rng.normal(0, 30, n)
    volume = rng.uniform(100, 1000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# ─── features.py 테스트 ───


class TestComputeFeatures:
    def test_column_count(self):
        df = _make_candles(300)
        feat = compute_features(df)
        assert feat.shape[1] == 27
        assert list(feat.columns) == BASE_FEATURE_NAMES

    def test_index_preserved(self):
        df = _make_candles(300)
        feat = compute_features(df)
        assert len(feat) == len(df)
        assert feat.index.equals(df.index)

    def test_nan_in_early_rows(self):
        """EMA200 등 긴 기간 지표로 인해 초기 행은 NaN."""
        df = _make_candles(300)
        feat = compute_features(df)
        assert feat.iloc[0].isna().any()

    def test_valid_rows_after_warmup(self):
        """충분한 워밍업 후 NaN 없는 행 존재."""
        df = _make_candles(300)
        feat = compute_features(df)
        valid = feat.dropna()
        assert len(valid) > 0

    def test_short_data_no_crash(self):
        """데이터가 짧아도 crash하지 않음 (NaN만 나옴)."""
        df = _make_candles(10)
        feat = compute_features(df)
        assert feat.shape[1] == 27


class TestMultiTfFeatures:
    def test_multi_tf_column_count(self):
        candles = {
            "15m": _make_candles(300),
            "1h": _make_candles(75),
        }
        feat = compute_multi_tf_features(candles, entry_tf="15m")
        # 15m: 27 + 1h: 27 = 54
        assert feat.shape[1] == 54

    def test_multi_tf_prefix(self):
        candles = {
            "15m": _make_candles(300),
            "1h": _make_candles(75),
        }
        feat = compute_multi_tf_features(candles, entry_tf="15m")
        prefixed = [c for c in feat.columns if c.startswith("1h_")]
        assert len(prefixed) == 27

    def test_entry_tf_only(self):
        """entry_tf만 있으면 단일 TF와 동일."""
        candles = {"15m": _make_candles(300)}
        feat = compute_multi_tf_features(candles, entry_tf="15m")
        assert feat.shape[1] == 27


class TestGetFeatureNames:
    def test_single_tf(self):
        names = get_feature_names("15m")
        assert len(names) == 27

    def test_multi_tf(self):
        names = get_feature_names("15m", ["1h", "4h"])
        assert len(names) == 27 * 3  # 15m + 1h + 4h

    def test_no_duplicate_entry_tf(self):
        """entry_tf가 extra에 포함되어도 중복 안 됨."""
        names = get_feature_names("15m", ["15m", "1h"])
        assert len(names) == 27 * 2  # 15m + 1h


# ─── label_generator.py 테스트 ───


class TestLabelGenerator:
    def test_label_values(self):
        df = _make_candles(100)
        labels = generate_direction_labels(df, horizon=5, threshold_pct=0.1)
        unique = set(labels.unique())
        assert unique.issubset({-1, 0, 1, 2})

    def test_last_horizon_rows_are_negative(self):
        """마지막 horizon개 행은 미래 데이터 없으므로 -1."""
        df = _make_candles(100)
        labels = generate_direction_labels(df, horizon=5)
        assert (labels.iloc[-5:] == -1).all()

    def test_label_distribution(self):
        """HOLD가 아닌 라벨이 존재."""
        df = _make_candles(300)
        labels = generate_direction_labels(df, horizon=10, threshold_pct=0.1)
        valid = labels[labels >= 0]
        assert (valid == 0).sum() > 0  # SHORT 존재
        assert (valid == 2).sum() > 0  # LONG 존재

    def test_all_hold_with_extreme_threshold(self):
        """극단적 threshold면 거의 전부 HOLD."""
        df = _make_candles(300)
        labels = generate_direction_labels(df, horizon=5, threshold_pct=100.0)
        valid = labels[labels >= 0]
        assert (valid == 1).sum() == len(valid)


# ─── walk_forward.py 테스트 ───


class TestWalkForward:
    def _make_index(self, months: int = 24) -> pd.DatetimeIndex:
        return pd.date_range("2022-01-01", periods=months * 30 * 24 * 4, freq="15min", tz="UTC")

    def test_fold_count(self):
        idx = self._make_index(24)  # 2년
        folds = generate_walk_forward_splits(
            idx, train_months=6, test_months=2, step_months=2
        )
        assert len(folds) > 0

    def test_no_test_overlap(self):
        """Test 구간이 겹치지 않음 (경계 시점 공유는 허용)."""
        idx = self._make_index(36)
        folds = generate_walk_forward_splits(
            idx, train_months=6, test_months=2, step_months=2
        )
        for i in range(len(folds) - 1):
            assert folds[i].test_end <= folds[i + 1].test_start

    def test_train_before_test(self):
        """모든 fold에서 train이 test보다 앞."""
        idx = self._make_index(24)
        folds = generate_walk_forward_splits(idx)
        for fold in folds:
            assert fold.train_end < fold.test_start

    def test_short_data_empty(self):
        """데이터가 너무 짧으면 fold 0개."""
        idx = pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC")
        folds = generate_walk_forward_splits(idx, train_months=6, test_months=2)
        assert len(folds) == 0

    def test_fold_id_sequential(self):
        idx = self._make_index(24)
        folds = generate_walk_forward_splits(idx)
        for i, fold in enumerate(folds):
            assert fold.fold_id == i


class TestApplyEmbargo:
    def test_removes_tail(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC")
        result = apply_embargo(idx, embargo_bars=10)
        assert len(result) == 90

    def test_zero_embargo(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC")
        result = apply_embargo(idx, embargo_bars=0)
        assert len(result) == 100

    def test_embargo_larger_than_data(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC")
        result = apply_embargo(idx, embargo_bars=10)
        assert len(result) == 5  # 원본 유지


# ─── models.py 테스트 ───

try:
    import torch
    from src.ml.models import LSTMClassifier, TransformerClassifier, PositionalEncoding
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
