"""src/ml/walk_forward.py 단위 테스트.

generate_walk_forward_splits (시간 기반 롤링 분할) + apply_embargo 검증.
"""

from __future__ import annotations

import pandas as pd

from src.ml.walk_forward import (
    WalkForwardFold,
    apply_embargo,
    generate_walk_forward_splits,
)


class TestWalkForward:
    def _make_index(self, months: int = 24) -> pd.DatetimeIndex:
        return pd.date_range(
            "2022-01-01", periods=months * 30 * 24 * 4, freq="15min", tz="UTC"
        )

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
