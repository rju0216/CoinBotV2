"""src/ml/sequence_utils.py 단위 테스트.

make_sequences (학습용) + make_sequence_from_recent (추론용) 검증.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.sequence_utils import make_sequence_from_recent, make_sequences


def _make_features(n: int = 100, n_features: int = 5) -> pd.DataFrame:
    """합성 피처 DataFrame 생성."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.normal(0, 1, (n, n_features)),
        index=pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
        columns=[f"f{i}" for i in range(n_features)],
    )


class TestMakeSequences:
    def test_shape(self):
        feat = _make_features(100, 5)
        labels = pd.Series([1] * 100, index=feat.index)
        X, y, idx = make_sequences(feat, labels, lookback=10)
        assert X.shape == (91, 10, 5)
        assert y.shape == (91,)
        assert len(idx) == 91

    def test_dtype(self):
        feat = _make_features(50, 3)
        labels = pd.Series([0] * 50, index=feat.index)
        X, y, _ = make_sequences(feat, labels, lookback=5)
        assert X.dtype == np.float32
        assert y.dtype == np.int64

    def test_excludes_negative_labels(self):
        """음수 라벨(-1, label_generator의 horizon 끝 시점)은 자동 제외."""
        feat = _make_features(100, 5)
        labels = pd.Series([1] * 95 + [-1] * 5, index=feat.index)
        X, y, _ = make_sequences(feat, labels, lookback=10)
        # lookback=10이면 96개 시퀀스(끝점 인덱스 9~99). 끝점 95~99의 5개가 -1 라벨
        assert X.shape[0] == 91 - 5
        assert (y >= 0).all()

    def test_excludes_nan_in_sequence(self):
        """시퀀스 윈도우 안에 NaN 한 개라도 있으면 그 시퀀스 제외."""
        feat = _make_features(50, 3)
        feat.iloc[10, 0] = np.nan
        labels = pd.Series([0] * 50, index=feat.index)
        X, y, _ = make_sequences(feat, labels, lookback=5)
        # 끝점 i가 10~14인 시퀀스(5개)가 인덱스 10을 포함 → 제외
        assert X.shape[0] == 46 - 5
        assert not np.isnan(X).any()

    def test_short_data_returns_empty(self):
        feat = _make_features(5, 3)
        labels = pd.Series([0] * 5, index=feat.index)
        X, y, idx = make_sequences(feat, labels, lookback=10)
        assert X.shape == (0, 10, 3)
        assert y.shape == (0,)
        assert len(idx) == 0

    def test_label_alignment(self):
        """시퀀스 i의 라벨 = features 인덱스 i의 라벨."""
        feat = _make_features(20, 3)
        labels = pd.Series(range(20), index=feat.index, dtype=int)
        X, y, idx = make_sequences(feat, labels, lookback=5)
        # 끝점 인덱스: 4~19 → 라벨 4~19
        assert list(y) == list(range(4, 20))
        assert list(idx) == list(feat.index[4:])


class TestMakeSequenceFromRecent:
    def test_shape(self):
        feat = _make_features(100, 5)
        seq = make_sequence_from_recent(feat, lookback=60)
        assert seq is not None
        assert seq.shape == (1, 60, 5)
        assert seq.dtype == np.float32

    def test_short_data_returns_none(self):
        feat = _make_features(30, 5)
        seq = make_sequence_from_recent(feat, lookback=60)
        assert seq is None

    def test_nan_returns_none(self):
        feat = _make_features(100, 5)
        feat.iloc[-10, 0] = np.nan
        seq = make_sequence_from_recent(feat, lookback=60)
        assert seq is None

    def test_uses_last_rows(self):
        """마지막 lookback개 행을 정확히 사용."""
        feat = _make_features(100, 5)
        seq = make_sequence_from_recent(feat, lookback=10)
        np.testing.assert_array_almost_equal(
            seq[0], feat.iloc[-10:].values.astype(np.float32)
        )
