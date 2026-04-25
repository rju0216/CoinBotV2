"""src/strategy/features.py 단위 테스트.

compute_features, compute_multi_tf_features, get_feature_names 검증.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.features import (
    BASE_FEATURE_NAMES,
    compute_features,
    compute_multi_tf_features,
    get_feature_names,
)


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
