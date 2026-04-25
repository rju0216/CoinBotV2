"""src/ml/label_generator.py 단위 테스트.

generate_direction_labels (3-class 방향 레이블) 검증.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.label_generator import generate_direction_labels


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
