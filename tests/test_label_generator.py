"""src/ml/label_generator.py 단위 테스트.

- generate_direction_labels: 3-class 방향 레이블 검증
- generate_triple_barrier_labels: BP-3-3 Triple-barrier 검증
- build_labels_from_config: train_cfg 분기 helper 검증
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.label_generator import (
    build_labels_from_config,
    generate_direction_labels,
    generate_triple_barrier_labels,
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


# ─── BP-3-3: Triple-barrier ───


def _candles_with_path(closes: list[float], highs: list[float] | None = None,
                      lows: list[float] | None = None) -> pd.DataFrame:
    """원하는 high/low/close 시퀀스를 직접 지정하는 합성 캔들."""
    n = len(closes)
    if highs is None:
        highs = [c + 1.0 for c in closes]
    if lows is None:
        lows = [c - 1.0 for c in closes]
    dates = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1.0] * n,
        },
        index=dates,
    )


class TestTripleBarrierLabels:
    def test_upper_hit_long(self):
        """1봉 후 high가 +1% 도달 → LONG."""
        # entry=100, next bar high=101.5 (>= +1%), low=99.5 (> -1%)
        df = _candles_with_path(
            closes=[100.0, 100.5, 100.5],
            highs=[100.5, 101.5, 100.7],
            lows=[99.8, 99.5, 100.3],
        )
        labels = generate_triple_barrier_labels(
            df, upper_pct=1.0, lower_pct=1.0, time_barrier_bars=2,
        )
        assert labels.iloc[0] == 2  # LONG (upper hit)

    def test_lower_hit_short(self):
        """1봉 후 low가 -1% 도달 → SHORT."""
        df = _candles_with_path(
            closes=[100.0, 99.5, 99.5],
            highs=[100.5, 100.0, 99.7],
            lows=[99.8, 98.5, 99.3],  # 98.5 < 99 (-1%)
        )
        labels = generate_triple_barrier_labels(
            df, upper_pct=1.0, lower_pct=1.0, time_barrier_bars=2,
        )
        assert labels.iloc[0] == 0  # SHORT (lower hit)

    def test_timeout_when_neither_hit(self):
        """barrier 도달 못하고 time 만료 → TIMEOUT (1)."""
        df = _candles_with_path(
            closes=[100.0] + [100.2] * 5,
            highs=[100.5] * 6,  # 모두 +1% 미달
            lows=[99.5] * 6,    # 모두 -1% 미달
        )
        labels = generate_triple_barrier_labels(
            df, upper_pct=1.0, lower_pct=1.0, time_barrier_bars=3,
        )
        assert labels.iloc[0] == 1  # TIMEOUT

    def test_simultaneous_hit_lower_priority(self):
        """동시 hit (high+low 모두 도달) → 사안 Y 가: Lower 우선 → SHORT."""
        df = _candles_with_path(
            closes=[100.0, 100.0, 100.0],
            highs=[100.5, 102.0, 100.7],  # +2% (upper hit)
            lows=[99.8, 98.0, 99.3],      # -2% (lower hit) — 같은 봉에서 둘 다
        )
        labels = generate_triple_barrier_labels(
            df, upper_pct=1.0, lower_pct=1.0, time_barrier_bars=2,
        )
        assert labels.iloc[0] == 0  # SHORT (Lower 우선)

    def test_last_rows_have_no_future(self):
        """마지막 time_barrier_bars 행은 미래 없음 → -1."""
        df = _candles_with_path(
            closes=[100.0] * 10,
        )
        labels = generate_triple_barrier_labels(
            df, upper_pct=1.0, lower_pct=1.0, time_barrier_bars=3,
        )
        # 마지막 3 행은 -1 (10 - 3 = 7번째까지만 라벨)
        assert (labels.iloc[-3:] == -1).all()

    def test_label_value_range(self):
        df = _candles_with_path(closes=[100.0 + i * 0.1 for i in range(50)])
        labels = generate_triple_barrier_labels(
            df, upper_pct=1.0, lower_pct=1.0, time_barrier_bars=10,
        )
        assert set(labels.unique()).issubset({-1, 0, 1, 2})


# ─── build_labels_from_config helper ───


class TestBuildLabelsFromConfig:
    def test_default_method_is_direction(self):
        df = _make_candles(100)
        labels, params, eff = build_labels_from_config(df, train_cfg={})
        assert params["method"] == "direction"
        assert params["horizon"] == 10
        assert params["threshold_pct"] == 0.3
        assert eff == 10

    def test_direction_passes_horizon_threshold(self):
        df = _make_candles(100)
        cfg = {"label_method": "direction", "horizon": 5, "threshold_pct": 0.5}
        labels, params, eff = build_labels_from_config(df, cfg)
        assert params == {"method": "direction", "horizon": 5, "threshold_pct": 0.5}
        assert eff == 5

    def test_triple_barrier_passes_options(self):
        df = _make_candles(100)
        cfg = {
            "label_method": "triple_barrier",
            "triple_barrier": {
                "upper_pct": 2.0,
                "lower_pct": 0.5,
                "time_barrier_bars": 20,
            },
        }
        labels, params, eff = build_labels_from_config(df, cfg)
        assert params == {
            "method": "triple_barrier",
            "upper_pct": 2.0,
            "lower_pct": 0.5,
            "time_barrier_bars": 20,
        }
        assert eff == 20

    def test_triple_barrier_default_options(self):
        df = _make_candles(100)
        cfg = {"label_method": "triple_barrier"}
        labels, params, eff = build_labels_from_config(df, cfg)
        assert params["upper_pct"] == 1.0
        assert params["lower_pct"] == 1.0
        assert eff == 10

    def test_unknown_method_raises(self):
        df = _make_candles(100)
        with pytest.raises(ValueError):
            build_labels_from_config(df, {"label_method": "unknown"})
