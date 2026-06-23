"""PATH_E TF-5a: trend_donchian_exp (edge 강화 옵션) 단위 테스트."""

from __future__ import annotations

import pandas as pd

from src.core.enums import SignalSide
from src.core.types import StrategyContext
from src.strategy.plugins.trend_donchian import TrendDonchian
from src.strategy.plugins.trend_donchian_exp import TrendDonchianExp


def _df(highs, lows, closes):
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes,
         "volume": [1.0] * n},
        index=idx,
    )


def _ctx(df, price):
    return StrategyContext(
        candles={"4h": df}, current_price=price, balance=10000.0,
        position=None, is_slot_occupied=False, params={}, now=df.index[-1],
    )


class TestLongOnly:
    def test_short_filtered(self):
        df = _df([10] * 22, [5] * 22, [8] * 21 + [1])   # SHORT 돌파
        s = TrendDonchianExp(
            {"entry_period": 20, "long_only": True}).generate_signal(_ctx(df, 1.0))
        assert s.side == SignalSide.HOLD

    def test_long_passes(self):
        df = _df([10] * 22, [1] * 22, [5] * 21 + [20])  # LONG 돌파
        s = TrendDonchianExp(
            {"entry_period": 20, "long_only": True}).generate_signal(_ctx(df, 20.0))
        assert s.side == SignalSide.LONG

    def test_short_passes_when_off(self):
        df = _df([10] * 22, [5] * 22, [8] * 21 + [1])
        s = TrendDonchianExp(
            {"entry_period": 20, "long_only": False}).generate_signal(_ctx(df, 1.0))
        assert s.side == SignalSide.SHORT


class TestBaselineEquivalence:
    """옵션 off(regime none, long_only false) = baseline trend_donchian 과 동일 신호."""

    def test_long(self):
        df = _df([10] * 22, [1] * 22, [5] * 21 + [20])
        base = TrendDonchian({"entry_period": 20}).generate_signal(_ctx(df, 20.0))
        exp = TrendDonchianExp({"entry_period": 20}).generate_signal(_ctx(df, 20.0))
        assert base.side == exp.side == SignalSide.LONG

    def test_short(self):
        df = _df([10] * 22, [5] * 22, [8] * 21 + [1])
        base = TrendDonchian({"entry_period": 20}).generate_signal(_ctx(df, 1.0))
        exp = TrendDonchianExp({"entry_period": 20}).generate_signal(_ctx(df, 1.0))
        assert base.side == exp.side == SignalSide.SHORT


class TestRegimeFilter:
    def test_none_passes(self):
        df = _df([10] * 22, [1] * 22, [5] * 21 + [20])
        s = TrendDonchianExp(
            {"entry_period": 20, "regime_filter_type": "none"}).generate_signal(_ctx(df, 20.0))
        assert s.side == SignalSide.LONG

    def test_adx_high_threshold_blocks(self):
        # ADX 0~100 → threshold 999 면 항상 미통과 → HOLD (또는 NaN→HOLD)
        df = _df([10] * 22, [1] * 22, [5] * 21 + [20])
        s = TrendDonchianExp(
            {"entry_period": 20, "regime_filter_type": "adx", "regime_threshold": 999}
        ).generate_signal(_ctx(df, 20.0))
        assert s.side == SignalSide.HOLD

    def test_ma_regime_nan_blocks(self):
        # 22봉 < 200 → ema200 NaN → ma 레짐 False → HOLD (NaN 보수적 처리)
        df = _df([10] * 22, [1] * 22, [5] * 21 + [20])
        s = TrendDonchianExp(
            {"entry_period": 20, "regime_filter_type": "ma"}).generate_signal(_ctx(df, 20.0))
        assert s.side == SignalSide.HOLD
