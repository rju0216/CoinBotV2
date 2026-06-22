"""PATH_E TF-1: trend_donchian plugin + compute_donchian 단위 테스트."""

from __future__ import annotations

import pandas as pd

from src.core.enums import PositionSide, SignalSide
from src.core.types import Position, Signal, StrategyContext
from src.strategy.indicators import compute_donchian
from src.strategy.plugins.trend_donchian import TrendDonchian


def _df(highs, lows, closes):
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes,
         "volume": [1.0] * n},
        index=idx,
    )


def _ctx(df, price, position=None):
    return StrategyContext(
        candles={"4h": df}, current_price=price, balance=10000.0,
        position=position, is_slot_occupied=position is not None,
        params={}, now=df.index[-1],
    )


class TestComputeDonchian:
    def test_causal_shift_excludes_current_bar(self):
        # upper[i] = max(high[i-period .. i-1]) — 현재봉 high 제외
        df = _df(highs=[10, 11, 12, 13, 14], lows=[1, 2, 3, 4, 5],
                 closes=[5, 5, 5, 5, 5])
        ch = compute_donchian(df, period=2)
        # i=2: max(high[0], high[1]) = 11 (현재봉 high=12 제외)
        assert ch["upper"].iloc[2] == 11
        assert ch["lower"].iloc[2] == 1
        # i=4: max(high[2], high[3]) = 13 (현재봉 high=14 제외)
        assert ch["upper"].iloc[4] == 13
        assert ch["lower"].iloc[4] == 3

    def test_initial_rows_nan(self):
        df = _df([10] * 5, [1] * 5, [5] * 5)
        ch = compute_donchian(df, period=3)
        assert pd.isna(ch["upper"].iloc[0])  # shift(1) + rolling → 초기 NaN


class TestEntry:
    def test_long_breakout(self):
        closes = [5] * 21 + [20]            # 마지막 close=20 > 직전 최고 10
        df = _df([10] * 22, [1] * 22, closes)
        sig = TrendDonchian({"entry_period": 20}).generate_signal(_ctx(df, 20.0))
        assert sig.side == SignalSide.LONG

    def test_short_breakout(self):
        closes = [8] * 21 + [1]             # 마지막 close=1 < 직전 최저 5
        df = _df([10] * 22, [5] * 22, closes)
        sig = TrendDonchian({"entry_period": 20}).generate_signal(_ctx(df, 1.0))
        assert sig.side == SignalSide.SHORT

    def test_no_breakout_hold(self):
        df = _df([10] * 22, [1] * 22, [5] * 22)
        sig = TrendDonchian({"entry_period": 20}).generate_signal(_ctx(df, 5.0))
        assert sig.side == SignalSide.HOLD

    def test_insufficient_data_hold(self):
        df = _df([10] * 5, [1] * 5, [5] * 5)
        sig = TrendDonchian({"entry_period": 20}).generate_signal(_ctx(df, 5.0))
        assert sig.side == SignalSide.HOLD


class TestTrailing:
    def _pos(self, side, sl, df):
        return Position(side=side, size=1.0, entry_price=18.0,
                        entry_time=df.index[0], strategy_name="trend_donchian",
                        stop_loss=sl)

    def test_long_raises_sl(self):
        # exit_lower(직전10봉 최저=10) > 현재 SL(8) → 상향
        df = _df([20] * 12, [10] * 12, [18] * 12)
        strat = TrendDonchian({"exit_period": 10})
        pos = self._pos(PositionSide.LONG, 8.0, df)
        assert strat.update_stop_loss(_ctx(df, 18.0, pos), pos) == 10.0

    def test_long_monotonic_holds(self):
        # exit_lower(5) < 현재 SL(15) → 단조성 유지(후퇴 금지)
        df = _df([20] * 12, [5] * 12, [18] * 12)
        strat = TrendDonchian({"exit_period": 10})
        pos = self._pos(PositionSide.LONG, 15.0, df)
        assert strat.update_stop_loss(_ctx(df, 18.0, pos), pos) == 15.0

    def test_short_lowers_sl(self):
        # exit_upper(직전10봉 최고=10) < 현재 SL(12) → 하향(SHORT 유리)
        df = _df([10] * 12, [1] * 12, [5] * 12)
        strat = TrendDonchian({"exit_period": 10})
        pos = self._pos(PositionSide.SHORT, 12.0, df)
        assert strat.update_stop_loss(_ctx(df, 5.0, pos), pos) == 10.0

    def test_short_monotonic_holds(self):
        # exit_upper(20) > 현재 SL(12) → 유지
        df = _df([20] * 12, [1] * 12, [5] * 12)
        strat = TrendDonchian({"exit_period": 10})
        pos = self._pos(PositionSide.SHORT, 12.0, df)
        assert strat.update_stop_loss(_ctx(df, 5.0, pos), pos) == 12.0

    def test_insufficient_data_returns_none(self):
        df = _df([20] * 5, [10] * 5, [18] * 5)
        strat = TrendDonchian({"exit_period": 10})
        pos = self._pos(PositionSide.LONG, 8.0, df)
        assert strat.update_stop_loss(_ctx(df, 18.0, pos), pos) is None


class TestTakeProfitInactive:
    def test_tp_far_long(self):
        df = _df([20] * 22, [1] * 22, [10] * 22)
        strat = TrendDonchian({"reward_risk_ratio": 100})
        tp = strat.compute_take_profit(_ctx(df, 10.0), Signal(side=SignalSide.LONG), 8.0)
        assert tp == 10.0 + 2.0 * 100   # risk=2 → TP=210, 사실상 안 닿음

    def test_tp_far_short(self):
        df = _df([20] * 22, [1] * 22, [10] * 22)
        strat = TrendDonchian({"reward_risk_ratio": 100})
        tp = strat.compute_take_profit(_ctx(df, 10.0), Signal(side=SignalSide.SHORT), 12.0)
        assert tp == 10.0 - 2.0 * 100
