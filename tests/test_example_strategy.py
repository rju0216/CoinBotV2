"""ExampleMACross 샘플 전략 단위 테스트 + 백테스트 통합."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.core.enums import SignalSide
from src.core.types import Signal, StrategyContext
from src.strategy.plugins.example import ExampleMACross
from src.strategy.registry import (
    register_strategy,
    reset_registry_for_testing,
)


@pytest.fixture(autouse=True)
def _isolated_registry():
    reset_registry_for_testing()
    yield
    reset_registry_for_testing()


def _make_ctx(df: pd.DataFrame, price: float, params: dict) -> StrategyContext:
    return StrategyContext(
        candles={"15m": df},
        current_price=price,
        balance=10000.0,
        position=None,
        is_slot_occupied=False,
        params=params,
        now=datetime.now(timezone.utc),
    )


def _trending_df(
    n: int, direction: str = "up", start: float = 67000.0, step: float = 20.0
) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    prices = []
    price = start
    for _ in range(n):
        prices.append(price)
        price += step if direction == "up" else -step
    closes = np.array(prices)
    df = pd.DataFrame(
        {
            "open": closes - 5,
            "high": closes + 10,
            "low": closes - 10,
            "close": closes,
            "volume": [1.0] * n,
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


# ---- Unit tests ----


class TestGenerateSignal:
    def test_hold_when_insufficient_data(self):
        s = ExampleMACross({"risk_per_trade_pct": 0.01, "max_leverage": 5})
        df = _trending_df(n=10)  # ma_slow=50 미만
        ctx = _make_ctx(df, price=67000, params=s.params)
        assert s.generate_signal(ctx).side == SignalSide.HOLD

    def test_long_on_uptrend_cross(self):
        """하락 시퀀스(느린 EMA가 빠른 EMA보다 큰 상태) → 상승 전환 시 LONG 크로스 발생."""
        s = ExampleMACross({"risk_per_trade_pct": 0.01, "max_leverage": 5})
        # 먼저 하락 60봉, 그 다음 상승 60봉 — 상승 단계 후반부에 ema_fast가 ema_slow를 상향 크로스
        down = _trending_df(n=60, direction="down", start=67000, step=10)
        up_start = float(down["close"].iloc[-1])
        up = _trending_df(n=60, direction="up", start=up_start, step=20)
        up.index = pd.date_range(
            down.index[-1] + pd.Timedelta("15min"),
            periods=60,
            freq="15min",
            tz="UTC",
        )
        df = pd.concat([down, up])
        # 상승 단계 끝 부근에서 크로스 검색
        found_long = False
        for i in range(70, len(df)):
            sub = df.iloc[:i]
            ctx = _make_ctx(sub, price=float(sub["close"].iloc[-1]), params=s.params)
            sig = s.generate_signal(ctx)
            if sig.side == SignalSide.LONG:
                found_long = True
                break
        assert found_long, "LONG cross not detected during uptrend transition"

    def test_short_on_downtrend_cross(self):
        s = ExampleMACross({"risk_per_trade_pct": 0.01, "max_leverage": 5})
        up = _trending_df(n=60, direction="up", start=67000, step=10)
        dn_start = float(up["close"].iloc[-1])
        dn = _trending_df(n=60, direction="down", start=dn_start, step=20)
        dn.index = pd.date_range(
            up.index[-1] + pd.Timedelta("15min"),
            periods=60,
            freq="15min",
            tz="UTC",
        )
        df = pd.concat([up, dn])
        found_short = False
        for i in range(70, len(df)):
            sub = df.iloc[:i]
            ctx = _make_ctx(sub, price=float(sub["close"].iloc[-1]), params=s.params)
            sig = s.generate_signal(ctx)
            if sig.side == SignalSide.SHORT:
                found_short = True
                break
        assert found_short, "SHORT cross not detected during downtrend transition"


class TestStopLossTakeProfit:
    def test_sl_below_for_long(self):
        s = ExampleMACross(
            {
                "risk_per_trade_pct": 0.01,
                "max_leverage": 5,
                "atr_period": 14,
                "atr_sl_mult": 1.5,
            }
        )
        df = _trending_df(n=60, direction="up")
        ctx = _make_ctx(df, price=67000, params=s.params)
        sig = Signal(side=SignalSide.LONG)
        sl = s.compute_stop_loss(ctx, sig)
        assert sl < ctx.current_price

    def test_sl_above_for_short(self):
        s = ExampleMACross(
            {
                "risk_per_trade_pct": 0.01,
                "max_leverage": 5,
                "atr_period": 14,
                "atr_sl_mult": 1.5,
            }
        )
        df = _trending_df(n=60, direction="down")
        ctx = _make_ctx(df, price=67000, params=s.params)
        sig = Signal(side=SignalSide.SHORT)
        sl = s.compute_stop_loss(ctx, sig)
        assert sl > ctx.current_price

    def test_tp_rr_ratio(self):
        s = ExampleMACross(
            {
                "risk_per_trade_pct": 0.01,
                "max_leverage": 5,
                "reward_risk_ratio": 2.0,
            }
        )
        df = _trending_df(n=60)
        ctx = _make_ctx(df, price=67000, params=s.params)
        sig = Signal(side=SignalSide.LONG)
        sl = 66000  # risk = 1000
        tp = s.compute_take_profit(ctx, sig, sl)
        assert abs(tp - (67000 + 1000 * 2.0)) < 1e-6

    def test_sl_fallback_without_atr_data(self):
        s = ExampleMACross(
            {
                "risk_per_trade_pct": 0.01,
                "max_leverage": 5,
                "atr_period": 14,
            }
        )
        df = _trending_df(n=5)  # ATR 계산 불가
        ctx = _make_ctx(df, price=67000, params=s.params)
        sig = Signal(side=SignalSide.LONG)
        sl = s.compute_stop_loss(ctx, sig)
        # 0.5% 폴백
        assert abs(sl - 67000 * 0.995) < 1e-6


# ---- 백테스트 통합 ----


def _make_bt_config(db_path: str) -> dict:
    return {
        "exchange": {"symbol": "BTC/USDT:USDT"},
        "database": {"path": db_path},
        "paper": {"initial_balance": 10000},
        "accounting": {"taker_fee_pct": 0.0005, "slippage_pct": 0.0},
        "risk": {
            "max_daily_loss_pct": 0.5,
            "max_drawdown_pct": 0.5,
            "max_position_size_btc": 1.0,
            "max_concurrent_positions": 1,
        },
        "strategies": {"active": ["example_macross"]},
        "example_macross": {
            "risk_per_trade_pct": 0.01,
            "max_leverage": 5,
            "ma_fast": 10,
            "ma_slow": 20,
            "atr_period": 14,
            "atr_sl_mult": 1.5,
            "reward_risk_ratio": 2.0,
        },
    }


@pytest.mark.asyncio
async def test_example_strategy_backtest_integration(tmp_path):
    """샘플 전략이 백테 엔진에 실제로 로드되고 실행되는지."""
    register_strategy(ExampleMACross)

    config = _make_bt_config(str(tmp_path / "bt.db"))

    # 하락 후 상승하는 캔들 → 크로스 이벤트 발생
    down = _trending_df(n=40, direction="down", start=67000, step=15)
    up_start = float(down["close"].iloc[-1])
    up = _trending_df(n=100, direction="up", start=up_start, step=20)
    up.index = pd.date_range(
        down.index[-1] + pd.Timedelta("15min"),
        periods=100,
        freq="15min",
        tz="UTC",
    )
    df15 = pd.concat([down, up])

    start = df15.index[0].to_pydatetime()
    end = df15.index[-1].to_pydatetime()
    eng = BacktestEngine(config, start=start, end=end)
    eng.inject_candles({"15m": df15})

    await eng.broker.initialize()
    bal = await eng.broker.get_balance()
    eng.risk_manager.set_initial_balance(bal)

    await eng.run()
    result = await eng.get_result()
    await eng.shutdown()

    # 전략이 최소 한 번은 평가·진입되었어야 함
    assert result.num_trades >= 1, "Strategy should have generated at least one trade"
    # 모든 거래는 strategy_name이 example_macross
    for t in result.trades:
        assert t["strategy_name"] == "example_macross"
        assert t["status"] == "closed"
