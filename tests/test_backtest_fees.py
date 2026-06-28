"""백테 결과 정합성 검증.

핵심 invariants:
  - sum(trades.csv["pnl"]) == metrics.json["integrated"]["total_pnl"]
  - initial_balance + sum(trades.pnl) == equity_curve 마지막 balance
  - fees(taker+slippage)가 balance/equity에 정확히 반영

데이터 단위 정합성(trades.csv ↔ metrics.json ↔ equity_curve)을 먼저 검증한다.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.core.enums import SignalSide
from src.core.types import Signal, StrategyContext
from src.strategy.helpers.sizing import risk_based_size
from src.strategy.indicators import compute_atr, compute_ema
from src.strategy.registry import register_strategy, reset_registry_for_testing
from tests.strategy_stub import StubStrategy


class _MACrossTestStrategy(StubStrategy):
    """테스트 전용 최소 전략 (15m EMA 크로스 + ATR SL/TP).

    뼈대에는 plugin 이 없으므로, fee 정합성 검증에 필요한 거래를 생성하기 위해
    테스트 내부에 인라인 정의한다 (src/strategy/plugins 에 등록하지 않음).
    """

    name = "macross_test"
    entry_timeframe = "15m"
    required_timeframes = ["15m"]

    def _df(self, ctx: StrategyContext) -> pd.DataFrame:
        return ctx.candles.get(self.entry_timeframe, pd.DataFrame())

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        df = self._df(ctx)
        ma_fast_p = int(self.params.get("ma_fast", 20))
        ma_slow_p = int(self.params.get("ma_slow", 50))
        if len(df) < ma_slow_p + 2:
            return Signal(side=SignalSide.HOLD)
        ma_fast = compute_ema(df, ma_fast_p)
        ma_slow = compute_ema(df, ma_slow_p)
        prev_diff = float(ma_fast.iloc[-2] - ma_slow.iloc[-2])
        curr_diff = float(ma_fast.iloc[-1] - ma_slow.iloc[-1])
        if prev_diff <= 0 and curr_diff > 0:
            return Signal(side=SignalSide.LONG)
        if prev_diff >= 0 and curr_diff < 0:
            return Signal(side=SignalSide.SHORT)
        return Signal(side=SignalSide.HOLD)

    def compute_stop_loss(self, ctx: StrategyContext, signal: Signal) -> float:
        df = self._df(ctx)
        atr_period = int(self.params.get("atr_period", 14))
        atr_mult = float(self.params.get("atr_sl_mult", 1.5))
        if len(df) < atr_period + 1:
            return (
                ctx.current_price * 0.995
                if signal.side == SignalSide.LONG
                else ctx.current_price * 1.005
            )
        atr_value = float(compute_atr(df, atr_period).iloc[-1])
        if signal.side == SignalSide.LONG:
            return ctx.current_price - atr_value * atr_mult
        return ctx.current_price + atr_value * atr_mult

    def compute_take_profit(
        self, ctx: StrategyContext, signal: Signal, stop_loss: float
    ) -> float:
        rr = float(self.params.get("reward_risk_ratio", 2.0))
        risk = abs(ctx.current_price - stop_loss)
        if signal.side == SignalSide.LONG:
            return ctx.current_price + risk * rr
        return ctx.current_price - risk * rr

    def compute_position_size(self, ctx, signal, stop_loss) -> float:
        return risk_based_size(
            ctx.current_price, stop_loss, ctx.balance,
            risk_per_trade_pct=float(self.params.get("risk_per_trade_pct", 0.01)),
            max_leverage=float(self.params.get("max_leverage", 5)),
            max_position_size=1.0,
        )


@pytest.fixture(autouse=True)
def _isolated_registry():
    reset_registry_for_testing()
    yield
    reset_registry_for_testing()


def _trending_df(n: int, direction: str = "up", start: float = 67000.0, step: float = 20.0) -> pd.DataFrame:
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


def _make_bt_config(db_path: str, taker_fee_pct: float, slippage_pct: float) -> dict:
    return {
        "exchange": {"name": "okx", "symbol": "BTC/USDT:USDT", "sandbox": False, "leverage": 5},
        "accounting": {
            "taker_fee_pct": taker_fee_pct,
            "slippage_pct": slippage_pct,
            "funding_enabled": False,
        },
        "paper": {"initial_balance": 10000.0},
        "data": {"history_bars": 300, "candle_dir": "data/candles"},
        "database": {"path": db_path},
        "logging": {"level": "INFO", "file": "logs/test.log", "max_size_mb": 1, "backup_count": 1},
        "strategies": {"active": ["macross_test"]},
        "macross_test": {
            "risk_per_trade_pct": 0.01,
            "max_leverage": 5,
            "ma_fast": 10,
            "ma_slow": 20,
            "atr_period": 14,
            "atr_sl_mult": 1.5,
            "reward_risk_ratio": 2.0,
        },
    }


async def _run_backtest(config: dict) -> tuple:
    """백테 실행 → (initial_balance, trades, equity_curve) 반환."""
    register_strategy(_MACrossTestStrategy)
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
    eng.account_tracker.set_initial_balance(bal)
    await eng.run()
    result = await eng.get_result()
    await eng.shutdown()
    return result.initial_balance, result.trades, result.equity_curve


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "taker_fee_pct,slippage_pct,case_name",
    [
        (0.0, 0.0, "zero_fees"),
        (0.0005, 0.0, "default_fee_no_slip"),
        (0.0005, 0.0005, "default_fee_with_slip"),
        (0.0, 0.001, "high_slip_no_fee"),
    ],
)
async def test_trades_pnl_sum_matches_equity_curve_final(
    tmp_path, taker_fee_pct: float, slippage_pct: float, case_name: str,
):
    """invariant 1: initial + sum(trades.pnl) == equity_curve 마지막 balance.

    여러 fee/slippage 조합에서 검증. 실패 시 paper_executor가 fees를
    balance에 미반영 (또는 이중 반영) 의미.
    """
    config = _make_bt_config(
        str(tmp_path / f"bt_{case_name}.db"),
        taker_fee_pct=taker_fee_pct,
        slippage_pct=slippage_pct,
    )
    initial, trades, equity_curve = await _run_backtest(config)
    if not trades:
        pytest.skip(f"{case_name}: 거래 발생 없음 — 정합성 검증 불가")

    sum_pnl = sum(t["pnl"] for t in trades)
    expected_final = initial + sum_pnl
    actual_final = equity_curve[-1][1]
    assert abs(expected_final - actual_final) < 0.01, (
        f"{case_name}: initial({initial}) + sum_pnl({sum_pnl}) = {expected_final} "
        f"!= equity_curve final({actual_final}). 차이={expected_final - actual_final}"
    )


@pytest.mark.asyncio
async def test_trades_pnl_sum_matches_metrics_total_pnl(tmp_path):
    """invariant 2: sum(trades.pnl) == metrics.json total_pnl.

    write_reports 후 디스크 파일 일치성 검증.
    """
    config = _make_bt_config(
        str(tmp_path / "bt.db"), taker_fee_pct=0.0005, slippage_pct=0.0005
    )
    register_strategy(_MACrossTestStrategy)
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
    eng.account_tracker.set_initial_balance(bal)
    await eng.run()
    out_dir = eng.write_reports(out_root=str(tmp_path / "reports"))
    await eng.shutdown()

    trades_csv = pd.read_csv(out_dir / "trades.csv")
    if len(trades_csv) == 0:
        pytest.skip("거래 발생 없음 — 정합성 검증 불가")

    with open(out_dir / "metrics.json") as f:
        metrics = json.load(f)

    sum_pnl = float(trades_csv["pnl"].sum())
    metrics_total_pnl = float(metrics["integrated"]["total_pnl"])

    assert abs(sum_pnl - metrics_total_pnl) < 0.01, (
        f"sum(trades.pnl)={sum_pnl} != metrics.total_pnl={metrics_total_pnl}. "
        f"차이={sum_pnl - metrics_total_pnl}"
    )


@pytest.mark.asyncio
async def test_higher_fees_reduce_balance_monotonically(tmp_path):
    """invariant 3: fees 증가 시 final_balance가 단조 감소 (gross PnL 동일 가정).

    같은 캔들·전략에서 taker_fee만 늘려도 결과가 동일하면 fees가 미반영된 것.
    """
    finals = []
    for taker_fee in [0.0, 0.0005, 0.001, 0.002]:
        config = _make_bt_config(
            str(tmp_path / f"bt_{taker_fee}.db"),
            taker_fee_pct=taker_fee,
            slippage_pct=0.0,
        )
        initial, trades, equity_curve = await _run_backtest(config)
        if not trades:
            pytest.skip("거래 발생 없음")
        finals.append(equity_curve[-1][1])

    # taker_fee 증가 → final_balance 감소 (단조)
    for i in range(1, len(finals)):
        assert finals[i] < finals[i - 1], (
            f"taker_fee 증가에도 final_balance가 감소 안 함: {finals}. "
            f"fees가 balance에 미반영 의심"
        )
