"""백테 결과 정합성 검증 (수수료 반영).

핵심 invariants:
  - initial_balance + sum(trades.pnl) == equity_curve 마지막 balance
  - sum(trades.pnl) == metrics.json total_pnl
  - fees 증가 시 final_balance 단조 감소
데이터 단위 정합성(trades ↔ metrics ↔ equity_curve)을 먼저 검증한다.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.core.enums import SignalSide
from src.core.types import Signal
from src.strategy.registry import register_strategy, reset_registry_for_testing
from tests.strategy_stub import StubStrategy


class _ReenterLongStrategy(StubStrategy):
    """슬롯이 비면 매번 LONG 진입, SL=-0.5% / TP=+1% (% 기반, 지표 불필요).

    하락 추세 캔들에서 SL 이 반복 hit → 다수 거래 생성 (수수료 민감도 확인용).
    """

    name = "reenter_long"
    entry_timeframe = "15m"
    required_timeframes = ["15m"]

    def generate_signal(self, ctx):
        return Signal(side=SignalSide.LONG)

    def compute_stop_loss(self, ctx, signal):
        return ctx.current_price * 0.995

    def compute_take_profit(self, ctx, signal, stop_loss):
        return ctx.current_price * 1.01


@pytest.fixture(autouse=True)
def _isolated_registry():
    reset_registry_for_testing()
    yield
    reset_registry_for_testing()


def _trending_down_df(n: int = 120, start: float = 67000.0, step: float = 20.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    closes = np.array([start - step * i for i in range(n)], dtype=float)
    df = pd.DataFrame(
        {
            "open": closes + 5,
            "high": closes + 10,
            "low": closes - 10,
            "close": closes,
            "volume": [1.0] * n,
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


def _make_config(taker_fee_pct: float, slippage_pct: float) -> dict:
    return {
        "exchange": {"symbol": "BTC/USDT:USDT"},
        "accounting": {
            "taker_fee_pct": taker_fee_pct,
            "slippage_pct": slippage_pct,
            "funding_enabled": False,
        },
        "backtest": {"initial_balance": 10000.0},
        "strategies": {"active": ["reenter_long"]},
        "reenter_long": {},
    }


def _run(config: dict) -> BacktestEngine:
    register_strategy(_ReenterLongStrategy)
    df = _trending_down_df()
    eng = BacktestEngine(
        config, df.index[0].to_pydatetime(), df.index[-1].to_pydatetime()
    )
    eng.inject_candles({"15m": df})
    eng.account_tracker.set_initial_balance(eng.balance)
    eng.run()
    return eng


@pytest.mark.parametrize(
    "taker_fee_pct,slippage_pct,case",
    [
        (0.0, 0.0, "zero"),
        (0.0005, 0.0, "fee_only"),
        (0.0005, 0.0005, "fee_slip"),
        (0.0, 0.001, "slip_only"),
    ],
)
def test_trades_pnl_sum_matches_equity_final(taker_fee_pct, slippage_pct, case):
    eng = _run(_make_config(taker_fee_pct, slippage_pct))
    assert eng.trades, f"{case}: 거래 발생 없음"
    initial = eng.account_tracker.initial_balance
    sum_pnl = sum(t["pnl"] for t in eng.trades)
    expected_final = initial + sum_pnl
    actual_final = eng.equity_curve[-1][1]
    assert abs(expected_final - actual_final) < 0.01, (
        f"{case}: {expected_final} != {actual_final}"
    )


def test_trades_pnl_sum_matches_metrics_total_pnl(tmp_path):
    eng = _run(_make_config(0.0005, 0.0005))
    out_dir = eng.write_reports(out_dir=tmp_path / "reports")
    trades_csv = pd.read_csv(out_dir / "trades.csv")
    assert len(trades_csv) > 0
    with open(out_dir / "metrics.json") as f:
        metrics = json.load(f)
    sum_pnl = float(trades_csv["pnl"].sum())
    assert abs(sum_pnl - float(metrics["total_pnl"])) < 0.01


def test_higher_fees_reduce_balance_monotonically():
    finals = []
    for taker_fee in [0.0, 0.0005, 0.001, 0.002]:
        eng = _run(_make_config(taker_fee, 0.0))
        assert eng.trades, "거래 발생 없음"
        finals.append(eng.equity_curve[-1][1])
    for i in range(1, len(finals)):
        assert finals[i] < finals[i - 1], f"fees 증가에도 감소 안 함: {finals}"
