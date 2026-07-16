"""Lookahead bias 회귀 방지 — 엔진 _slice_candles 인과성.

백테 엔진의 lookahead 차단은 `BacktestEngine._slice_candles(ts)` 가 `df.index < ts`
로 진행 중 봉을 배제하는 데 있다. ts 시점에 그 봉은 아직 마감 전이므로 close 가
피처에 새면 안 된다. 이 규율이 지켜지는지 합성 캔들로 검증한다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.core.enums import SignalSide
from src.core.types import Signal
from src.strategy.registry import register_strategy, reset_registry_for_testing
from tests.strategy_stub import StubStrategy


@pytest.fixture(autouse=True)
def _isolated_registry():
    reset_registry_for_testing()
    yield
    reset_registry_for_testing()


def _candles(n: int = 20) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    closes = [67000.0 + i for i in range(n)]
    df = pd.DataFrame(
        {
            "open": closes,
            "high": [c + 5 for c in closes],
            "low": [c - 5 for c in closes],
            "close": closes,
            "volume": [1.0] * n,
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


def _config() -> dict:
    return {"strategies": {"active": []}}


def test_slice_searchsorted_equivalent_to_boolean_mask():
    """D-031: searchsorted 슬라이스가 부울마스크(df.index < ts)와 결과 동일 (동작 보존).

    경계: ts 가 봉 정각·봉 사이·전체 이전/이후. 인덱스 정렬 전제(엔진 계약)에서 등가.
    """
    eng = BacktestEngine(_config(), "2024-01-01", "2024-01-02")
    df = _candles(20)
    eng.inject_candles({"15m": df})
    ts_cases = [
        df.index[0],                              # 첫 봉 정각 → 앞이 빔
        df.index[10],                             # 중간 봉 정각
        df.index[-1],                             # 마지막 봉 정각
        df.index[5] + pd.Timedelta(minutes=7),    # 봉 사이
        df.index[0] - pd.Timedelta(minutes=1),    # 전체 이전 → 빔
        df.index[-1] + pd.Timedelta(minutes=1),   # 전체 이후 → 전부
    ]
    for ts in ts_cases:
        got = eng._slice_candles(ts)["15m"]
        expected = df[df.index < ts]
        pd.testing.assert_frame_equal(got, expected)


def test_slice_excludes_current_and_future_bars():
    eng = BacktestEngine(_config(), "2024-01-01", "2024-01-02")
    df = _candles(20)
    eng.inject_candles({"15m": df})

    ts = df.index[10]
    sliced = eng._slice_candles(ts)["15m"]
    # ts '미만' 만 포함 — ts 봉과 그 이후는 배제
    assert (sliced.index < ts).all()
    assert len(sliced) == 10
    # 마지막 포함 봉은 직전 마감 봉 (index[9])
    assert sliced.index[-1] == df.index[9]


def test_ctx_last_bar_is_prior_closed_bar_during_signal():
    """generate_signal 이 보는 ctx.candles[tf].iloc[-1] 이 항상 직전 마감 봉인지."""
    seen_last_close: list[float] = []
    bar_open_prices: list[float] = []

    class _RecorderStrategy(StubStrategy):
        name = "recorder"
        entry_timeframe = "15m"
        required_timeframes = ["15m"]

        def generate_signal(self, ctx):
            df = ctx.candles["15m"]
            if not df.empty:
                seen_last_close.append(float(df["close"].iloc[-1]))
                bar_open_prices.append(ctx.current_price)
            return Signal(side=SignalSide.HOLD)

    register_strategy(_RecorderStrategy)
    df = _candles(12)
    eng = BacktestEngine(
        {"strategies": {"active": ["recorder"]}, "recorder": {}},
        df.index[0].to_pydatetime(),
        df.index[-1].to_pydatetime(),
    )
    eng.inject_candles({"15m": df})
    eng.account_tracker.set_initial_balance(eng.balance)
    eng.run()

    assert seen_last_close, "generate_signal 이 호출되지 않음"
    # 매 평가에서 직전 마감 봉 close < 현재 봉 open (단조 증가 시리즈이므로)
    for last_close, bar_open in zip(seen_last_close, bar_open_prices):
        assert last_close < bar_open
