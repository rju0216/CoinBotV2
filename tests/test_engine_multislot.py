"""N-트랜치 용량 확장 테스트 (Phase 6 (라), `backtest.max_slots`).

검증 축:
- 기본값 1 = 기존 단일슬롯 동작 (2번째 진입 거부)
- max_slots=N 이면 트랜치 N개까지 축적, 초과분 거부
- 트랜치별 **독립 진입가·SL/TP** — 하나가 청산돼도 나머지 유지
- 미실현·mark-to-market 자본이 **트랜치 합산**
- `is_slot_occupied` = 용량 소진 여부 (N=1 이면 기존 의미)
- max_slots>1 + allow_reverse = **하드페일**(조용한 동작 차이 금지)
합성 캔들 + StubStrategy 파생 — 실데이터 비의존.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.core.enums import PositionSide, SignalSide
from src.core.types import Signal
from src.strategy.registry import register_strategy, reset_registry_for_testing
from tests.strategy_stub import StubStrategy


@pytest.fixture(autouse=True)
def _isolated_registry():
    reset_registry_for_testing()
    yield
    reset_registry_for_testing()


def _candles(n=40, start=67_000.0, drift=0.0, spread=5.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    rows, price = [], start
    for _ in range(n):
        o = price
        c = price + drift
        rows.append([o, max(o, c) + spread, min(o, c) - spread, c, 1.0])
        price = c
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)
    df.index.name = "timestamp"
    return df


def _cfg(active, max_slots=None, balance=10_000.0) -> dict:
    bt = {"initial_balance": balance}
    if max_slots is not None:
        bt["max_slots"] = max_slots
    cfg = {
        "exchange": {"symbol": "BTC/USDT:USDT"},
        "accounting": {"taker_fee_pct": 0.0005, "slippage_pct": 0.0},
        "backtest": bt,
        "strategies": {"active": active},
    }
    for name in active:
        cfg[name] = {}
    return cfg


class _AlwaysLong(StubStrategy):
    """매 1m 봉마다 LONG 신호. SL/TP 는 진입가 대비 ±아주 넓게(청산 안 됨)."""

    name = "always_long"
    entry_timeframe = "1m"
    required_timeframes = ["1m"]

    def generate_signal(self, ctx):
        return Signal(side=SignalSide.LONG)

    def compute_stop_loss(self, ctx, signal):
        return ctx.current_price * 0.5

    def compute_take_profit(self, ctx, signal, sl):
        return ctx.current_price * 2.0

    def compute_position_size(self, ctx, signal, stop_loss):
        return 0.01


def _run(eng: BacktestEngine) -> None:
    eng.account_tracker.set_initial_balance(eng.balance)
    eng.run()


# ---- 용량 ----

def test_default_is_single_slot():
    """max_slots 미지정 → 1. 매 봉 신호가 와도 트랜치는 1개뿐."""
    register_strategy(_AlwaysLong)
    eng = BacktestEngine(_cfg(["always_long"]), "2024-01-01", "2024-01-02")
    eng.inject_candles({"1m": _candles(n=10)})
    eng.account_tracker.set_initial_balance(eng.balance)
    assert eng.max_slots == 1
    # run 대신 수동 루프: 종료 강제청산 전 상태를 본다
    for ts, c in eng.candles_per_tf["1m"].iterrows():
        eng.evaluate_strategies_on_bar("1m", eng._slice_candles(ts), float(c["open"]),
                                       eng.balance, ts.to_pydatetime())
    assert len(eng.positions) == 1
    assert eng.has_capacity is False


def test_max_slots_accumulates_tranches():
    register_strategy(_AlwaysLong)
    eng = BacktestEngine(_cfg(["always_long"], max_slots=3), "2024-01-01", "2024-01-02")
    eng.inject_candles({"1m": _candles(n=10)})
    eng.account_tracker.set_initial_balance(eng.balance)
    for ts, c in eng.candles_per_tf["1m"].iterrows():
        eng.evaluate_strategies_on_bar("1m", eng._slice_candles(ts), float(c["open"]),
                                       eng.balance, ts.to_pydatetime())
    assert len(eng.positions) == 3          # 용량까지만
    assert eng.has_capacity is False
    # 봉당 최대 1 트랜치 → 진입 시각이 서로 다름
    assert len({p.entry_time for p in eng.positions}) == 3


def test_tranches_have_independent_entry_prices():
    """상승 드리프트에서 트랜치마다 진입가가 다르다(평균단가로 뭉개지지 않음)."""
    register_strategy(_AlwaysLong)
    eng = BacktestEngine(_cfg(["always_long"], max_slots=3), "2024-01-01", "2024-01-02")
    eng.inject_candles({"1m": _candles(n=10, drift=+50)})
    eng.account_tracker.set_initial_balance(eng.balance)
    for ts, c in eng.candles_per_tf["1m"].iterrows():
        eng.evaluate_strategies_on_bar("1m", eng._slice_candles(ts), float(c["open"]),
                                       eng.balance, ts.to_pydatetime())
    entries = [p.entry_price for p in eng.positions]
    assert len(set(entries)) == 3
    assert entries == sorted(entries)       # 드리프트 방향대로


def test_one_tranche_exit_keeps_others():
    """SL 이 트랜치별로 독립 발동 — 하나 청산돼도 나머지는 유지."""
    register_strategy(_AlwaysLong)
    eng = BacktestEngine(_cfg(["always_long"], max_slots=3), "2024-01-01", "2024-01-02")
    eng.inject_candles({"1m": _candles(n=6)})
    eng.account_tracker.set_initial_balance(eng.balance)
    for ts, c in eng.candles_per_tf["1m"].iterrows():
        eng.evaluate_strategies_on_bar("1m", eng._slice_candles(ts), float(c["open"]),
                                       eng.balance, ts.to_pydatetime())
    assert len(eng.positions) == 3
    victim = eng.positions[1]
    victim.stop_loss = victim.entry_price * 1.5      # 즉시 hit 되도록 조작(LONG SL 위)
    fill = eng.check_candle_sl_tp(victim, victim.entry_price * 1.6, victim.entry_price)
    assert fill is not None
    eng.close_position(victim, fill[0], fill[1], now=victim.entry_time)
    assert len(eng.positions) == 2
    assert all(p is not victim for p in eng.positions)
    assert len(eng.trades) == 1


# ---- 계측 ----

def test_unrealized_and_mtm_sum_over_tranches():
    register_strategy(_AlwaysLong)
    eng = BacktestEngine(_cfg(["always_long"], max_slots=2), "2024-01-01", "2024-01-02")
    eng.inject_candles({"1m": _candles(n=5)})
    eng.account_tracker.set_initial_balance(eng.balance)
    for ts, c in eng.candles_per_tf["1m"].iterrows():
        eng.evaluate_strategies_on_bar("1m", eng._slice_candles(ts), float(c["open"]),
                                       eng.balance, ts.to_pydatetime())
    assert len(eng.positions) == 2
    px = 68_000.0
    expected = sum((px - p.entry_price) * p.size for p in eng.positions)
    assert eng._unrealized(px) == pytest.approx(expected)
    assert eng._mark_to_market(px) == pytest.approx(eng.balance + expected)


def test_mtm_curve_recorded_and_equity_curve_untouched():
    """equity_curve(실현) 는 기존 그대로, mtm 곡선이 별도로 추가된다."""
    register_strategy(_AlwaysLong)
    eng = BacktestEngine(_cfg(["always_long"], max_slots=2), "2024-01-01", "2024-01-02")
    eng.inject_candles({"1m": _candles(n=8, drift=+50)})
    _run(eng)
    assert len(eng.equity_curve) == len(eng.equity_curve_mtm)
    assert all(isinstance(v, float) for _, v in eng.equity_curve_mtm)
    # 미실현 이익 구간에서는 mtm > 실현잔고 인 시점이 존재해야 한다
    assert any(m > e for (_, e), (_, m) in zip(eng.equity_curve, eng.equity_curve_mtm))


def test_is_slot_occupied_means_no_capacity():
    register_strategy(_AlwaysLong)
    eng = BacktestEngine(_cfg(["always_long"], max_slots=2), "2024-01-01", "2024-01-02")
    eng.inject_candles({"1m": _candles(n=5)})
    eng.account_tracker.set_initial_balance(eng.balance)
    strat = eng.strategies[0]
    ts = eng.candles_per_tf["1m"].index[0]
    ctx = eng._build_ctx(strat, eng._slice_candles(ts), 67_000.0, eng.balance,
                         ts.to_pydatetime())
    assert ctx.is_slot_occupied is False                  # 0/2
    for t, c in eng.candles_per_tf["1m"].iterrows():
        eng.evaluate_strategies_on_bar("1m", eng._slice_candles(t), float(c["open"]),
                                       eng.balance, t.to_pydatetime())
    ctx2 = eng._build_ctx(strat, eng._slice_candles(ts), 67_000.0, eng.balance,
                          ts.to_pydatetime())
    assert ctx2.is_slot_occupied is True                  # 2/2
    assert ctx2.position is eng.positions[-1]             # 최신 자기 트랜치 바인딩


# ---- 가드 ----

def test_reverse_with_multislot_hard_fails():
    """max_slots>1 + allow_reverse 조합은 구성 시 하드페일(조용한 무시 금지)."""

    class _Reversing(_AlwaysLong):
        name = "reversing"

        def __init__(self, params):
            super().__init__(params)
            self.allow_reverse = True

    register_strategy(_Reversing)
    with pytest.raises(ValueError, match="allow_reverse"):
        BacktestEngine(_cfg(["reversing"], max_slots=3), "2024-01-01", "2024-01-02")
    # 단일슬롯에서는 정상 구성
    BacktestEngine(_cfg(["reversing"], max_slots=1), "2024-01-01", "2024-01-02")


def test_multiple_strategies_each_open_one_tranche_per_bar():
    """★앙상블 가드★ 용량이 남으면 같은 봉에서 여러 전략이 각자 1 트랜치를 연다."""

    class _LongA(_AlwaysLong):
        name = "long_a"

    class _LongB(_AlwaysLong):
        name = "long_b"

    register_strategy(_LongA)
    register_strategy(_LongB)
    eng = BacktestEngine(_cfg(["long_a", "long_b"], max_slots=4), "2024-01-01", "2024-01-02")
    eng.inject_candles({"1m": _candles(n=3)})
    eng.account_tracker.set_initial_balance(eng.balance)
    ts = eng.candles_per_tf["1m"].index[0]
    eng.evaluate_strategies_on_bar("1m", eng._slice_candles(ts), 67_000.0,
                                   eng.balance, ts.to_pydatetime())
    assert len(eng.positions) == 2                      # 한 봉에 두 전략 각 1개
    assert {p.strategy_name for p in eng.positions} == {"long_a", "long_b"}


def test_single_slot_still_admits_only_one_strategy():
    """max_slots=1 에서는 첫 전략이 용량을 소진 → 두 번째는 진입 불가(기존 동작 보존)."""

    class _LongA(_AlwaysLong):
        name = "long_a"

    class _LongB(_AlwaysLong):
        name = "long_b"

    register_strategy(_LongA)
    register_strategy(_LongB)
    eng = BacktestEngine(_cfg(["long_a", "long_b"], max_slots=1), "2024-01-01", "2024-01-02")
    eng.inject_candles({"1m": _candles(n=3)})
    eng.account_tracker.set_initial_balance(eng.balance)
    ts = eng.candles_per_tf["1m"].index[0]
    eng.evaluate_strategies_on_bar("1m", eng._slice_candles(ts), 67_000.0,
                                   eng.balance, ts.to_pydatetime())
    assert len(eng.positions) == 1
    assert eng.positions[0].strategy_name == "long_a"   # active 리스트 순서 = 우선순위


def test_horizon_ensemble_aliases_registered():
    """지평 슬리브 3종이 서로 다른 이름으로 등록되고 EconL3 로직을 상속한다."""
    from src.strategy.plugins.econ_l3 import EconL3
    from src.strategy.plugins.econ_l3_ensemble import EconL3H3, EconL3H4, EconL3H5

    names = {EconL3H3.name, EconL3H4.name, EconL3H5.name}
    assert names == {"econ_l3_h3", "econ_l3_h4", "econ_l3_h5"}
    for cls in (EconL3H3, EconL3H4, EconL3H5):
        assert issubclass(cls, EconL3)
        # 로직 오버라이드 없음 — 이름만 분리
        assert cls.generate_signal is EconL3.generate_signal


def test_invalid_max_slots_rejected():
    register_strategy(_AlwaysLong)
    with pytest.raises(ValueError, match="max_slots"):
        BacktestEngine(_cfg(["always_long"], max_slots=0), "2024-01-01", "2024-01-02")


def test_shutdown_closes_all_tranches():
    register_strategy(_AlwaysLong)
    eng = BacktestEngine(_cfg(["always_long"], max_slots=3), "2024-01-01", "2024-01-02")
    eng.inject_candles({"1m": _candles(n=8)})
    _run(eng)
    assert len(eng.positions) == 0
    assert len(eng.trades) == 3
    assert all(t["exit_reason"] == "engine_shutdown" for t in eng.trades)
    # 손익 정합: 초기잔고 + Σpnl == 최종잔고
    assert eng.balance == pytest.approx(10_000.0 + sum(t["pnl"] for t in eng.trades))
