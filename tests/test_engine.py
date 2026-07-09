"""BacktestEngine 메커니즘 단위 테스트.

네트워크(캔들 로딩) 없이 구성 가능한 부분만 검증한다:
TF union 산출, 시그널→포지션 매핑, 캔들 SL/TP 체결 판정.
end-to-end 진입·청산 흐름은 test_backtest_engine.py.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.backtest.engine import BacktestEngine, signal_side_to_position_side
from src.core.enums import ExitReason, PositionSide, PositionStatus, SignalSide
from src.core.types import Position
from src.strategy.registry import register_strategy, reset_registry_for_testing
from tests.strategy_stub import StubStrategy


@pytest.fixture(autouse=True)
def _isolated_registry():
    reset_registry_for_testing()
    yield
    reset_registry_for_testing()


def _config(active: list[str]) -> dict:
    return {"strategies": {"active": active}}


class _FastSlowStrategy(StubStrategy):
    name = "fast_strategy"
    entry_timeframe = "1m"
    required_timeframes = ["1m", "15m"]


class _SlowStrategy(StubStrategy):
    name = "slow_strategy"
    entry_timeframe = "4h"
    required_timeframes = ["1d", "4h"]


class TestConstruction:
    def test_engine_constructs(self):
        register_strategy(_FastSlowStrategy)
        eng = BacktestEngine(_config(["fast_strategy"]), "2024-01-01", "2024-01-02")
        assert eng.position is None
        assert eng.strategies[0].name == "fast_strategy"
        assert eng.balance == 10000.0  # backtest.initial_balance 기본값


class TestTimeframeUnion:
    def test_single_strategy(self):
        register_strategy(_FastSlowStrategy)
        eng = BacktestEngine(_config(["fast_strategy"]), "2024-01-01", "2024-01-02")
        assert eng.timeframes == ["1m", "15m"]
        assert eng.master_timeframe == "1m"

    def test_multiple_strategies_union_sorted(self):
        register_strategy(_FastSlowStrategy)
        register_strategy(_SlowStrategy)
        eng = BacktestEngine(
            _config(["fast_strategy", "slow_strategy"]), "2024-01-01", "2024-01-02"
        )
        assert eng.timeframes == ["1m", "15m", "4h", "1d"]
        assert eng.master_timeframe == "1m"

    def test_only_slow_strategy(self):
        register_strategy(_SlowStrategy)
        eng = BacktestEngine(_config(["slow_strategy"]), "2024-01-01", "2024-01-02")
        assert eng.timeframes == ["4h", "1d"]
        assert eng.master_timeframe == "4h"


class TestSignalSideMapping:
    def test_long_short_hold(self):
        assert signal_side_to_position_side(SignalSide.LONG) == PositionSide.LONG
        assert signal_side_to_position_side(SignalSide.SHORT) == PositionSide.SHORT
        assert signal_side_to_position_side(SignalSide.HOLD) == PositionSide.NONE


def _make_position(
    side: PositionSide, entry: float, sl: float | None, tp: float | None
) -> Position:
    return Position(
        side=side,
        size=0.1,
        entry_price=entry,
        entry_time=datetime.now(timezone.utc),
        strategy_name="test",
        stop_loss=sl,
        take_profit=tp,
        status=PositionStatus.OPEN,
    )


class TestCandleSLTPFill:
    """캔들 high/low 로 SL/TP 체결 판정. 한 캔들 내 둘 다 도달 시 SL 우선(기본)."""

    def setup_method(self):
        reset_registry_for_testing()
        register_strategy(_FastSlowStrategy)
        self.eng = BacktestEngine(
            _config(["fast_strategy"]), "2024-01-01", "2024-01-02"
        )

    def test_long_sl_only_hit(self):
        pos = _make_position(PositionSide.LONG, 67000, sl=66500, tp=68000)
        assert self.eng.check_candle_sl_tp(pos, 67800, 66400) == (
            66500, ExitReason.SL_HIT
        )

    def test_long_tp_only_hit(self):
        pos = _make_position(PositionSide.LONG, 67000, sl=66500, tp=68000)
        assert self.eng.check_candle_sl_tp(pos, 68100, 66700) == (
            68000, ExitReason.TP_HIT
        )

    def test_long_both_hit_sl_wins(self):
        pos = _make_position(PositionSide.LONG, 67000, sl=66500, tp=68000)
        assert self.eng.check_candle_sl_tp(pos, 68100, 66400) == (
            66500, ExitReason.SL_HIT
        )

    def test_short_sl_only_hit(self):
        pos = _make_position(PositionSide.SHORT, 67000, sl=67500, tp=66000)
        assert self.eng.check_candle_sl_tp(pos, 67600, 66800) == (
            67500, ExitReason.SL_HIT
        )

    def test_short_tp_only_hit(self):
        pos = _make_position(PositionSide.SHORT, 67000, sl=67500, tp=66000)
        assert self.eng.check_candle_sl_tp(pos, 67200, 65900) == (
            66000, ExitReason.TP_HIT
        )

    def test_short_both_hit_sl_wins(self):
        pos = _make_position(PositionSide.SHORT, 67000, sl=67500, tp=66000)
        assert self.eng.check_candle_sl_tp(pos, 67600, 65900) == (
            67500, ExitReason.SL_HIT
        )

    def test_no_hit_returns_none(self):
        pos = _make_position(PositionSide.LONG, 67000, sl=66500, tp=68000)
        assert self.eng.check_candle_sl_tp(pos, 67800, 66600) is None

    def test_position_with_no_sl_tp_returns_none(self):
        pos = _make_position(PositionSide.LONG, 67000, sl=None, tp=None)
        assert self.eng.check_candle_sl_tp(pos, 67800, 66600) is None

    def test_none_position_returns_none(self):
        assert self.eng.check_candle_sl_tp(None, 67800, 66600) is None  # type: ignore
