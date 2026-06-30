"""AbstractEngine 단위 테스트.

추상 메서드(initialize/shutdown/run)는 구체 엔진 통합 테스트에서 다룬다.
여기서는 추상 거부, TF union, 캔들 SL/TP 체결 판정만 검증한다.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.core.engine_base import (
    AbstractEngine,
    signal_side_to_position_side,
)
from src.core.enums import (
    ExitReason,
    PositionSide,
    PositionStatus,
    SignalSide,
)
from src.core.types import Position
from tests.strategy_stub import StubStrategy
from src.strategy.registry import (
    register_strategy,
    reset_registry_for_testing,
)


# 테스트용 더미 구체 엔진 (추상 메서드만 채움 — 비즈니스 로직 미사용)
class _ConcreteEngine(AbstractEngine):
    async def initialize(self) -> None: ...
    async def shutdown(self) -> None: ...
    async def run(self) -> None: ...
    async def _record_trade_open(self, *args, **kwargs) -> int: return 0
    async def _record_trade_close(self, *args, **kwargs) -> None: return None


_STRATEGY_PARAMS_DEFAULT = {"risk_per_trade_pct": 0.01, "max_leverage": 5}


def _config_with_active(active: list[str], **strategy_params) -> dict:
    cfg = {
        "exchange": {"symbol": "BTC/USDT:USDT"},
        "database": {"path": "data/test.db"},
        "strategies": {"active": active},
    }
    # 활성 전략 각각에 기본 param 주입 (startup 검증 통과용)
    for name in active:
        cfg[name] = dict(_STRATEGY_PARAMS_DEFAULT)
    cfg.update(strategy_params)
    return cfg


@pytest.fixture(autouse=True)
def _isolated_registry():
    reset_registry_for_testing()
    yield
    reset_registry_for_testing()


class _FastSlowStrategy(StubStrategy):
    name = "fast_strategy"
    entry_timeframe = "1m"
    required_timeframes = ["1m", "15m"]

    def generate_signal(self, ctx): return None  # type: ignore
    def compute_stop_loss(self, ctx, signal): return 0.0
    def compute_take_profit(self, ctx, signal, sl): return 0.0


class _SlowStrategy(StubStrategy):
    name = "slow_strategy"
    entry_timeframe = "4h"
    required_timeframes = ["1d", "4h"]

    def generate_signal(self, ctx): return None  # type: ignore
    def compute_stop_loss(self, ctx, signal): return 0.0
    def compute_take_profit(self, ctx, signal, sl): return 0.0


class TestAbstractEngine:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            AbstractEngine({}, "paper")  # type: ignore

    def test_concrete_engine_constructs(self):
        register_strategy(_FastSlowStrategy)
        cfg = _config_with_active(
            ["fast_strategy"],
            fast_strategy={"risk_per_trade_pct": 0.01, "max_leverage": 5},
        )
        eng = _ConcreteEngine(cfg, mode="paper")
        assert eng.mode == "paper"
        assert eng.position is None
        assert eng.strategies[0].name == "fast_strategy"


class TestTimeframeUnion:
    def test_single_strategy(self):
        register_strategy(_FastSlowStrategy)
        cfg = _config_with_active(["fast_strategy"])
        eng = _ConcreteEngine(cfg, mode="paper")
        assert eng.timeframes == ["1m", "15m"]
        assert eng.master_timeframe == "1m"

    def test_multiple_strategies_union_sorted(self):
        register_strategy(_FastSlowStrategy)
        register_strategy(_SlowStrategy)
        cfg = _config_with_active(["fast_strategy", "slow_strategy"])
        eng = _ConcreteEngine(cfg, mode="paper")
        # 합집합: 1m, 15m, 4h, 1d (오름차순)
        assert eng.timeframes == ["1m", "15m", "4h", "1d"]
        assert eng.master_timeframe == "1m"  # 가장 작은 TF가 마스터

    def test_only_slow_strategy(self):
        register_strategy(_SlowStrategy)
        cfg = _config_with_active(["slow_strategy"])
        eng = _ConcreteEngine(cfg, mode="paper")
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
    """캔들 high/low로 SL/TP 체결 판정. 한 캔들 내 둘 다 도달 시 SL 우선."""

    def setup_method(self):
        register_strategy(_FastSlowStrategy)
        cfg = _config_with_active(["fast_strategy"])
        self.eng = _ConcreteEngine(cfg, mode="paper")

    def test_long_sl_only_hit(self):
        pos = _make_position(PositionSide.LONG, 67000, sl=66500, tp=68000)
        result = self.eng.check_candle_sl_tp(pos, candle_high=67800, candle_low=66400)
        assert result == (66500, ExitReason.SL_HIT)

    def test_long_tp_only_hit(self):
        pos = _make_position(PositionSide.LONG, 67000, sl=66500, tp=68000)
        result = self.eng.check_candle_sl_tp(pos, candle_high=68100, candle_low=66700)
        assert result == (68000, ExitReason.TP_HIT)

    def test_long_both_hit_sl_wins(self):
        """SL과 TP 모두 한 캔들 내 도달 시 SL 우선."""
        pos = _make_position(PositionSide.LONG, 67000, sl=66500, tp=68000)
        result = self.eng.check_candle_sl_tp(pos, candle_high=68100, candle_low=66400)
        assert result == (66500, ExitReason.SL_HIT)

    def test_short_sl_only_hit(self):
        pos = _make_position(PositionSide.SHORT, 67000, sl=67500, tp=66000)
        result = self.eng.check_candle_sl_tp(pos, candle_high=67600, candle_low=66800)
        assert result == (67500, ExitReason.SL_HIT)

    def test_short_tp_only_hit(self):
        pos = _make_position(PositionSide.SHORT, 67000, sl=67500, tp=66000)
        result = self.eng.check_candle_sl_tp(pos, candle_high=67200, candle_low=65900)
        assert result == (66000, ExitReason.TP_HIT)

    def test_short_both_hit_sl_wins(self):
        pos = _make_position(PositionSide.SHORT, 67000, sl=67500, tp=66000)
        result = self.eng.check_candle_sl_tp(pos, candle_high=67600, candle_low=65900)
        assert result == (67500, ExitReason.SL_HIT)

    def test_no_hit_returns_none(self):
        pos = _make_position(PositionSide.LONG, 67000, sl=66500, tp=68000)
        assert self.eng.check_candle_sl_tp(pos, 67800, 66600) is None

    def test_position_with_no_sl_tp_returns_none(self):
        pos = _make_position(PositionSide.LONG, 67000, sl=None, tp=None)
        assert self.eng.check_candle_sl_tp(pos, 67800, 66600) is None

    def test_none_position_returns_none(self):
        assert self.eng.check_candle_sl_tp(None, 67800, 66600) is None  # type: ignore


class _DynamicExitStrategy(StubStrategy):
    """update_stop_loss/update_take_profit 가 설정된 값을 반환하도록 오버라이드."""

    name = "dyn_strategy"
    entry_timeframe = "1m"
    required_timeframes = ["1m"]
    sl_return: float | None = None
    tp_return: float | None = None

    def update_stop_loss(self, ctx, position):
        return self.sl_return

    def update_take_profit(self, ctx, position):
        return self.tp_return

    def should_force_exit(self, ctx, position):
        return None


class TestDynamicStopTakeUpdate:
    """check_strategy_exits 가 update_stop_loss/update_take_profit 반환값으로
    position.stop_loss/take_profit 를 갱신하고, 그 값이 다음 봉 check_candle_sl_tp
    체결에 반영되는지 검증 (트레일링 SL · 이동 중심선 TP 메커니즘 = D2 update_take_profit).
    """

    def _engine_with_position(self, side, sl, tp, sl_ret, tp_ret):
        register_strategy(_DynamicExitStrategy)
        cfg = _config_with_active(["dyn_strategy"])
        eng = _ConcreteEngine(cfg, mode="paper")
        eng.strategies[0].sl_return = sl_ret
        eng.strategies[0].tp_return = tp_ret
        pos = _make_position(side, 67000, sl, tp)
        pos.strategy_name = "dyn_strategy"
        eng._position = pos
        return eng

    def _now(self):
        return datetime.now(timezone.utc)

    def test_update_take_profit_updates_position(self):
        eng = self._engine_with_position(
            PositionSide.LONG, sl=66500, tp=68000, sl_ret=None, tp_ret=67500,
        )
        eng.check_strategy_exits({}, 67000, 10000.0, self._now())
        assert eng.position.take_profit == 67500

    def test_update_take_profit_none_keeps_existing(self):
        eng = self._engine_with_position(
            PositionSide.LONG, sl=66500, tp=68000, sl_ret=None, tp_ret=None,
        )
        eng.check_strategy_exits({}, 67000, 10000.0, self._now())
        assert eng.position.take_profit == 68000

    def test_update_stop_loss_updates_position(self):
        eng = self._engine_with_position(
            PositionSide.LONG, sl=66500, tp=68000, sl_ret=66800, tp_ret=None,
        )
        eng.check_strategy_exits({}, 67000, 10000.0, self._now())
        assert eng.position.stop_loss == 66800

    def test_dynamic_tp_then_fills_at_new_value(self):
        # 이동 중심선 TP: 68000→67500 으로 당기면 high=67600 봉이 67500 에 체결
        eng = self._engine_with_position(
            PositionSide.LONG, sl=66500, tp=68000, sl_ret=None, tp_ret=67500,
        )
        eng.check_strategy_exits({}, 67000, 10000.0, self._now())
        result = eng.check_candle_sl_tp(eng.position, candle_high=67600, candle_low=66900)
        assert result == (67500, ExitReason.TP_HIT)

    def test_trailing_sl_then_fills_at_new_value(self):
        # 트레일링 SL: 66500→66800 으로 올리면 low=66700 봉이 66800 에 체결 (I-002 ③)
        eng = self._engine_with_position(
            PositionSide.LONG, sl=66500, tp=68000, sl_ret=66800, tp_ret=None,
        )
        eng.check_strategy_exits({}, 67000, 10000.0, self._now())
        result = eng.check_candle_sl_tp(eng.position, candle_high=67200, candle_low=66700)
        assert result == (66800, ExitReason.SL_HIT)


