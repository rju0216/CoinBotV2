"""Strategy 레지스트리·base 인터페이스 단위 테스트."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.core.enums import SignalSide
from src.core.types import Signal, StrategyContext
from src.strategy.base import StrategyModule
from src.strategy.registry import (
    get_registered_strategies,
    get_strategy_class,
    load_active_strategies,
    register_strategy,
    reset_registry_for_testing,
)


@pytest.fixture(autouse=True)
def _isolated_registry():
    """각 테스트마다 레지스트리 격리."""
    reset_registry_for_testing()
    yield
    reset_registry_for_testing()


class _DummyStrategy(StrategyModule):
    name = "dummy"
    entry_timeframe = "15m"
    required_timeframes = ["15m"]

    def generate_signal(self, ctx):
        return Signal(side=SignalSide.HOLD)

    def compute_stop_loss(self, ctx, signal):
        return ctx.current_price * 0.99

    def compute_take_profit(self, ctx, signal, stop_loss):
        return ctx.current_price * 1.02


def _make_ctx(price: float = 67000.0, balance: float = 10000.0) -> StrategyContext:
    return StrategyContext(
        candles={"15m": pd.DataFrame()},
        current_price=price,
        balance=balance,
        position=None,
        is_slot_occupied=False,
        params={},
        now=datetime.now(timezone.utc),
    )


class TestRegistration:
    def test_register_and_lookup(self):
        register_strategy(_DummyStrategy)
        assert "dummy" in get_registered_strategies()
        assert get_strategy_class("dummy") is _DummyStrategy

    def test_register_idempotent_same_class(self):
        # 같은 클래스 두 번 등록은 OK (re-import 시나리오)
        register_strategy(_DummyStrategy)
        register_strategy(_DummyStrategy)
        assert len(get_registered_strategies()) == 1

    def test_register_duplicate_name_different_class_raises(self):
        register_strategy(_DummyStrategy)

        class _OtherDummy(StrategyModule):
            name = "dummy"
            entry_timeframe = "15m"
            required_timeframes = ["15m"]

            def generate_signal(self, ctx): return Signal(side=SignalSide.HOLD)
            def compute_stop_loss(self, ctx, s): return 0.0
            def compute_take_profit(self, ctx, s, sl): return 0.0

        with pytest.raises(ValueError, match="already registered"):
            register_strategy(_OtherDummy)

    def test_register_missing_name_raises(self):
        class _NoName(StrategyModule):
            entry_timeframe = "15m"
            required_timeframes = ["15m"]
            def generate_signal(self, ctx): return Signal(side=SignalSide.HOLD)
            def compute_stop_loss(self, ctx, s): return 0.0
            def compute_take_profit(self, ctx, s, sl): return 0.0

        with pytest.raises(TypeError, match="name"):
            register_strategy(_NoName)

    def test_register_missing_entry_timeframe_raises(self):
        class _NoTF(StrategyModule):
            name = "no_tf"
            required_timeframes = ["15m"]
            def generate_signal(self, ctx): return Signal(side=SignalSide.HOLD)
            def compute_stop_loss(self, ctx, s): return 0.0
            def compute_take_profit(self, ctx, s, sl): return 0.0

        with pytest.raises(TypeError, match="entry_timeframe"):
            register_strategy(_NoTF)


class TestLookup:
    def test_unknown_strategy_raises_keyerror(self):
        with pytest.raises(KeyError, match="not found"):
            get_strategy_class("ghost_strategy")


class TestLoadActiveStrategies:
    def test_loads_in_declared_order(self):
        register_strategy(_DummyStrategy)

        class _SecondDummy(StrategyModule):
            name = "dummy2"
            entry_timeframe = "4h"
            required_timeframes = ["4h"]
            def generate_signal(self, ctx): return Signal(side=SignalSide.HOLD)
            def compute_stop_loss(self, ctx, s): return 0.0
            def compute_take_profit(self, ctx, s, sl): return 0.0

        register_strategy(_SecondDummy)

        base = {"risk_per_trade_pct": 0.01, "max_leverage": 5}
        config = {
            "strategies": {"active": ["dummy2", "dummy"]},
            "dummy": {**base, "foo": 1},
            "dummy2": {**base, "bar": 2},
        }
        instances = load_active_strategies(config)
        # 우선순위 = 선언 순서
        assert [s.name for s in instances] == ["dummy2", "dummy"]
        assert instances[0].params["bar"] == 2
        assert instances[1].params["foo"] == 1

    def test_missing_required_param_raises(self):
        """I-008: 전략 params 필수 키 누락은 시작 시점에 감지."""
        register_strategy(_DummyStrategy)
        # risk_per_trade_pct 누락
        config = {
            "strategies": {"active": ["dummy"]},
            "dummy": {"max_leverage": 5},
        }
        with pytest.raises(ValueError, match="missing required params.*risk_per_trade_pct"):
            load_active_strategies(config)

    def test_missing_max_leverage_raises(self):
        register_strategy(_DummyStrategy)
        config = {
            "strategies": {"active": ["dummy"]},
            "dummy": {"risk_per_trade_pct": 0.01},
        }
        with pytest.raises(ValueError, match="missing required params.*max_leverage"):
            load_active_strategies(config)

    def test_empty_params_raises(self):
        register_strategy(_DummyStrategy)
        config = {"strategies": {"active": ["dummy"]}, "dummy": {}}
        with pytest.raises(ValueError, match="missing required params"):
            load_active_strategies(config)

    def test_empty_active_returns_empty(self):
        config = {"strategies": {"active": []}}
        assert load_active_strategies(config) == []

    def test_missing_strategies_section_returns_empty(self):
        assert load_active_strategies({}) == []

    def test_unknown_active_name_raises(self):
        config = {"strategies": {"active": ["does_not_exist"]}}
        with pytest.raises(KeyError):
            load_active_strategies(config)

    def test_duplicate_active_raises(self):
        register_strategy(_DummyStrategy)
        config = {
            "strategies": {"active": ["dummy", "dummy"]},
            "dummy": {"risk_per_trade_pct": 0.01, "max_leverage": 5},
        }
        with pytest.raises(ValueError, match="Duplicate"):
            load_active_strategies(config)


class TestStrategyModuleAbstract:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            StrategyModule({})  # type: ignore

    def test_default_hooks_are_noop(self):
        register_strategy(_DummyStrategy)
        s = _DummyStrategy({})
        ctx = _make_ctx()
        # 선택 훅은 모두 None 반환 (no-op)
        assert s.on_bar_close(ctx, "15m") is None
        assert s.update_stop_loss(ctx, None) is None  # type: ignore[arg-type]
        assert s.should_force_exit(ctx, None) is None  # type: ignore[arg-type]
        assert s.generate_pyramid_signal(ctx, None) is None  # type: ignore[arg-type]
        # supports_pyramiding 기본값
        assert s.supports_pyramiding is False
