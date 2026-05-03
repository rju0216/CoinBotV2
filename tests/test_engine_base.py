"""AbstractEngine 단위 테스트 (단계 8).

추상 메서드(initialize/shutdown/run)는 단계 9/10에서 구체 엔진과 통합 테스트.
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
from src.strategy.base import StrategyModule
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
    # 활성 전략 각각에 필수 키 default 주입 (I-008 startup 검증 통과용)
    for name in active:
        cfg[name] = dict(_STRATEGY_PARAMS_DEFAULT)
    cfg.update(strategy_params)
    return cfg


@pytest.fixture(autouse=True)
def _isolated_registry():
    reset_registry_for_testing()
    yield
    reset_registry_for_testing()


class _FastSlowStrategy(StrategyModule):
    name = "fast_strategy"
    entry_timeframe = "1m"
    required_timeframes = ["1m", "15m"]

    def generate_signal(self, ctx): return None  # type: ignore
    def compute_stop_loss(self, ctx, signal): return 0.0
    def compute_take_profit(self, ctx, signal, sl): return 0.0


class _SlowStrategy(StrategyModule):
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
    """캔들 high/low로 SL/TP 체결 판정. SL 우선 (정책 (a))."""

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
        """SL과 TP 모두 한 캔들 내 도달 시 SL 우선 (정책 (a))."""
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


# ---- BP-2-2 동적 사이징 helper ----


class _ATRStrategy(StrategyModule):
    name = "atr_strategy"
    entry_timeframe = "15m"
    required_timeframes = ["15m"]

    def generate_signal(self, ctx): return None  # type: ignore
    def compute_stop_loss(self, ctx, signal): return 0.0
    def compute_take_profit(self, ctx, signal, sl): return 0.0


def _make_ctx_with_candles(candles_15m):
    """단순 ctx mock — _compute_volatility_factor가 사용하는 필드만."""
    from src.core.types import StrategyContext
    return StrategyContext(
        candles={"15m": candles_15m},
        current_price=float(candles_15m["close"].iloc[-1]),
        balance=10000.0,
        position=None,
        is_slot_occupied=False,
        params={},
        now=datetime(2024, 1, 1, tzinfo=timezone.utc),
        precomputed_features=None,
    )


def _synthetic_15m_candles(n: int, atr_pct: float, base_price: float = 67000.0):
    """일정한 ATR_pct를 갖는 합성 15m 캔들 (high-low = base × atr_pct)."""
    import pandas as pd
    timestamps = pd.date_range(
        "2024-01-01", periods=n, freq="15min", tz="UTC"
    )
    range_ = base_price * atr_pct
    rows = []
    for _ in range(n):
        o = base_price
        c = base_price
        h = base_price + range_ / 2
        low = base_price - range_ / 2
        rows.append([o, h, low, c, 1.0])
    df = pd.DataFrame(
        rows, columns=["open", "high", "low", "close", "volume"], index=timestamps
    )
    df.index.name = "timestamp"
    return df


class TestVolatilityFactorHelper:
    def test_disabled_returns_one(self):
        register_strategy(_ATRStrategy)
        cfg = _config_with_active(["atr_strategy"])
        cfg["risk"] = {"volatility_targeting": {"enabled": False}}
        eng = _ConcreteEngine(cfg, mode="backtest")
        strategy = next(s for s in eng.strategies if s.name == "atr_strategy")
        ctx = _make_ctx_with_candles(_synthetic_15m_candles(30, atr_pct=0.01))
        assert eng._compute_volatility_factor(strategy, ctx) == 1.0

    def test_enabled_factor_above_one_when_volatile(self):
        register_strategy(_ATRStrategy)
        cfg = _config_with_active(["atr_strategy"])
        cfg["risk"] = {
            "volatility_targeting": {
                "enabled": True,
                "target_atr_pct": 0.005,
                "lookback": 14,
            }
        }
        eng = _ConcreteEngine(cfg, mode="backtest")
        strategy = next(s for s in eng.strategies if s.name == "atr_strategy")
        # ATR_pct = 0.01 (target 0.005의 2배) → factor ≈ 2.0
        ctx = _make_ctx_with_candles(_synthetic_15m_candles(30, atr_pct=0.01))
        factor = eng._compute_volatility_factor(strategy, ctx)
        assert factor == pytest.approx(2.0, rel=0.05)

    def test_enabled_factor_below_one_when_calm(self):
        register_strategy(_ATRStrategy)
        cfg = _config_with_active(["atr_strategy"])
        cfg["risk"] = {
            "volatility_targeting": {
                "enabled": True,
                "target_atr_pct": 0.005,
                "lookback": 14,
            }
        }
        eng = _ConcreteEngine(cfg, mode="backtest")
        strategy = next(s for s in eng.strategies if s.name == "atr_strategy")
        # ATR_pct = 0.0025 → factor ≈ 0.5
        ctx = _make_ctx_with_candles(_synthetic_15m_candles(30, atr_pct=0.0025))
        factor = eng._compute_volatility_factor(strategy, ctx)
        assert factor == pytest.approx(0.5, rel=0.10)

    def test_insufficient_candles_returns_one(self):
        register_strategy(_ATRStrategy)
        cfg = _config_with_active(["atr_strategy"])
        cfg["risk"] = {
            "volatility_targeting": {"enabled": True, "lookback": 14}
        }
        eng = _ConcreteEngine(cfg, mode="backtest")
        strategy = next(s for s in eng.strategies if s.name == "atr_strategy")
        # 5봉만 (lookback 14보다 적음) → fallback 1.0
        ctx = _make_ctx_with_candles(_synthetic_15m_candles(5, atr_pct=0.01))
        assert eng._compute_volatility_factor(strategy, ctx) == 1.0
