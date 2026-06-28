"""_on_bar_closed 청산 디스패치 — master_timeframe 가드 검증.

다중 TF(15m/1h/4h) 동시 마감 시 청산 경로(check_candle_sl_tp / check_strategy_exits)가
master_timeframe 봉에서만 실행되어야 함. 비-master_tf 봉은 같은 포지션 청산을 중복
감지하면 안 됨 (POSITION_CLOSED 중복 발행 → EXIT 알림 N개 + daily_pnl 일시 부풀림).
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.enums import PositionSide, SignalSide
from src.core.types import Position, Signal
from src.live.engine import CoreEngine
from src.strategy.base import StrategyModule  # noqa: F401
from tests.strategy_stub import StubStrategy
from src.strategy.registry import register_strategy, reset_registry_for_testing


@pytest.fixture(autouse=True)
def _isolated_registry():
    reset_registry_for_testing()
    yield
    reset_registry_for_testing()


class _MultiTFStrategy(StubStrategy):
    name = "multitf"
    entry_timeframe = "15m"
    required_timeframes = ["15m", "1h", "4h"]

    def generate_signal(self, ctx): return Signal(side=SignalSide.HOLD)
    def compute_stop_loss(self, ctx, s): return 0.0
    def compute_take_profit(self, ctx, s, sl): return 0.0


def _build_multitf_engine() -> CoreEngine:
    register_strategy(_MultiTFStrategy)
    config = {
        "exchange": {"symbol": "BTC/USDT:USDT"},
        "database": {"path": "data/test.db"},
        "paper": {"initial_balance": 10000},
        "strategies": {"active": ["multitf"]},
        "multitf": {"risk_per_trade_pct": 0.01, "max_leverage": 5},
    }
    eng = CoreEngine(config, mode="paper")

    eng.broker = MagicMock()
    eng.broker.is_live = False          # paper → _sync_unexpected_close 경로 skip
    eng.broker.executor = None          # circuit breaker 체크 skip
    eng.broker.get_balance = AsyncMock(return_value=10000.0)
    eng.broker.get_position = AsyncMock(return_value=None)

    eng.data_store = MagicMock()
    eng.data_store.append_candle = MagicMock()
    eng.data_store.get_df = MagicMock(return_value=None)
    eng.data_store.log_equity = AsyncMock()

    eng.orderbook_collector = None
    eng._circuit_breaker_open = False

    # 청산/평가 경로는 mock — 호출 여부만 검증
    eng._close_with_funding = AsyncMock()
    eng.check_strategy_exits = MagicMock(return_value=None)
    eng.evaluate_strategies_on_bar = AsyncMock()

    # SHORT 포지션: entry 67129, SL 67896(위), TP 65527(아래)
    eng._position = Position(
        side=PositionSide.SHORT,
        size=0.069,
        entry_price=67129.70,
        entry_time=datetime(2026, 6, 3, 5, 15, tzinfo=timezone.utc),
        strategy_name="multitf",
        stop_loss=67895.99,
        take_profit=65527.52,
    )
    return eng


def _tp_hit_candle(tf: str, ts_ms: int) -> dict:
    """SHORT TP(65527.52) 도달 봉 — low <= tp."""
    return {
        "timeframe": tf,
        "candle": {
            "timestamp": ts_ms,
            "open": 65800, "high": 65900, "low": 65000, "close": 65200,
            "volume": 1.0,
        },
    }


class TestIBLE010MasterTfLiquidationGuard:
    def test_master_tf_15m(self):
        assert _build_multitf_engine().master_timeframe == "15m"

    @pytest.mark.asyncio
    async def test_non_master_tf_does_not_liquidate(self):
        """비-master_tf(1h) 봉이 TP 도달해도 청산(_close_with_funding) 미호출."""
        eng = _build_multitf_engine()
        await eng._on_bar_closed(_tp_hit_candle("1h", 1_000_000))
        eng._close_with_funding.assert_not_called()
        # 비-master_tf 도 봉 평가(evaluate)는 수행
        eng.evaluate_strategies_on_bar.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_master_tf_4h_does_not_liquidate(self):
        """비-master_tf(4h) 봉도 청산 미호출 (4h 경계 중복 발행 방지)."""
        eng = _build_multitf_engine()
        await eng._on_bar_closed(_tp_hit_candle("4h", 2_000_000))
        eng._close_with_funding.assert_not_called()

    @pytest.mark.asyncio
    async def test_master_tf_liquidates_once(self):
        """master_tf(15m) 봉이 TP 도달 시 청산 1회 호출."""
        eng = _build_multitf_engine()
        await eng._on_bar_closed(_tp_hit_candle("15m", 3_000_000))
        eng._close_with_funding.assert_awaited_once()
        # tp_hit 으로 호출됐는지 (exit_price=TP, reason=TP_HIT)
        args = eng._close_with_funding.await_args.args
        assert args[0] == pytest.approx(65527.52)  # exit_price = TP
