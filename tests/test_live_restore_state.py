"""CoreEngine._restore_state 5개 시나리오 테스트 (잠재 이슈 I-001/I-002).

거래소 포지션과 DB open trades의 매칭·자동 입양(7-1)·뼈대 에러 중단(7 (a))을
mock을 통해 검증한다.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.enums import PositionSide, PositionStatus, SignalSide
from src.core.types import Signal
from src.live.engine import CoreEngine
from src.strategy.base import StrategyModule
from src.strategy.registry import (
    register_strategy,
    reset_registry_for_testing,
)


@pytest.fixture(autouse=True)
def _isolated_registry():
    reset_registry_for_testing()
    yield
    reset_registry_for_testing()


class _PassthroughStrategy(StrategyModule):
    name = "macross"
    entry_timeframe = "15m"
    required_timeframes = ["15m"]

    def generate_signal(self, ctx): return Signal(side=SignalSide.HOLD)
    def compute_stop_loss(self, ctx, s): return 0.0
    def compute_take_profit(self, ctx, s, sl): return 0.0


def _make_config(active: list[str] | None = None) -> dict:
    return {
        "exchange": {"symbol": "BTC/USDT:USDT"},
        "database": {"path": "data/test.db"},
        "paper": {"initial_balance": 10000},
        "strategies": {"active": active or []},
        "macross": {"risk_per_trade_pct": 0.01, "max_leverage": 5},
    }


def _build_engine(active: list[str] | None = None) -> CoreEngine:
    """CoreEngine을 생성하되 broker/data_store를 mock으로 교체."""
    if active and "macross" in active:
        register_strategy(_PassthroughStrategy)
    eng = CoreEngine(_make_config(active), mode="paper")

    # broker/data_store mock
    eng.broker = MagicMock()
    eng.broker.is_live = False  # paper mode (I-BL013 fetch_actual_exit이 None 반환 → fallback)
    eng.broker.get_balance = AsyncMock(return_value=10000.0)
    eng.broker.get_position = AsyncMock(return_value=None)

    eng.data_store = MagicMock()
    eng.data_store.get_initial_balance = AsyncMock(return_value=10000.0)
    eng.data_store.set_initial_balance = AsyncMock()
    eng.data_store.get_peak_equity = AsyncMock(return_value=10000.0)
    eng.data_store.get_open_trades = AsyncMock(return_value=[])
    eng.data_store.close_trade = AsyncMock()
    return eng


def _fake_trade(
    *,
    id: int = 1,
    strategy_name: str = "macross",
    side: str = "long",
    size: float = 0.1,
    entry_price: float = 67000,
    sl: float = 66500,
    tp: float = 68000,
    ts: str | None = None,
) -> dict:
    return {
        "id": id,
        "strategy_name": strategy_name,
        "side": side,
        "size": size,
        "entry_price": entry_price,
        "stop_loss": sl,
        "take_profit": tp,
        "timestamp": ts or datetime.now(timezone.utc).isoformat(),
        "status": "open",
    }


@pytest.mark.asyncio
async def test_clean_startup_no_position(_isolated_registry):
    """시나리오 5: 거래소 ∅ + DB ∅ → 정상 빈 슬롯."""
    eng = _build_engine(active=["macross"])
    await eng._restore_state()
    assert eng.position is None


@pytest.mark.asyncio
async def test_db_open_but_exchange_empty_closes_stale(_isolated_registry):
    """시나리오 2: 거래소 ∅ + DB O → DB trades 사후 closed 처리.

    I-BL013 fix 후: paper 모드(is_live=False)이므로 fetch_actual_exit None 반환 →
    fallback 동작 (SL 가격 추정 + WARNING). entry=67000, SL=66500, size=0.1, long.
    pnl = (66500 - 67000) × 0.1 = -50.

    I-BL016: timestamp 부재(fetch fallback) → 보수적으로 different-day 가정,
    daily_pnl 누적 skip.
    """
    eng = _build_engine(active=["macross"])
    eng.data_store.get_open_trades = AsyncMock(
        return_value=[_fake_trade(id=42)]
    )
    await eng._restore_state()
    eng.data_store.close_trade.assert_called_once()
    call_kwargs = eng.data_store.close_trade.call_args.kwargs
    assert call_kwargs["trade_id"] == 42
    # I-BL013 fallback: SL 가격 추정 + 단순 PnL (수수료/슬리피지 누락)
    assert call_kwargs["exit_price"] == 66500
    assert call_kwargs["pnl"] == pytest.approx(-50.0)
    assert call_kwargs["exit_reason"] == "engine_shutdown"
    assert eng.position is None
    # I-BL016: fetch fallback path는 timestamp 부재 → daily_pnl 누적 안 됨
    assert eng.risk_manager.daily_pnl == 0.0


@pytest.mark.asyncio
async def test_case2_same_day_exit_accrues_daily_pnl(_isolated_registry):
    """I-BL016 (BL-2-4 hotfix-L): case 2 same-day(UTC) 청산 →
    risk_manager.daily_pnl 누적."""
    eng = _build_engine(active=["macross"])
    eng.data_store.get_open_trades = AsyncMock(
        return_value=[_fake_trade(id=50)]
    )
    # _fetch_actual_exit mock — same-day(UTC) timestamp 반환
    now_utc = datetime.now(timezone.utc)
    same_day_ts_ms = int(now_utc.timestamp() * 1000)
    eng._fetch_actual_exit = AsyncMock(
        return_value=(66800.0, -20.0, "sl_hit", same_day_ts_ms)
    )
    await eng._restore_state()
    # add_pnl 호출 → daily_pnl 누적
    assert eng.risk_manager.daily_pnl == pytest.approx(-20.0)
    # close_trade도 호출 (DB 정리는 same/different-day 무관)
    eng.data_store.close_trade.assert_called_once()


@pytest.mark.asyncio
async def test_case2_different_day_exit_skips_daily_pnl(_isolated_registry):
    """I-BL016: case 2 different-day(UTC) 청산 → daily_pnl 누적 skip
    (DB 정리는 진행)."""
    eng = _build_engine(active=["macross"])
    eng.data_store.get_open_trades = AsyncMock(
        return_value=[_fake_trade(id=51)]
    )
    # _fetch_actual_exit mock — 1일 전 timestamp 반환
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    different_day_ts_ms = int(yesterday.timestamp() * 1000)
    eng._fetch_actual_exit = AsyncMock(
        return_value=(66800.0, -20.0, "sl_hit", different_day_ts_ms)
    )
    await eng._restore_state()
    # different-day → add_pnl 호출 안 됨
    assert eng.risk_manager.daily_pnl == 0.0
    # close_trade는 호출됨 (DB 정리는 진행)
    eng.data_store.close_trade.assert_called_once()


@pytest.mark.asyncio
async def test_case2_mixed_same_and_different_day(_isolated_registry):
    """I-BL016: case 2 다중 trade 혼합 — same-day pnl 만 누적."""
    eng = _build_engine(active=["macross"])
    eng.data_store.get_open_trades = AsyncMock(
        return_value=[
            _fake_trade(id=60),  # 첫 호출: same-day
            _fake_trade(id=61),  # 두 번째 호출: different-day
        ]
    )
    now_utc = datetime.now(timezone.utc)
    same_ts = int(now_utc.timestamp() * 1000)
    diff_ts = int((now_utc - timedelta(days=1)).timestamp() * 1000)
    call_count = {"n": 0}

    async def mock_fetch(trade):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return (67100.0, 30.0, "tp_hit", same_ts)
        return (66500.0, -50.0, "sl_hit", diff_ts)

    eng._fetch_actual_exit = mock_fetch
    await eng._restore_state()
    # same-day +30 만 누적, different-day -50 은 skip
    assert eng.risk_manager.daily_pnl == pytest.approx(30.0)
    # close_trade는 두 trade 모두 호출됨
    assert eng.data_store.close_trade.call_count == 2


@pytest.mark.asyncio
async def test_skeleton_with_exchange_position_raises(_isolated_registry):
    """시나리오 3: 거래소 O + 전략 0개 → RuntimeError (정책 7 (a))."""
    eng = _build_engine(active=[])
    eng.broker.get_position = AsyncMock(
        return_value={
            "side": PositionSide.LONG,
            "size": 0.1,
            "entry_price": 67000,
        }
    )
    with pytest.raises(RuntimeError, match="no active strategies"):
        await eng._restore_state()


@pytest.mark.asyncio
async def test_matched_trade_adopts_as_open(_isolated_registry):
    """시나리오 4 (match): 거래소 O + DB O + strategy active → 정상 OPEN 복원."""
    eng = _build_engine(active=["macross"])
    eng.broker.get_position = AsyncMock(
        return_value={
            "side": PositionSide.LONG,
            "size": 0.1,
            "entry_price": 67000,
        }
    )
    eng.data_store.get_open_trades = AsyncMock(
        return_value=[_fake_trade(id=7)]
    )
    await eng._restore_state()
    assert eng.position is not None
    assert eng.position.strategy_name == "macross"
    assert eng.position.status == PositionStatus.OPEN
    assert eng.position.trade_id == 7
    assert eng.position.stop_loss == 66500
    assert eng.position.take_profit == 68000


@pytest.mark.asyncio
async def test_matched_trade_but_strategy_removed_becomes_orphan(_isolated_registry):
    """시나리오 4 (orphan): 매칭 성공하나 strategy_name이 active 리스트에 없음 → ORPHAN."""
    eng = _build_engine(active=["macross"])
    eng.broker.get_position = AsyncMock(
        return_value={
            "side": PositionSide.LONG,
            "size": 0.1,
            "entry_price": 67000,
        }
    )
    eng.data_store.get_open_trades = AsyncMock(
        return_value=[_fake_trade(id=8, strategy_name="retired_strategy")]
    )
    await eng._restore_state()
    assert eng.position is not None
    assert eng.position.strategy_name == "retired_strategy"
    assert eng.position.status == PositionStatus.ORPHAN


@pytest.mark.asyncio
async def test_exchange_position_no_db_match_becomes_unknown_orphan(_isolated_registry):
    """거래소 O + 전략 ≥1 + DB 매칭 실패 → strategy='_unknown' ORPHAN."""
    eng = _build_engine(active=["macross"])
    eng.broker.get_position = AsyncMock(
        return_value={
            "side": PositionSide.LONG,
            "size": 0.5,  # DB에 없는 size
            "entry_price": 67000,
        }
    )
    # DB는 다른 size의 open trade 보유
    eng.data_store.get_open_trades = AsyncMock(
        return_value=[_fake_trade(id=9, size=0.1)]
    )
    await eng._restore_state()
    assert eng.position is not None
    assert eng.position.strategy_name == "_unknown"
    assert eng.position.status == PositionStatus.ORPHAN
    assert eng.position.trade_id is None


@pytest.mark.asyncio
async def test_side_mismatch_does_not_match(_isolated_registry):
    """같은 size지만 side 다르면 매칭 실패 → _unknown orphan."""
    eng = _build_engine(active=["macross"])
    eng.broker.get_position = AsyncMock(
        return_value={
            "side": PositionSide.SHORT,
            "size": 0.1,
            "entry_price": 67000,
        }
    )
    eng.data_store.get_open_trades = AsyncMock(
        return_value=[_fake_trade(id=10, side="long", size=0.1)]
    )
    await eng._restore_state()
    assert eng.position.strategy_name == "_unknown"
    assert eng.position.status == PositionStatus.ORPHAN


class TestMatcher:
    def test_match_side_and_size(self):
        pos = {"side": PositionSide.LONG, "size": 0.1, "entry_price": 67000}
        trades = [_fake_trade(id=1, side="long", size=0.1)]
        assert CoreEngine._match_trade_to_exchange(trades, pos)["id"] == 1

    def test_no_match_when_side_differs(self):
        pos = {"side": PositionSide.SHORT, "size": 0.1, "entry_price": 67000}
        trades = [_fake_trade(id=1, side="long", size=0.1)]
        assert CoreEngine._match_trade_to_exchange(trades, pos) is None

    def test_no_match_when_size_differs(self):
        pos = {"side": PositionSide.LONG, "size": 0.5, "entry_price": 67000}
        trades = [_fake_trade(id=1, side="long", size=0.1)]
        assert CoreEngine._match_trade_to_exchange(trades, pos) is None

    def test_empty_trades_returns_none(self):
        pos = {"side": PositionSide.LONG, "size": 0.1, "entry_price": 67000}
        assert CoreEngine._match_trade_to_exchange([], pos) is None


class TestBarDeduplication:
    """I-005: watch_ohlcv 진행 중 봉 재발행에 대한 중복 처리 차단."""

    def setup_method(self):
        self.eng = _build_engine(active=["macross"])

    def test_new_ts_processes(self):
        assert self.eng._should_process_bar("15m", 1000) is True

    def test_same_ts_skipped(self):
        assert self.eng._should_process_bar("15m", 1000) is True
        assert self.eng._should_process_bar("15m", 1000) is False
        assert self.eng._should_process_bar("15m", 1000) is False

    def test_older_ts_skipped(self):
        assert self.eng._should_process_bar("15m", 2000) is True
        assert self.eng._should_process_bar("15m", 1500) is False

    def test_newer_ts_after_older_still_processes(self):
        assert self.eng._should_process_bar("15m", 1000) is True
        assert self.eng._should_process_bar("15m", 1000) is False  # 중복
        assert self.eng._should_process_bar("15m", 2000) is True   # 새 봉

    def test_different_timeframes_independent(self):
        assert self.eng._should_process_bar("15m", 1000) is True
        # 다른 TF는 각자의 시퀀스
        assert self.eng._should_process_bar("4h", 1000) is True
        assert self.eng._should_process_bar("1d", 999) is True
