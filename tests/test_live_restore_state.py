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
    # I-BLE001: _restore_daily_pnl 가 호출하는 get_daily_pnl mock (각 테스트가 override 가능)
    eng.data_store.get_daily_pnl = AsyncMock(return_value=0.0)
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

    I-BLE001: close_trade 호출 시 closed_at 인자 전달 (fetch fallback 은 now_utc.isoformat()).
    daily_pnl 은 _restore_daily_pnl 의 get_daily_pnl mock 반환값으로 설정.
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
    # I-BLE001: closed_at 인자 전달 검증 (fallback path 는 now_utc ISO)
    assert "closed_at" in call_kwargs and call_kwargs["closed_at"] is not None
    assert eng.position is None
    # I-BLE001: _restore_daily_pnl 호출 → mock get_daily_pnl 반환값 (default 0.0)
    eng.data_store.get_daily_pnl.assert_called()
    assert eng.risk_manager.daily_pnl == 0.0


@pytest.mark.asyncio
async def test_case2_same_day_exit_uses_get_daily_pnl(_isolated_registry):
    """I-BLE001 ⑥ (기존 I-BL016 갱신): case 2 의 same-day add_pnl 분기 제거 후,
    close_trade 가 closed_at=exit_ts 로 호출되고 _restore_daily_pnl 이
    get_daily_pnl 의 반환값 (DB COALESCE 쿼리 결과) 으로 메모리 daily_pnl 설정.
    """
    eng = _build_engine(active=["macross"])
    eng.data_store.get_open_trades = AsyncMock(
        return_value=[_fake_trade(id=50)]
    )
    # mock get_daily_pnl: DB 쿼리 결과 시뮬레이트 (same-day -20.0 합산)
    eng.data_store.get_daily_pnl = AsyncMock(return_value=-20.0)
    # _fetch_actual_exit mock — same-day(UTC) timestamp 반환
    now_utc = datetime.now(timezone.utc)
    same_day_ts_ms = int(now_utc.timestamp() * 1000)
    eng._fetch_actual_exit = AsyncMock(
        return_value=(66800.0, -20.0, "sl_hit", same_day_ts_ms)
    )
    await eng._restore_state()
    # close_trade 호출 + closed_at 가 same-day ISO 인지 검증
    eng.data_store.close_trade.assert_called_once()
    call_kwargs = eng.data_store.close_trade.call_args.kwargs
    expected_iso = datetime.fromtimestamp(
        same_day_ts_ms / 1000, tz=timezone.utc,
    ).isoformat()
    assert call_kwargs["closed_at"] == expected_iso
    # _restore_daily_pnl 호출 → mock 반환값으로 메모리 daily_pnl 설정
    eng.data_store.get_daily_pnl.assert_called()
    assert eng.risk_manager.daily_pnl == pytest.approx(-20.0)


@pytest.mark.asyncio
async def test_case2_different_day_exit_closed_at_is_yesterday(_isolated_registry):
    """I-BLE001: case 2 different-day 청산 시 closed_at 인자가 어제 ISO 로 전달.
    daily_pnl 은 get_daily_pnl mock 반환값 (DB 쿼리상 어제 영역 제외 → 0.0).
    """
    eng = _build_engine(active=["macross"])
    eng.data_store.get_open_trades = AsyncMock(
        return_value=[_fake_trade(id=51)]
    )
    # mock get_daily_pnl: DB COALESCE 쿼리상 어제 closed 는 오늘 합산 제외 → 0.0
    eng.data_store.get_daily_pnl = AsyncMock(return_value=0.0)
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    different_day_ts_ms = int(yesterday.timestamp() * 1000)
    eng._fetch_actual_exit = AsyncMock(
        return_value=(66800.0, -20.0, "sl_hit", different_day_ts_ms)
    )
    await eng._restore_state()
    # closed_at = 어제 ISO 검증
    call_kwargs = eng.data_store.close_trade.call_args.kwargs
    expected_iso = datetime.fromtimestamp(
        different_day_ts_ms / 1000, tz=timezone.utc,
    ).isoformat()
    assert call_kwargs["closed_at"] == expected_iso
    # daily_pnl 은 get_daily_pnl 결과 (0.0)
    assert eng.risk_manager.daily_pnl == 0.0


@pytest.mark.asyncio
async def test_case2_mixed_close_trade_called_per_trade(_isolated_registry):
    """I-BLE001: case 2 다중 trade — 각 trade 별 close_trade 호출 + 각자 closed_at 정확."""
    eng = _build_engine(active=["macross"])
    eng.data_store.get_open_trades = AsyncMock(
        return_value=[
            _fake_trade(id=60),  # 첫 호출: same-day
            _fake_trade(id=61),  # 두 번째 호출: different-day
        ]
    )
    eng.data_store.get_daily_pnl = AsyncMock(return_value=30.0)
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
    # close_trade 두 번 호출 + 각자 closed_at 정확
    assert eng.data_store.close_trade.call_count == 2
    call_args_list = eng.data_store.close_trade.call_args_list
    assert call_args_list[0].kwargs["closed_at"] == datetime.fromtimestamp(
        same_ts / 1000, tz=timezone.utc,
    ).isoformat()
    assert call_args_list[1].kwargs["closed_at"] == datetime.fromtimestamp(
        diff_ts / 1000, tz=timezone.utc,
    ).isoformat()
    # daily_pnl = mock get_daily_pnl 반환값 (DB 쿼리상 same-day 만 +30)
    assert eng.risk_manager.daily_pnl == pytest.approx(30.0)


@pytest.mark.asyncio
async def test_close_with_funding_recalibrates_daily_pnl_after_sync(_isolated_registry):
    """I-BLE001 ④ 신규: _close_with_funding 안에서 sync 후 synced_count > 0 시
    daily_pnl 이 get_daily_pnl 결과로 재정렬.
    """
    from src.core.enums import ExitReason
    eng = _build_engine(active=["macross"])
    eng.broker.is_live = True   # sync 트리거 위해 live mode mock
    eng.broker.cancel_all_orders = AsyncMock()
    eng.broker.close_position = AsyncMock(return_value=None)
    eng.broker.get_position = AsyncMock(return_value=None)
    eng.broker.get_balance = AsyncMock(return_value=5226.25)
    eng.broker.fetch_funding_history = AsyncMock(return_value=[])
    eng.config = {"exchange": {"symbol": "BTC/USDT:USDT"},
                  "accounting": {"taker_fee_pct": 0.0005}}
    # 포지션 없음 — close_with_funding 가 close_position 단순 통과 후 sync 만 진행
    eng._position = None

    # sync_all_unsynced mock — synced_count=3 반환
    import src.live.engine as live_engine_mod
    import src.live.trade_sync as trade_sync_mod
    original_sync = trade_sync_mod.sync_all_unsynced
    trade_sync_mod.sync_all_unsynced = AsyncMock(
        return_value={"synced_count": 3, "failed_count": 0, "errors": []}
    )
    # 메모리 daily_pnl 사전 설정 (엔진 추정값, 예: +50.0)
    eng.risk_manager.daily_pnl = 50.0
    # sync 후 DB 의 OKX 실값 합산 결과 — get_daily_pnl mock
    eng.data_store.get_daily_pnl = AsyncMock(return_value=48.5)

    try:
        # _close_with_funding 의 sync 영역만 단독 호출 — 직접 sync_all_unsynced 흐름 시뮬
        # close_position 영역은 None position 이라 즉시 return (no-op)
        await eng._close_with_funding(
            exit_price=80000.0, reason=ExitReason.SL_HIT,
            now=datetime.now(timezone.utc),
        )
        # daily_pnl 재정렬 검증 — get_daily_pnl 결과로 갱신
        assert eng.risk_manager.daily_pnl == pytest.approx(48.5)
    finally:
        trade_sync_mod.sync_all_unsynced = original_sync


@pytest.mark.asyncio
async def test_clean_startup_recovers_daily_pnl(_isolated_registry):
    """I-BLE001 ③ 신규: case 1 (clean startup) 종료 시점에 _restore_daily_pnl 호출.
    DB 의 오늘 누적 daily_pnl 이 메모리에 복원되는 영역 검증.
    """
    eng = _build_engine(active=["macross"])
    # 오늘 누적 +$42.50 (예: 어제 close 안 했지만 오늘 다른 close 영역)
    eng.data_store.get_daily_pnl = AsyncMock(return_value=42.50)
    await eng._restore_state()
    # case 1 (clean startup) 흐름 — position 없음
    assert eng.position is None
    # _restore_daily_pnl 호출 → daily_pnl 복원
    eng.data_store.get_daily_pnl.assert_called()
    assert eng.risk_manager.daily_pnl == pytest.approx(42.50)


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
