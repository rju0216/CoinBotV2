"""TradeSyncer 단위 테스트 (BLE-6-1 + I-BLE002 + I-BLE004 + I-BLE007).

I-BLE007 영역: OKX positions-history 영역 직접 사용 영역. pnl, funding_fee, entry/exit_price,
size, trading_fee, closed_at 영역 영역 모두 OKX positions-history 영역 영역. fetch_my_trades
영역 영역 entry/exit_order_id 영역 매칭용.

핵심 검증:
- paper 가드
- positions-history 영역 직접 사용 (pnl, funding_fee 등)
- 매칭 영역 (closed_at vs uTime ±15분 + side + size)
- 매칭 실패 영역 진단 로그 (I-BLE004)
- pagination 정상 (fills)
- order_id 매칭 (DB 기존 값 우선 + fills fallback)
"""

from __future__ import annotations

import logging
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.live.trade_sync import sync_all_unsynced

CONTRACT_SIZE = 0.01  # BTC/USDT:USDT OKX contract size


def _iso_to_ms(iso_str: str) -> int:
    return int(datetime.fromisoformat(iso_str).timestamp() * 1000)


def _make_broker(
    is_live: bool,
    fetch_my_trades_return=None,
    fetch_positions_history_return=None,
    fetch_my_trades_side_effect=None,
    contract_size: float = CONTRACT_SIZE,
):
    broker = MagicMock()
    broker.is_live = is_live
    executor = MagicMock()
    executor.exchange = MagicMock()
    executor.contract_size = contract_size
    if fetch_my_trades_side_effect is not None:
        executor.exchange.fetch_my_trades = AsyncMock(side_effect=fetch_my_trades_side_effect)
    else:
        executor.exchange.fetch_my_trades = AsyncMock(
            return_value=fetch_my_trades_return or [],
        )
    executor.exchange.fetch_positions_history = AsyncMock(
        return_value=fetch_positions_history_return or [],
    )
    broker.executor = executor
    return broker


def _make_data_store(unsynced_trades):
    ds = MagicMock()
    ds.get_unsynced_trades = AsyncMock(return_value=unsynced_trades)
    ds.update_synced_trade = AsyncMock()
    return ds


def _make_position(
    *,
    side: str,
    size_btc: float,
    open_price: float,
    close_price: float,
    fee: float,
    funding: float,
    realized_pnl: float,
    close_ts_ms: int,
):
    """OKX positions-history 영역 응답 형식 mock (ccxt 영역 영역에서 info 영역 추출)."""
    return {
        "info": {
            "direction": side,                             # 'long' or 'short'
            "openAvgPx": str(open_price),
            "closeAvgPx": str(close_price),
            "closeTotalPos": str(size_btc / CONTRACT_SIZE),   # contracts
            "fee": str(-abs(fee)),                          # OKX 부호: 음수 = 비용
            "fundingFee": str(funding),                     # 부호 그대로 (양수=수익, 음수=비용)
            "realizedPnl": str(realized_pnl),
            "uTime": str(close_ts_ms),
            "posId": "test_pos_id_123",
        }
    }


@pytest.mark.asyncio
async def test_paper_mode_guard():
    """paper 모드 → ccxt 호출 안 됨, 빈 결과."""
    broker = _make_broker(is_live=False)
    ds = _make_data_store(unsynced_trades=[])
    result = await sync_all_unsynced(broker, ds, "BTC/USDT:USDT")
    assert result == {"synced_count": 0, "failed_count": 0, "errors": []}
    broker.executor.exchange.fetch_my_trades.assert_not_called()
    broker.executor.exchange.fetch_positions_history.assert_not_called()


@pytest.mark.asyncio
async def test_sync_uses_okx_realized_pnl_directly():
    """I-BLE007: pnl, funding_fee, entry/exit_price, size 등 OKX positions-history 영역 직접 사용."""
    db_trade = {
        "id": 18,
        "timestamp": "2026-05-28T07:45:01+00:00",
        "closed_at": "2026-05-28T14:05:34+00:00",
        "side": "short",
        "size": 0.127,
        "entry_order_id": "preset_entry",
        "exit_order_id": "preset_exit",
    }
    # Trade 18 영역 OKX positions-history mock — 사용자 OKX $94.32 영역 재현
    okx_position = _make_position(
        side="short",
        size_btc=0.127,
        open_price=73446.6,
        close_price=72637.5,
        fee=9.276340,
        funding=0.840955,    # 양수 (수익)
        realized_pnl=94.3203,
        close_ts_ms=_iso_to_ms("2026-05-28T14:05:34+00:00"),
    )
    broker = _make_broker(
        is_live=True,
        fetch_positions_history_return=[okx_position],
    )
    ds = _make_data_store([db_trade])
    result = await sync_all_unsynced(broker, ds, "BTC/USDT:USDT")
    assert result["synced_count"] == 1
    assert result["failed_count"] == 0

    call = ds.update_synced_trade.call_args.kwargs
    assert call["trade_id"] == 18
    assert call["pnl"] == pytest.approx(94.3203)                    # OKX realizedPnl 직접
    assert call["funding_fee"] == pytest.approx(0.840955)            # 양수 부호 그대로
    assert call["entry_price"] == pytest.approx(73446.6)              # openAvgPx
    assert call["exit_price"] == pytest.approx(72637.5)               # closeAvgPx
    assert call["trading_fee"] == pytest.approx(9.276340, abs=0.01)   # abs(fee)
    # pnl_pct = 94.32 / (73446.6 × 0.127) × 100 ≈ 1.011%
    expected_pct = 94.3203 / (73446.6 * 0.127) * 100
    assert call["pnl_pct"] == pytest.approx(expected_pct, abs=0.01)
    # entry/exit_order_id 영역 DB 기존 값 보존
    assert call["entry_order_id"] == "preset_entry"
    assert call["exit_order_id"] == "preset_exit"


@pytest.mark.asyncio
async def test_funding_negative_sign_preserved():
    """I-BLE007: funding 음수 (비용) 영역 부호 그대로 저장."""
    db_trade = {
        "id": 6,
        "timestamp": "2026-05-11T03:15:05+00:00",
        "closed_at": "2026-05-11T16:11:00+00:00",
        "side": "long",
        "size": 0.0589,
        "entry_order_id": None,
        "exit_order_id": None,
    }
    okx_position = _make_position(
        side="long",
        size_btc=0.0589,
        open_price=80570.8,
        close_price=81716.9,
        fee=4.7713,
        funding=-0.5245,    # 음수 (비용)
        realized_pnl=62.0949,
        close_ts_ms=_iso_to_ms("2026-05-11T16:11:00+00:00"),
    )
    broker = _make_broker(
        is_live=True,
        fetch_positions_history_return=[okx_position],
    )
    ds = _make_data_store([db_trade])
    result = await sync_all_unsynced(broker, ds, "BTC/USDT:USDT")
    assert result["synced_count"] == 1
    call = ds.update_synced_trade.call_args.kwargs
    # funding 음수 부호 그대로 저장
    assert call["funding_fee"] == pytest.approx(-0.5245)
    assert call["pnl"] == pytest.approx(62.0949)


@pytest.mark.asyncio
async def test_match_uses_closed_at_within_15min_window():
    """I-BLE007: 매칭 영역 ±15분 영역 — 외부 청산 시점 vs 라이브 인지 시점 차이 흡수."""
    db_trade = {
        "id": 19,
        "timestamp": "2026-05-28T18:00:00+00:00",
        "closed_at": "2026-05-28T21:45:00+00:00",   # 라이브 영역 인지 시점
        "side": "short",
        "size": 0.0988,
        "entry_order_id": None,
        "exit_order_id": None,
    }
    # OKX 영역 영역 영역 7분 차이 (SL trigger 시점)
    okx_uTime = _iso_to_ms("2026-05-28T21:37:39+00:00")
    okx_position = _make_position(
        side="short",
        size_btc=0.0988,
        open_price=73254.4,
        close_price=73789.9,
        fee=7.2566,
        funding=0.0,
        realized_pnl=-60.11,
        close_ts_ms=okx_uTime,
    )
    broker = _make_broker(
        is_live=True,
        fetch_positions_history_return=[okx_position],
    )
    ds = _make_data_store([db_trade])
    result = await sync_all_unsynced(broker, ds, "BTC/USDT:USDT")
    # ±15분 안에 있어야 매칭 성공
    assert result["synced_count"] == 1
    call = ds.update_synced_trade.call_args.kwargs
    assert call["pnl"] == pytest.approx(-60.11)


@pytest.mark.asyncio
async def test_match_failure_side_mismatch(caplog):
    """매칭 실패 — side 영역 불일치 영역 → WARNING + 진단 로그 (I-BLE004 영역)."""
    db_trade = {
        "id": 99,
        "timestamp": "2026-05-28T07:00:00+00:00",
        "closed_at": "2026-05-28T08:00:00+00:00",
        "side": "long",   # DB 영역 long
        "size": 0.1,
        "entry_order_id": None,
        "exit_order_id": None,
    }
    okx_position = _make_position(
        side="short",   # OKX 영역 short — 불일치
        size_btc=0.1,
        open_price=73000,
        close_price=72000,
        fee=4.0,
        funding=0.0,
        realized_pnl=100.0,
        close_ts_ms=_iso_to_ms("2026-05-28T08:00:00+00:00"),
    )
    broker = _make_broker(
        is_live=True,
        fetch_positions_history_return=[okx_position],
    )
    ds = _make_data_store([db_trade])
    with caplog.at_level(logging.WARNING):
        result = await sync_all_unsynced(broker, ds, "BTC/USDT:USDT")
    assert result["synced_count"] == 0
    assert result["failed_count"] == 1
    log_text = "\n".join(r.message for r in caplog.records)
    assert "매칭 실패" in log_text
    # 진단 영역 — side 무관 후보 1건, side 일치 0건
    assert "position 후보" in log_text


@pytest.mark.asyncio
async def test_match_failure_size_mismatch():
    """매칭 실패 — size 영역 불일치 영역."""
    db_trade = {
        "id": 5,
        "timestamp": "2026-05-11T01:45:02+00:00",
        "closed_at": "2026-05-11T03:00:00+00:00",
        "side": "long",
        "size": 0.0561,   # DB size
        "entry_order_id": None,
        "exit_order_id": None,
    }
    okx_position = _make_position(
        side="long",
        size_btc=0.10,   # 다른 size — tolerance 초과
        open_price=80000,
        close_price=80500,
        fee=4.0,
        funding=0.0,
        realized_pnl=50.0,
        close_ts_ms=_iso_to_ms("2026-05-11T03:00:00+00:00"),
    )
    broker = _make_broker(
        is_live=True,
        fetch_positions_history_return=[okx_position],
    )
    ds = _make_data_store([db_trade])
    result = await sync_all_unsynced(broker, ds, "BTC/USDT:USDT")
    assert result["synced_count"] == 0
    assert result["failed_count"] == 1


@pytest.mark.asyncio
async def test_order_id_fallback_from_fills():
    """I-BLE007: DB entry/exit_order_id None 영역 영역 — fills 영역 영역 추출."""
    db_trade = {
        "id": 10,
        "timestamp": "2026-05-14T14:45:02+00:00",
        "closed_at": "2026-05-14T14:50:15+00:00",
        "side": "short",
        "size": 0.075,
        "entry_order_id": None,   # DB 영역 None
        "exit_order_id": None,
    }
    db_ts_ms = _iso_to_ms(db_trade["timestamp"])
    exit_ts_ms = _iso_to_ms(db_trade["closed_at"])
    okx_position = _make_position(
        side="short",
        size_btc=0.075,
        open_price=80304.9,
        close_price=80737.7,
        fee=6.04,
        funding=0.0,
        realized_pnl=-38.50,
        close_ts_ms=exit_ts_ms,
    )
    # fills 영역 — entry order (fillPnl=0) + exit order (fillPnl≠0)
    okx_fills = [
        {
            "order": "entry_order_x", "id": "fe1",
            "timestamp": db_ts_ms + 100,
            "price": 80304.9, "amount": 7.5,   # contracts
            "side": "sell",
            "info": {"fillPnl": "0"},
            "fee": {"cost": 3.01, "currency": "USDT"},
        },
        {
            "order": "exit_order_y", "id": "fx1",
            "timestamp": exit_ts_ms + 100,
            "price": 80737.7, "amount": 7.5,
            "side": "buy",
            "info": {"fillPnl": "-32.46"},
            "fee": {"cost": 3.03, "currency": "USDT"},
        },
    ]
    broker = _make_broker(
        is_live=True,
        fetch_my_trades_return=okx_fills,
        fetch_positions_history_return=[okx_position],
    )
    ds = _make_data_store([db_trade])
    result = await sync_all_unsynced(broker, ds, "BTC/USDT:USDT")
    assert result["synced_count"] == 1
    call = ds.update_synced_trade.call_args.kwargs
    # fills 영역 영역 추출
    assert call["entry_order_id"] == "entry_order_x"
    assert call["exit_order_id"] == "exit_order_y"


@pytest.mark.asyncio
async def test_pagination_basic():
    """fetch_my_trades pagination — fills 영역 영역."""
    db_trade = {
        "id": 500,
        "timestamp": "2026-05-01T00:00:00+00:00",
        "closed_at": "2026-05-01T01:00:00+00:00",
        "side": "long",
        "size": 0.05,
        "entry_order_id": "x", "exit_order_id": "y",
    }
    db_ts_ms = _iso_to_ms(db_trade["timestamp"])
    # 100 fills + 50 fills (다음 page)
    page1 = [
        {"order": f"d_{i}", "id": f"p1_{i}",
         "timestamp": db_ts_ms + i * 1000,
         "price": 80000.0 + i, "amount": 1.0, "side": "buy",
         "info": {"fillPnl": "0"},
         "fee": {"cost": 0.1, "currency": "USDT"}}
        for i in range(100)
    ]
    page2 = [
        {"order": f"d_{i}", "id": f"p2_{i}",
         "timestamp": db_ts_ms + i * 1000,
         "price": 80000.0 + i, "amount": 1.0, "side": "buy",
         "info": {"fillPnl": "0"},
         "fee": {"cost": 0.1, "currency": "USDT"}}
        for i in range(100, 150)
    ]
    # positions-history 영역 — 매칭 영역 영역 (그래야 sync 완료)
    okx_position = _make_position(
        side="long", size_btc=0.05, open_price=80000, close_price=80100,
        fee=4.0, funding=0.0, realized_pnl=1.0,
        close_ts_ms=_iso_to_ms(db_trade["closed_at"]),
    )
    broker = _make_broker(
        is_live=True,
        fetch_my_trades_side_effect=[page1, page2, []],
        fetch_positions_history_return=[okx_position],
    )
    ds = _make_data_store([db_trade])
    await sync_all_unsynced(broker, ds, "BTC/USDT:USDT")
    assert broker.executor.exchange.fetch_my_trades.call_count == 2
