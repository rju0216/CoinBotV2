"""TradeSyncer 단위 테스트 (BLE-6-1).

핵심 검증:
- paper 가드
- id 기반 매칭 성공
- 시간 매칭 fallback 성공 (기존 id 없는 trade)
- 매칭 실패 시 WARNING + synced_at NULL 유지
- 다중 fill aggregate 정확성
- fee currency != USDT 처리
"""

from __future__ import annotations

import logging
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.live.trade_sync import sync_all_unsynced


def _iso_to_ms(iso_str: str) -> int:
    """test helper — db_trade['timestamp'] 와 fill ts 일치 보장."""
    return int(datetime.fromisoformat(iso_str).timestamp() * 1000)


def _make_broker(is_live: bool, fetch_my_trades_return=None, fetch_my_trades_side_effect=None):
    broker = MagicMock()
    broker.is_live = is_live
    executor = MagicMock()
    executor.exchange = MagicMock()
    if fetch_my_trades_side_effect is not None:
        executor.exchange.fetch_my_trades = AsyncMock(side_effect=fetch_my_trades_side_effect)
    else:
        executor.exchange.fetch_my_trades = AsyncMock(return_value=fetch_my_trades_return or [])
    broker.executor = executor
    return broker


def _make_data_store(unsynced_trades):
    ds = MagicMock()
    ds.get_unsynced_trades = AsyncMock(return_value=unsynced_trades)
    ds.update_synced_trade = AsyncMock()
    return ds


@pytest.mark.asyncio
async def test_paper_mode_guard():
    """K=가: paper 모드 (broker.is_live=False) → fetch_my_trades 호출 안 됨, 빈 결과."""
    broker = _make_broker(is_live=False)
    ds = _make_data_store(unsynced_trades=[])
    result = await sync_all_unsynced(broker, ds, "BTC/USDT:USDT")
    assert result == {"synced_count": 0, "failed_count": 0, "errors": []}
    # paper 가드라 ccxt 호출 X
    broker.executor.exchange.fetch_my_trades.assert_not_called()


@pytest.mark.asyncio
async def test_id_match_success():
    """entry_order_id/exit_order_id 있는 trade → id 매칭 + DB UPDATE."""
    db_trade = {
        "id": 100,
        "timestamp": "2026-05-23T05:43:11+00:00",
        "side": "short",
        "size": 0.0897,
        "funding_fee": 0.0,
        "entry_order_id": "order_entry_xyz",
        "exit_order_id": "order_exit_abc",
    }
    okx_fills = [
        {
            "order": "order_entry_xyz",
            "timestamp": 1748000591000,
            "price": 75458.90,
            "amount": 0.0897,
            "side": "sell",
            "reduceOnly": False,
            "fee": {"cost": 3.38, "currency": "USDT"},
        },
        {
            "order": "order_exit_abc",
            "timestamp": 1748001500000,
            "price": 74337.09,
            "amount": 0.0897,
            "side": "buy",
            "reduceOnly": True,
            "fee": {"cost": 3.33, "currency": "USDT"},
        },
    ]
    broker = _make_broker(is_live=True, fetch_my_trades_return=okx_fills)
    ds = _make_data_store([db_trade])
    result = await sync_all_unsynced(broker, ds, "BTC/USDT:USDT")
    assert result["synced_count"] == 1
    assert result["failed_count"] == 0
    # update_synced_trade 인자 검증
    ds.update_synced_trade.assert_called_once()
    call = ds.update_synced_trade.call_args.kwargs
    assert call["trade_id"] == 100
    assert call["entry_price"] == pytest.approx(75458.90)
    assert call["exit_price"] == pytest.approx(74337.09)
    assert call["trading_fee"] == pytest.approx(3.38 + 3.33)
    # SHORT: gross = (74337.09 - 75458.90) × 0.0897 × -1 = 100.61
    # net = gross - entry_fee - exit_fee = 100.61 - 6.71 = 93.90
    assert call["pnl"] == pytest.approx(
        (74337.09 - 75458.90) * 0.0897 * -1 - 3.38 - 3.33, abs=0.01,
    )
    assert call["entry_order_id"] == "order_entry_xyz"
    assert call["exit_order_id"] == "order_exit_abc"


@pytest.mark.asyncio
async def test_time_match_fallback_success():
    """기존 trade (id 없음) → 시간/방향/size/reduceOnly 매칭 + OKX id 같이 저장."""
    db_trade = {
        "id": 1,
        "timestamp": "2026-05-06T12:15:01+00:00",
        "side": "long",
        "size": 0.0750,  # round 영향 ±0.005 tolerance 안
        "funding_fee": 0.0,
        "entry_order_id": None,
        "exit_order_id": None,
    }
    db_ts_ms = _iso_to_ms(db_trade["timestamp"])
    okx_fills = [
        {
            "order": "okx_order_111",
            "timestamp": db_ts_ms + 500,  # 0.5초 후
            "price": 82159.50,
            "amount": 0.07503,  # round 영향
            "side": "buy",
            "reduceOnly": False,
            "fee": {"cost": 3.08, "currency": "USDT"},
        },
        {
            "order": "okx_order_222",
            "timestamp": db_ts_ms + 1800_000,  # 30분 후 (SL hit)
            "price": 81707.30,
            "amount": 0.07503,
            "side": "sell",
            "reduceOnly": True,
            "fee": {"cost": 3.06, "currency": "USDT"},
        },
    ]
    broker = _make_broker(is_live=True, fetch_my_trades_return=okx_fills)
    ds = _make_data_store([db_trade])
    result = await sync_all_unsynced(broker, ds, "BTC/USDT:USDT")
    assert result["synced_count"] == 1
    call = ds.update_synced_trade.call_args.kwargs
    # OKX id 도 같이 저장 (시간 매칭 성공해도 id 보존)
    assert call["entry_order_id"] == "okx_order_111"
    assert call["exit_order_id"] == "okx_order_222"


@pytest.mark.asyncio
async def test_match_failure_size_mismatch(caplog):
    """size tolerance 초과 → 매칭 실패 → WARNING + synced_at NULL 유지 (update 호출 안 됨)."""
    db_trade = {
        "id": 5,
        "timestamp": "2026-05-11T01:45:02+00:00",
        "side": "long",
        "size": 0.0561,  # tolerance 안 들어맞는 size
        "funding_fee": 0.0,
        "entry_order_id": None,
        "exit_order_id": None,
    }
    okx_fills = [
        {
            "order": "okx_x", "timestamp": 1747187102000,
            "price": 81454.9, "amount": 0.10,  # 0.10 vs 0.0561 → 차이 0.044 > 0.005 tolerance
            "side": "buy", "reduceOnly": False,
            "fee": {"cost": 4.0, "currency": "USDT"},
        },
    ]
    broker = _make_broker(is_live=True, fetch_my_trades_return=okx_fills)
    ds = _make_data_store([db_trade])
    with caplog.at_level(logging.WARNING):
        result = await sync_all_unsynced(broker, ds, "BTC/USDT:USDT")
    assert result["synced_count"] == 0
    assert result["failed_count"] == 1
    assert any("매칭 실패" in r.message for r in caplog.records)
    ds.update_synced_trade.assert_not_called()


@pytest.mark.asyncio
async def test_multi_fill_aggregate():
    """다중 fill (한 order 의 여러 fills) → aggregate 후 price avg / amount 합 / fee 합 정확."""
    db_trade = {
        "id": 200,
        "timestamp": "2026-05-23T05:43:11+00:00",
        "side": "short",
        "size": 0.0897,
        "funding_fee": 0.0,
        "entry_order_id": None,
        "exit_order_id": None,
    }
    db_ts_ms = _iso_to_ms(db_trade["timestamp"])
    okx_fills = [
        # Entry: 한 order 가 2 fills 로 분할
        {"order": "order_E", "timestamp": db_ts_ms + 100,
         "price": 75000.0, "amount": 0.05,
         "side": "sell", "reduceOnly": False,
         "fee": {"cost": 1.875, "currency": "USDT"}},
        {"order": "order_E", "timestamp": db_ts_ms + 200,
         "price": 75002.0, "amount": 0.0397,
         "side": "sell", "reduceOnly": False,
         "fee": {"cost": 1.488, "currency": "USDT"}},
        # Exit: 단일 fill
        {"order": "order_X", "timestamp": db_ts_ms + 60_000,
         "price": 74000.0, "amount": 0.0897,
         "side": "buy", "reduceOnly": True,
         "fee": {"cost": 3.32, "currency": "USDT"}},
    ]
    broker = _make_broker(is_live=True, fetch_my_trades_return=okx_fills)
    ds = _make_data_store([db_trade])
    result = await sync_all_unsynced(broker, ds, "BTC/USDT:USDT")
    assert result["synced_count"] == 1
    call = ds.update_synced_trade.call_args.kwargs
    # Entry aggregate: avg_price = (75000×0.05 + 75002×0.0397) / 0.0897 ≈ 75000.885
    expected_avg = (75000.0 * 0.05 + 75002.0 * 0.0397) / 0.0897
    assert call["entry_price"] == pytest.approx(expected_avg, abs=0.01)
    # fee_usdt = 1.875 + 1.488 (entry) + 3.32 (exit) = 6.683
    assert call["trading_fee"] == pytest.approx(1.875 + 1.488 + 3.32, abs=0.01)


@pytest.mark.asyncio
async def test_fee_currency_not_usdt_handled(caplog):
    """M=가: fee currency != 'USDT' → WARNING + fee 0 처리 (해당 fill 만)."""
    db_trade = {
        "id": 300,
        "timestamp": "2026-05-23T05:43:11+00:00",
        "side": "short",
        "size": 0.0897,
        "funding_fee": 0.0,
        "entry_order_id": None,
        "exit_order_id": None,
    }
    db_ts_ms = _iso_to_ms(db_trade["timestamp"])
    okx_fills = [
        {"order": "order_E", "timestamp": db_ts_ms + 100,
         "price": 75458.9, "amount": 0.0897,
         "side": "sell", "reduceOnly": False,
         "fee": {"cost": 0.1, "currency": "OKB"}},  # OKB 할인 (M=가 처리)
        {"order": "order_X", "timestamp": db_ts_ms + 60_000,
         "price": 74337.09, "amount": 0.0897,
         "side": "buy", "reduceOnly": True,
         "fee": {"cost": 3.33, "currency": "USDT"}},
    ]
    broker = _make_broker(is_live=True, fetch_my_trades_return=okx_fills)
    ds = _make_data_store([db_trade])
    with caplog.at_level(logging.WARNING):
        result = await sync_all_unsynced(broker, ds, "BTC/USDT:USDT")
    assert result["synced_count"] == 1
    # OKB fee currency WARNING 발생
    assert any("fee currency=OKB" in r.message for r in caplog.records)
    # entry fee = 0 (OKB skip), exit fee = 3.33
    call = ds.update_synced_trade.call_args.kwargs
    assert call["trading_fee"] == pytest.approx(0.0 + 3.33, abs=0.001)
