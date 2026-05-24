"""TradeSyncer 단위 테스트 (BLE-6-1 + I-BLE002).

핵심 검증:
- paper 가드
- id 기반 매칭 성공
- 시간 매칭 fallback 성공 (기존 id 없는 trade)
- 매칭 실패 시 WARNING + synced_at NULL 유지
- 다중 fill aggregate 정확성 (contracts × contract_size = BTC 변환)
- fee currency != USDT 처리
- pagination 정상 동작 (PAGE_LIMIT 차면 다음 page fetch)
- pagination id dedup (중복 fill 차단)

I-BLE002 fix 반영:
- okx_fills 의 `amount` 는 contracts 단위 (× contract_size = BTC)
- reduce_only 판별은 `info.fillPnl` 합 ≠ 0 (OKX 응답에 reduceOnly 필드 부재)
- broker.executor.contract_size 로 BTC 변환
"""

from __future__ import annotations

import logging
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.live.trade_sync import sync_all_unsynced

CONTRACT_SIZE = 0.01  # BTC/USDT:USDT OKX contract size


def _iso_to_ms(iso_str: str) -> int:
    """test helper — db_trade['timestamp'] 와 fill ts 일치 보장."""
    return int(datetime.fromisoformat(iso_str).timestamp() * 1000)


def _make_broker(
    is_live: bool,
    fetch_my_trades_return=None,
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
    """entry_order_id/exit_order_id 있는 trade → id 매칭 + DB UPDATE.

    I-BLE002: amount 는 contracts 단위 (8.97 contracts = 0.0897 BTC), reduce_only 는 fillPnl 기반.
    """
    db_trade = {
        "id": 100,
        "timestamp": "2026-05-23T05:43:11+00:00",
        "side": "short",
        "size": 0.0897,   # BTC
        "funding_fee": 0.0,
        "entry_order_id": "order_entry_xyz",
        "exit_order_id": "order_exit_abc",
    }
    okx_fills = [
        {
            "order": "order_entry_xyz",
            "id": "fill_1",
            "timestamp": 1748000591000,
            "price": 75458.90,
            "amount": 8.97,   # contracts (= 0.0897 BTC)
            "side": "sell",
            "info": {"fillPnl": "0"},        # entry → fillPnl=0
            "fee": {"cost": 3.38, "currency": "USDT"},
        },
        {
            "order": "order_exit_abc",
            "id": "fill_2",
            "timestamp": 1748001500000,
            "price": 74337.09,
            "amount": 8.97,   # contracts
            "side": "buy",
            "info": {"fillPnl": "100.61"},    # exit → fillPnl=실현PnL
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
    """기존 trade (id 없음) → 시간/방향/size/reduce_only 매칭 + OKX id 같이 저장.

    I-BLE002: contracts × contract_size 후 size tolerance 비교, fillPnl 기반 reduce_only.
    """
    db_trade = {
        "id": 1,
        "timestamp": "2026-05-06T12:15:01+00:00",
        "side": "long",
        "size": 0.0750,   # BTC (round 영향 ±0.005 tolerance 안)
        "funding_fee": 0.0,
        "entry_order_id": None,
        "exit_order_id": None,
    }
    db_ts_ms = _iso_to_ms(db_trade["timestamp"])
    okx_fills = [
        {
            "order": "okx_order_111",
            "id": "fill_e1",
            "timestamp": db_ts_ms + 500,  # 0.5초 후
            "price": 82159.50,
            "amount": 7.503,   # contracts (= 0.07503 BTC, round 영향)
            "side": "buy",
            "info": {"fillPnl": "0"},        # entry
            "fee": {"cost": 3.08, "currency": "USDT"},
        },
        {
            "order": "okx_order_222",
            "id": "fill_x1",
            "timestamp": db_ts_ms + 1800_000,  # 30분 후 (SL hit)
            "price": 81707.30,
            "amount": 7.503,
            "side": "sell",
            "info": {"fillPnl": "-32.46"},   # exit
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
    """size tolerance 초과 → 매칭 실패 → WARNING + synced_at NULL 유지 (update 호출 안 됨).

    I-BLE002: 10.0 contracts × 0.01 = 0.10 BTC vs db_size 0.0561 BTC → diff 0.044 > 0.005 tolerance.
    """
    db_trade = {
        "id": 5,
        "timestamp": "2026-05-11T01:45:02+00:00",
        "side": "long",
        "size": 0.0561,   # BTC
        "funding_fee": 0.0,
        "entry_order_id": None,
        "exit_order_id": None,
    }
    okx_fills = [
        {
            "order": "okx_x", "id": "fill_z",
            "timestamp": 1747187102000,
            "price": 81454.9, "amount": 10.0,   # contracts (= 0.10 BTC, tolerance 초과)
            "side": "buy",
            "info": {"fillPnl": "0"},
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
    """다중 fill (한 order 의 여러 fills) → aggregate 후 price avg / amount(BTC) / fee 합 정확.

    I-BLE002: 각 fill amount(contracts) 합 × contract_size = BTC. fillPnl 합으로 reduce_only.
    """
    db_trade = {
        "id": 200,
        "timestamp": "2026-05-23T05:43:11+00:00",
        "side": "short",
        "size": 0.0897,   # BTC
        "funding_fee": 0.0,
        "entry_order_id": None,
        "exit_order_id": None,
    }
    db_ts_ms = _iso_to_ms(db_trade["timestamp"])
    okx_fills = [
        # Entry: 한 order 가 2 fills 로 분할 (5.0 + 3.97 = 8.97 contracts = 0.0897 BTC)
        {"order": "order_E", "id": "fe1",
         "timestamp": db_ts_ms + 100,
         "price": 75000.0, "amount": 5.0,
         "side": "sell",
         "info": {"fillPnl": "0"},
         "fee": {"cost": 1.875, "currency": "USDT"}},
        {"order": "order_E", "id": "fe2",
         "timestamp": db_ts_ms + 200,
         "price": 75002.0, "amount": 3.97,
         "side": "sell",
         "info": {"fillPnl": "0"},
         "fee": {"cost": 1.488, "currency": "USDT"}},
        # Exit: 단일 fill (8.97 contracts = 0.0897 BTC)
        {"order": "order_X", "id": "fx1",
         "timestamp": db_ts_ms + 60_000,
         "price": 74000.0, "amount": 8.97,
         "side": "buy",
         "info": {"fillPnl": "89.7"},
         "fee": {"cost": 3.32, "currency": "USDT"}},
    ]
    broker = _make_broker(is_live=True, fetch_my_trades_return=okx_fills)
    ds = _make_data_store([db_trade])
    result = await sync_all_unsynced(broker, ds, "BTC/USDT:USDT")
    assert result["synced_count"] == 1
    call = ds.update_synced_trade.call_args.kwargs
    # Entry aggregate: avg_price = (75000×5.0 + 75002×3.97) / 8.97 ≈ 75000.885
    expected_avg = (75000.0 * 5.0 + 75002.0 * 3.97) / 8.97
    assert call["entry_price"] == pytest.approx(expected_avg, abs=0.01)
    # fee_usdt = 1.875 + 1.488 (entry) + 3.32 (exit) = 6.683
    assert call["trading_fee"] == pytest.approx(1.875 + 1.488 + 3.32, abs=0.01)


@pytest.mark.asyncio
async def test_fee_currency_not_usdt_handled(caplog):
    """M=가: fee currency != 'USDT' → WARNING + fee 0 처리 (해당 fill 만).

    I-BLE002: amount contracts 단위 유지, fillPnl 기반 reduce_only.
    """
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
        {"order": "order_E", "id": "fe_okb",
         "timestamp": db_ts_ms + 100,
         "price": 75458.9, "amount": 8.97,
         "side": "sell",
         "info": {"fillPnl": "0"},
         "fee": {"cost": 0.1, "currency": "OKB"}},  # OKB 할인 (M=가 처리)
        {"order": "order_X", "id": "fx_usdt",
         "timestamp": db_ts_ms + 60_000,
         "price": 74337.09, "amount": 8.97,
         "side": "buy",
         "info": {"fillPnl": "100.61"},
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


# ---- I-BLE002 신규 pagination 테스트 ----


@pytest.mark.asyncio
async def test_pagination_basic():
    """pagination 정상 — Page 1 (100 fills) → Page 2 (50 fills) → 종료.

    fetch_my_trades 가 PAGE_LIMIT=100 채우면 다음 page 호출, 100 미만이면 마지막 page.
    """
    db_trade = {
        "id": 500,
        "timestamp": "2026-05-01T00:00:00+00:00",
        "side": "long",
        "size": 0.05,
        "funding_fee": 0.0,
        "entry_order_id": None,
        "exit_order_id": None,
    }
    db_ts_ms = _iso_to_ms(db_trade["timestamp"])

    # Page 1: 100 fills (id 0~99), 모두 같은 dummy order (매칭 안 됨)
    page1 = [
        {
            "order": f"dummy_{i}", "id": f"p1_{i}",
            "timestamp": db_ts_ms + i * 1000,
            "price": 80000.0 + i, "amount": 1.0,
            "side": "buy",
            "info": {"fillPnl": "0"},
            "fee": {"cost": 0.1, "currency": "USDT"},
        }
        for i in range(100)
    ]
    # Page 2: 50 fills (id 100~149), entry/exit 후보 포함
    page2 = [
        {
            "order": f"dummy_{i}", "id": f"p2_{i}",
            "timestamp": db_ts_ms + i * 1000,
            "price": 80000.0 + i, "amount": 1.0,
            "side": "buy",
            "info": {"fillPnl": "0"},
            "fee": {"cost": 0.1, "currency": "USDT"},
        }
        for i in range(100, 150)
    ]
    # Page 1 마지막 ts > Page 2 첫 ts 가 아니라 정상 시계열 — last_ts 기반 since 진행 보장
    broker = _make_broker(
        is_live=True,
        fetch_my_trades_side_effect=[page1, page2, []],
    )
    ds = _make_data_store([db_trade])
    await sync_all_unsynced(broker, ds, "BTC/USDT:USDT")
    # Page 1 100 fills → 다음 page fetch. Page 2 50 fills (< PAGE_LIMIT) → break.
    # 총 fetch_my_trades 호출 2회 (Page 1, Page 2). Page 3 ([]) 미도달.
    assert broker.executor.exchange.fetch_my_trades.call_count == 2


@pytest.mark.asyncio
async def test_match_failure_diagnostic_log(caplog):
    """I-BLE004: 매칭 실패 시 WARNING 로그에 by_order 후보 진단 정보 포함되는지.

    Trade 1 case 재현 — entry size mismatch (사용자 외 거래 섞임) + exit size 정확 일치.
    사용자가 즉시 'entry 영역 못 찾음 + exit 단독 매칭 가능' 을 로그로 파악 가능.
    """
    db_trade = {
        "id": 7,
        "timestamp": "2026-05-06T12:15:01+00:00",
        "side": "long",
        "size": 0.075,   # BTC
        "funding_fee": 0.0,
        "entry_order_id": None,
        "exit_order_id": None,
    }
    db_ts_ms = _iso_to_ms(db_trade["timestamp"])
    okx_fills = [
        # Entry 후보 (ts ±60s 안, side=buy 일치, size 0.0235 BTC 로 mismatch)
        {"order": "diff_size_entry", "id": "f1",
         "timestamp": db_ts_ms + 500,
         "price": 82170.0, "amount": 2.35,   # contracts (= 0.0235 BTC)
         "side": "buy",
         "info": {"fillPnl": "0"},
         "fee": {"cost": 1.0, "currency": "USDT"}},
        # Exit 후보 (ts 후 + size 0.075 BTC 정확 일치 + reduce_only)
        {"order": "matching_exit", "id": "f2",
         "timestamp": db_ts_ms + 1000,
         "price": 81707.30, "amount": 7.5,    # contracts (= 0.075 BTC)
         "side": "sell",
         "info": {"fillPnl": "-34.71"},
         "fee": {"cost": 3.06, "currency": "USDT"}},
    ]
    broker = _make_broker(is_live=True, fetch_my_trades_return=okx_fills)
    ds = _make_data_store([db_trade])
    with caplog.at_level(logging.WARNING):
        result = await sync_all_unsynced(broker, ds, "BTC/USDT:USDT")
    assert result["synced_count"] == 0
    assert result["failed_count"] == 1
    # 진단 정보 검증
    log_text = "\n".join(r.message for r in caplog.records)
    assert "매칭 실패" in log_text
    assert "entry buy 후보 (ts ±60s, size 무관): 1건" in log_text
    assert "exit sell reduce_only 후보 (size ±0.005 BTC, ts > db_ts): 1건" in log_text
    # 후보 order id (마지막 12자) + size 노출 확인
    assert "diff_size_ent" in log_text or "size=0.0235" in log_text
    assert "matching_exit" in log_text or "size=0.0750" in log_text


@pytest.mark.asyncio
async def test_pagination_dedup():
    """pagination 중복 fill (같은 id 가 두 page 에 등장) → seen_ids dedup, 무한 루프 차단."""
    db_trade = {
        "id": 600,
        "timestamp": "2026-05-01T00:00:00+00:00",
        "side": "long",
        "size": 0.05,
        "funding_fee": 0.0,
        "entry_order_id": None,
        "exit_order_id": None,
    }
    db_ts_ms = _iso_to_ms(db_trade["timestamp"])

    # Page 1: 100 fills (id 0~99)
    page1 = [
        {
            "order": f"dup_{i}", "id": f"d_{i}",
            "timestamp": db_ts_ms + i * 1000,
            "price": 80000.0 + i, "amount": 1.0,
            "side": "buy",
            "info": {"fillPnl": "0"},
            "fee": {"cost": 0.1, "currency": "USDT"},
        }
        for i in range(100)
    ]
    # Page 2: Page 1 의 마지막 fill 들이 다시 등장 (중복) — 신규 fill 0건
    page2 = page1[-50:]  # 같은 id 50건
    broker = _make_broker(
        is_live=True,
        fetch_my_trades_side_effect=[page1, page2, []],
    )
    ds = _make_data_store([db_trade])
    await sync_all_unsynced(broker, ds, "BTC/USDT:USDT")
    # Page 1 (100 신규) → Page 2 fetch → 모두 중복 (new_fills 비어있음) → break.
    # 따라서 fetch_my_trades 호출 2회. 무한 루프 안 일어남.
    assert broker.executor.exchange.fetch_my_trades.call_count == 2
