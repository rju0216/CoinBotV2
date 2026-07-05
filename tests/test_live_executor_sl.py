"""I-PE009: LiveExecutor.update_stop_loss — 기존 SL algo cancel + 재등록 흐름 단위 테스트.

⚠️ OKX algo cancel 실동작·중복 동작은 A5 라이브 실검증. 여기선 호출 흐름만 mock 검증.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.enums import PositionSide
from src.execution.live_executor import LiveExecutor


def _executor() -> LiveExecutor:
    ex = LiveExecutor.__new__(LiveExecutor)  # __init__(ccxt 초기화) 우회
    ex.symbol = "BTC/USDT:USDT"
    ex.exchange = MagicMock()
    ex.exchange.cancel_order = AsyncMock()
    ex._call = AsyncMock()  # cancel 호출 캡처
    ex.place_stop_loss = AsyncMock(return_value={"id": "new_sl"})
    return ex


@pytest.mark.asyncio
async def test_update_stop_loss_cancels_existing_sl_then_replaces():
    ex = _executor()
    ex.fetch_open_algo_orders = AsyncMock(return_value=[
        {"id": "algo_sl", "info": {"slTriggerPx": "64000"}},
    ])
    await ex.update_stop_loss(PositionSide.LONG, 61000.0, 0.05)
    # 기존 SL algo cancel (via _call(exchange.cancel_order, ...))
    ex._call.assert_awaited_once()
    assert ex._call.call_args.args[0] is ex.exchange.cancel_order
    assert ex._call.call_args.args[1] == "algo_sl"
    # 신규 SL 재등록
    ex.place_stop_loss.assert_awaited_once_with(PositionSide.LONG, 61000.0, 0.05)


@pytest.mark.asyncio
async def test_update_stop_loss_no_existing_sl_just_places():
    ex = _executor()
    ex.fetch_open_algo_orders = AsyncMock(return_value=[])  # 기존 SL 없음
    await ex.update_stop_loss(PositionSide.SHORT, 63000.0, 0.05)
    ex._call.assert_not_awaited()  # cancel 없음
    ex.place_stop_loss.assert_awaited_once_with(PositionSide.SHORT, 63000.0, 0.05)


@pytest.mark.asyncio
async def test_update_stop_loss_fetch_fail_still_replaces():
    ex = _executor()
    ex.fetch_open_algo_orders = AsyncMock(side_effect=RuntimeError("net"))
    await ex.update_stop_loss(PositionSide.LONG, 61000.0, 0.05)
    # 조회 실패해도 재등록은 진행 (안전망 최소 확보)
    ex.place_stop_loss.assert_awaited_once()
