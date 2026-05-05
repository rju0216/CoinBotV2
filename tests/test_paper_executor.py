"""PaperExecutor 단위 테스트 — orderbook 통합 (BL-2-2 Step 2)."""

from __future__ import annotations

import pytest

from src.core.enums import OrderType, PositionSide
from src.execution.paper_executor import PaperExecutor


def _config(initial=10000.0):
    return {
        "exchange": {"symbol": "BTC/USDT:USDT"},
        "paper": {"initial_balance": initial},
        "accounting": {"taker_fee_pct": 0.0005, "slippage_pct": 0.0},
    }


class TestOpenPositionWithoutOrderbook:
    """기존 동작 (orderbook 미주입) — fill_price 그대로 사용."""

    @pytest.mark.asyncio
    async def test_uses_fill_price_directly(self):
        ex = PaperExecutor(_config())
        await ex.initialize()
        result = await ex.open_position(
            PositionSide.LONG, 0.1, fill_price=67000.0,
        )
        assert result["price"] == 67000.0
        pos = await ex.get_position()
        assert pos["entry_price"] == 67000.0


class TestOpenPositionWithOrderbook:
    """BL-2-2: orderbook 주입 시 VWAP 침투 가격 사용."""

    @pytest.mark.asyncio
    async def test_long_uses_ask_vwap(self):
        ex = PaperExecutor(_config())
        await ex.initialize()
        ob = {
            "bids": [[66999.0, 1.0]],
            "asks": [[67000.5, 0.5], [67001.0, 1.0]],  # 첫 ask 0.5 BTC
        }
        # size 0.3 → 첫 ask 안에 흡수 → 67000.5
        result = await ex.open_position(
            PositionSide.LONG, 0.3, fill_price=67000.0, orderbook=ob,
        )
        assert result["price"] == pytest.approx(67000.5)

    @pytest.mark.asyncio
    async def test_long_multi_level_vwap(self):
        ex = PaperExecutor(_config())
        await ex.initialize()
        ob = {
            "bids": [],
            "asks": [[67000.0, 0.5], [67001.0, 1.0]],
        }
        # size 1.0 → 0.5 @ 67000 + 0.5 @ 67001 = VWAP 67000.5
        result = await ex.open_position(
            PositionSide.LONG, 1.0, fill_price=67000.0, orderbook=ob,
        )
        expected = (0.5 * 67000.0 + 0.5 * 67001.0) / 1.0
        assert result["price"] == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_short_uses_bid_vwap(self):
        ex = PaperExecutor(_config())
        await ex.initialize()
        ob = {
            "bids": [[66999.5, 0.5], [66999.0, 1.0]],
            "asks": [[67000.5, 1.0]],
        }
        # SHORT 0.3 → 첫 bid 안에 흡수 → 66999.5
        result = await ex.open_position(
            PositionSide.SHORT, 0.3, fill_price=66999.0, orderbook=ob,
        )
        assert result["price"] == pytest.approx(66999.5)

    @pytest.mark.asyncio
    async def test_empty_orderbook_falls_back_to_fill_price(self):
        """사안 CC''' 가: orderbook depth 부족 (또는 빈 dict) → fill_price 그대로."""
        ex = PaperExecutor(_config())
        await ex.initialize()
        ob_empty = {"bids": [], "asks": []}
        result = await ex.open_position(
            PositionSide.LONG, 0.1, fill_price=67000.0, orderbook=ob_empty,
        )
        assert result["price"] == 67000.0  # fallback


class TestClosePositionWithOrderbook:
    @pytest.mark.asyncio
    async def test_close_long_uses_bid_vwap(self):
        """LONG 청산 = SHORT 방향 = bid 침투."""
        ex = PaperExecutor(_config())
        await ex.initialize()
        # LONG 진입
        await ex.open_position(PositionSide.LONG, 0.3, fill_price=67000.0)
        # 청산 시 호가창 — bids 가격으로 청산
        ob = {
            "bids": [[66999.5, 0.5]],
            "asks": [[67005.0, 0.5]],
        }
        result = await ex.close_position(
            PositionSide.LONG, 0.3, fill_price=67005.0, orderbook=ob,
        )
        assert result["price"] == pytest.approx(66999.5)  # bid VWAP

    @pytest.mark.asyncio
    async def test_close_short_uses_ask_vwap(self):
        """SHORT 청산 = LONG 방향 = ask 침투."""
        ex = PaperExecutor(_config())
        await ex.initialize()
        await ex.open_position(PositionSide.SHORT, 0.3, fill_price=67000.0)
        ob = {
            "bids": [[66995.0, 0.5]],
            "asks": [[67000.5, 0.5]],
        }
        result = await ex.close_position(
            PositionSide.SHORT, 0.3, fill_price=66995.0, orderbook=ob,
        )
        assert result["price"] == pytest.approx(67000.5)  # ask VWAP


class TestBackwardCompat:
    @pytest.mark.asyncio
    async def test_existing_call_signature_works(self):
        """기존 호출 (orderbook 인자 없음) 그대로 동작."""
        ex = PaperExecutor(_config())
        await ex.initialize()
        result = await ex.open_position(
            PositionSide.LONG, 0.1, fill_price=67000.0, order_type=OrderType.MARKET,
        )
        assert result["price"] == 67000.0
