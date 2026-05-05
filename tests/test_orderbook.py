"""src/data/orderbook.py 단위 테스트 (BL-2-2 Step 1)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pandas as pd
import pytest

from src.core.enums import PositionSide
from src.data.orderbook import (
    OrderBookCollector,
    compute_market_impact,
    row_to_ccxt,
)


def _book(bids=None, asks=None) -> dict:
    return {
        "bids": bids or [],
        "asks": asks or [],
        "timestamp": 1700000000000,
    }


# ─── compute_market_impact ───


class TestComputeMarketImpact:
    def test_long_first_level_only(self):
        """size 0.1 BTC, 첫 ask 0.5 BTC 안에 흡수 → 첫 호가 가격."""
        ob = _book(asks=[[67000.0, 0.5], [67001.0, 1.0]])
        price = compute_market_impact(ob, PositionSide.LONG, 0.1)
        assert price == pytest.approx(67000.0)

    def test_short_first_level_only(self):
        """SHORT 0.3 BTC, 첫 bid 0.5 BTC 안에 흡수 → 첫 bid 가격."""
        ob = _book(bids=[[66999.0, 0.5], [66998.0, 1.0]])
        price = compute_market_impact(ob, PositionSide.SHORT, 0.3)
        assert price == pytest.approx(66999.0)

    def test_long_multi_level_vwap(self):
        """size 1.5 BTC: 0.5 @ 67000 + 1.0 @ 67001 → VWAP."""
        ob = _book(asks=[[67000.0, 0.5], [67001.0, 1.0]])
        price = compute_market_impact(ob, PositionSide.LONG, 1.5)
        # VWAP = (0.5*67000 + 1.0*67001) / 1.5 = 66950.667 — wait, 0.5*67000=33500, 1.0*67001=67001, sum=100501, /1.5=67000.667
        expected = (0.5 * 67000.0 + 1.0 * 67001.0) / 1.5
        assert price == pytest.approx(expected)

    def test_short_multi_level_vwap(self):
        """SHORT 1.0 BTC: 0.4 @ 66999 + 0.6 @ 66998 → VWAP."""
        ob = _book(bids=[[66999.0, 0.4], [66998.0, 0.6]])
        price = compute_market_impact(ob, PositionSide.SHORT, 1.0)
        expected = (0.4 * 66999.0 + 0.6 * 66998.0) / 1.0
        assert price == pytest.approx(expected)

    def test_depth_insufficient_uses_last_price(self, caplog):
        """size > 호가창 총 깊이: 마지막 호가 가격으로 잔여 채움 + warning."""
        ob = _book(asks=[[67000.0, 0.1], [67001.0, 0.1]])  # 총 0.2 깊이
        with caplog.at_level("WARNING"):
            price = compute_market_impact(ob, PositionSide.LONG, 1.0)
        assert price is not None
        assert any("depth 부족" in r.message for r in caplog.records)
        # VWAP: 0.1*67000 + 0.1*67001 + 0.8*67001 = (6700 + 6700.1 + 53600.8) / 1.0
        expected = (0.1 * 67000.0 + 0.1 * 67001.0 + 0.8 * 67001.0) / 1.0
        assert price == pytest.approx(expected)

    def test_zero_size_returns_none(self):
        ob = _book(asks=[[67000.0, 0.5]])
        assert compute_market_impact(ob, PositionSide.LONG, 0.0) is None

    def test_empty_orderbook_returns_none(self):
        ob = _book()  # 빈
        assert compute_market_impact(ob, PositionSide.LONG, 0.1) is None

    def test_none_side_returns_none(self):
        ob = _book(asks=[[67000.0, 0.5]])
        assert compute_market_impact(ob, PositionSide.NONE, 0.1) is None

    def test_invalid_level_skipped(self):
        """price 0 또는 amount 0 level은 skip."""
        ob = _book(asks=[[0, 0.5], [67001.0, 1.0]])  # 첫 level 무효
        price = compute_market_impact(ob, PositionSide.LONG, 0.5)
        assert price == pytest.approx(67001.0)


# ─── OrderBookCollector ───


class TestOrderBookCollector:
    @pytest.mark.asyncio
    async def test_disabled_returns_none(self):
        cfg = {
            "exchange": {"symbol": "BTC/USDT:USDT"},
            "live": {"orderbook": {"enabled": False}},
        }
        mock = AsyncMock()
        c = OrderBookCollector(cfg, mock)
        result = await c.fetch_and_save()
        assert result is None
        mock.fetch_order_book.assert_not_called()

    @pytest.mark.asyncio
    async def test_enabled_fetches_and_saves(self, tmp_path):
        cfg = {
            "exchange": {"symbol": "BTC/USDT:USDT"},
            "live": {
                "orderbook": {
                    "enabled": True,
                    "depth": 5,
                    "save_dir": str(tmp_path / "orderbook"),
                }
            },
        }
        ob_payload = {
            "bids": [[66999.0, 0.5], [66998.0, 1.0]],
            "asks": [[67000.0, 0.5], [67001.0, 1.0]],
            "timestamp": 1700000000000,
        }
        mock = AsyncMock()
        mock.fetch_order_book = AsyncMock(return_value=ob_payload)

        c = OrderBookCollector(cfg, mock)
        result = await c.fetch_and_save()

        assert result == ob_payload
        mock.fetch_order_book.assert_called_once_with("BTC/USDT:USDT", limit=5)
        # parquet 생성 확인
        parquet_files = list((tmp_path / "orderbook").glob("*.parquet"))
        assert len(parquet_files) == 1
        df = pd.read_parquet(parquet_files[0])
        assert len(df) == 1
        assert df["bid_price_0"].iloc[0] == 66999.0
        assert df["ask_price_0"].iloc[0] == 67000.0

    @pytest.mark.asyncio
    async def test_multiple_fetches_append_to_same_daily_file(self, tmp_path):
        cfg = {
            "exchange": {"symbol": "BTC/USDT:USDT"},
            "live": {
                "orderbook": {
                    "enabled": True,
                    "depth": 5,
                    "save_dir": str(tmp_path / "orderbook"),
                }
            },
        }
        ts_base = 1700000000000  # 같은 날짜
        ob1 = {"bids": [[66999.0, 0.5]], "asks": [[67000.0, 0.5]], "timestamp": ts_base}
        ob2 = {"bids": [[66998.0, 0.6]], "asks": [[67001.0, 0.6]], "timestamp": ts_base + 60_000}

        mock = AsyncMock()
        c = OrderBookCollector(cfg, mock)

        mock.fetch_order_book = AsyncMock(return_value=ob1)
        await c.fetch_and_save()
        mock.fetch_order_book = AsyncMock(return_value=ob2)
        await c.fetch_and_save()

        files = list((tmp_path / "orderbook").glob("*.parquet"))
        assert len(files) == 1  # 같은 날짜 파일에 append
        df = pd.read_parquet(files[0])
        assert len(df) == 2

    @pytest.mark.asyncio
    async def test_fetch_failure_returns_none_no_crash(self):
        cfg = {
            "exchange": {"symbol": "BTC/USDT:USDT"},
            "live": {
                "orderbook": {"enabled": True, "depth": 5, "save_dir": "/tmp/nonexistent_safe"},
            },
        }
        mock = AsyncMock()
        mock.fetch_order_book = AsyncMock(side_effect=RuntimeError("boom"))
        c = OrderBookCollector(cfg, mock)
        result = await c.fetch_and_save()
        assert result is None  # 거래 흐름 차단 X


# ─── row_to_ccxt ───


class TestRowToCcxt:
    def test_round_trip(self, tmp_path):
        cfg = {
            "exchange": {"symbol": "BTC/USDT:USDT"},
            "live": {
                "orderbook": {
                    "enabled": True,
                    "depth": 5,
                    "save_dir": str(tmp_path),
                }
            },
        }
        c = OrderBookCollector(cfg, AsyncMock())
        ob = {
            "bids": [[66999.0, 0.5], [66998.0, 1.0]],
            "asks": [[67000.0, 0.5], [67001.0, 1.0]],
            "timestamp": 1700000000000,
        }
        from datetime import datetime, timezone
        ts = datetime.fromtimestamp(1700000000, tz=timezone.utc)
        row = c._snapshot_to_row(ts, ob)
        recovered = row_to_ccxt(row, depth=5)
        assert len(recovered["bids"]) == 2
        assert recovered["bids"][0] == [66999.0, 0.5]
        assert recovered["asks"][1] == [67001.0, 1.0]
