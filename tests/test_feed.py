"""DataFeed WebSocket 동작 테스트 (I-BL006 watchdog timeout 포함)."""

from __future__ import annotations

import asyncio
import logging

import pytest

import src.data.feed as feed_module
from src.core.event_bus import EventBus
from src.data.feed import DataFeed


class _MockExchange:
    """ccxt.pro exchange mock — behavior에 따라 watch_ohlcv 동작 시뮬."""

    def __init__(self, behavior: str):
        self.behavior = behavior  # "hang" / "raise_then_ok" / "ok"
        self.call_count = 0
        self.closed = False

    async def watch_ohlcv(self, symbol: str, timeframe: str):
        self.call_count += 1
        if self.behavior == "hang":
            # 무한 대기 — wait_for timeout 발동 검증용
            await asyncio.sleep(3600)
        if self.behavior == "raise_then_ok":
            if self.call_count == 1:
                raise ConnectionError("simulated disconnect")
        # ok / 2번째 호출 — 정상 ohlcv 1봉 반환
        return [[1700000000000, 67000.0, 67100.0, 66900.0, 67050.0, 12.34]]

    async def close(self):
        self.closed = True


def _make_feed(exchange):
    config = {
        "exchange": {"symbol": "BTC/USDT:USDT"},
        "data": {"history_bars": 300, "candle_dir": "data/candles"},
    }
    bus = EventBus()
    feed = DataFeed(config, bus, timeframes=["15m"])
    feed.exchange = exchange  # ccxt 인스턴스 교체
    return feed, bus


class TestWatchTfWatchdog:
    """I-BL006: watch_ohlcv hang 시 watchdog timeout으로 재시도 루프 진입."""

    @pytest.mark.asyncio
    async def test_hang_triggers_timeout_and_retries(self, monkeypatch, caplog):
        """watch_ohlcv가 hang하면 wait_for timeout → WARNING 로그 + 재시도."""
        # 테스트 가속: timeout 0.1초로 단축
        monkeypatch.setattr(feed_module, "WEBSOCKET_WATCHDOG_TIMEOUT_SEC", 0.1)
        # 재시도 sleep도 단축 (5초 → 0.05초로 patch는 어려움 — asyncio.sleep 자체는
        # 변경 안 함. 대신 충분한 시간 후 stop 호출)

        exchange = _MockExchange(behavior="hang")
        feed, _ = _make_feed(exchange)

        async def stop_after_delay():
            # timeout 발동 + sleep(5) 진입 직후 stop
            await asyncio.sleep(0.3)
            feed._running = False

        with caplog.at_level(logging.WARNING):
            await asyncio.gather(
                feed.stream(),
                stop_after_delay(),
            )

        # watchdog timeout WARNING 메시지 출력 확인
        assert any(
            "watchdog timeout" in record.message for record in caplog.records
        ), f"watchdog timeout 로그 없음. 로그: {[r.message for r in caplog.records]}"
        # watch_ohlcv가 최소 1번 호출됐어야 함 (재시도 루프 진입)
        assert exchange.call_count >= 1

    @pytest.mark.asyncio
    async def test_normal_ohlcv_publishes_bar_closed(self, monkeypatch):
        """정상 ohlcv 수신 시 BAR_CLOSED 이벤트 publish."""
        monkeypatch.setattr(feed_module, "WEBSOCKET_WATCHDOG_TIMEOUT_SEC", 1.0)

        exchange = _MockExchange(behavior="ok")
        feed, bus = _make_feed(exchange)

        published_events: list = []

        async def collect(data):
            published_events.append(data)
            feed._running = False  # 1회 받고 즉시 종료

        bus.subscribe("bar_closed", collect)

        await feed.stream()

        assert len(published_events) >= 1
        evt = published_events[0]
        assert evt["timeframe"] == "15m"
        assert evt["candle"]["close"] == 67050.0

    @pytest.mark.asyncio
    async def test_exception_triggers_reconnect_loop(self, monkeypatch, caplog):
        """ConnectionError 등 일반 예외도 5초 sleep 후 재시도 (TimeoutError 별도 처리와 공존)."""
        monkeypatch.setattr(feed_module, "WEBSOCKET_WATCHDOG_TIMEOUT_SEC", 5.0)

        exchange = _MockExchange(behavior="raise_then_ok")
        feed, bus = _make_feed(exchange)

        published_events: list = []

        async def collect(data):
            published_events.append(data)
            feed._running = False

        bus.subscribe("bar_closed", collect)

        with caplog.at_level(logging.ERROR):
            # 첫 호출 raise → sleep 5 → 재호출 ok → publish → stop
            # 5초 sleep을 기다려야 함. 빠르게 검증하려면 sleep도 patch 가능하지만
            # 현재 테스트는 그냥 timeout 6초까지 기다림
            await asyncio.wait_for(feed.stream(), timeout=10.0)

        # ERROR 메시지 출력 + 재시도 후 정상 publish
        assert any(
            "WebSocket error" in record.message for record in caplog.records
        )
        assert exchange.call_count == 2  # 첫 raise + 두 번째 ok
        assert len(published_events) == 1
