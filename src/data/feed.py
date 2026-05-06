"""실시간 캔들 피드 — WebSocket 구독 + 캐시 백필.

엔진이 활성 전략들의 entry_timeframe + required_timeframes 합집합을
주입하면 DataFeed는 그 TF 리스트만 백필·구독한다. 특정 전략(매크로/스캘핑)
하드코딩 분기는 제거.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import ccxt.pro as ccxtpro
import pandas as pd

from src.core.enums import EventType
from src.core.event_bus import EventBus

logger = logging.getLogger(__name__)

# I-BL006 fix: ccxt.pro watch_ohlcv가 disconnect 후 reconnect는 성공하지만 일정 시간
# 후 새 봉 수신 stuck 발생. 봉 진행 중 close 변동마다 update가 들어오는 정상 시
# 봉 마감 간격(15m=900s)보다 훨씬 짧은 빈도. 120초 update 없으면 hang 판정 → cancel
# + 5초 sleep 후 watch_ohlcv 재호출 (ccxt 내부에서 새 connection 시도).
WEBSOCKET_WATCHDOG_TIMEOUT_SEC = 120.0

TF_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


class DataFeed:
    def __init__(
        self,
        config: dict[str, Any],
        event_bus: EventBus,
        timeframes: list[str],
    ) -> None:
        """
        Args:
            timeframes: 엔진이 결정한 활성 TF 리스트 (전략들의 union).
        """
        self.config = config
        self.event_bus = event_bus
        self.symbol = config["exchange"]["symbol"]
        self.timeframes = list(dict.fromkeys(timeframes))  # 순서 유지·중복 제거

        data_cfg = config.get("data", {}) or {}
        self.history_bars = int(data_cfg.get("history_bars", 300))
        self.candle_dir = data_cfg.get("candle_dir", "data/candles")

        exchange_cfg: dict[str, Any] = {"options": {"defaultType": "swap"}}
        ec = config["exchange"]
        if ec.get("api_key"):
            exchange_cfg["apiKey"] = ec["api_key"]
            exchange_cfg["secret"] = ec["secret"]
            exchange_cfg["password"] = ec["passphrase"]
        if ec.get("sandbox"):
            exchange_cfg["sandbox"] = True

        self.exchange = ccxtpro.okx(exchange_cfg)
        self._running = False

    def _csv_path(self, timeframe: str) -> str:
        filename = (
            f"{self.symbol.replace('/', '_').replace(':', '_')}_{timeframe}.csv"
        )
        return os.path.join(self.candle_dir, filename)

    async def backfill(self) -> dict[str, list]:
        """모든 활성 TF에 대해 CSV 캐시 + API 병합 백필."""
        result: dict[str, list] = {}
        for tf in self.timeframes:
            result[tf] = await self._backfill_tf(tf)
        return result

    async def _backfill_tf(self, tf: str) -> list:
        csv_path = self._csv_path(tf)
        cached: list[list] = []
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, index_col="timestamp", parse_dates=True)
                for ts, row in df.iterrows():
                    cached.append([
                        int(ts.timestamp() * 1000),
                        row["open"], row["high"], row["low"],
                        row["close"], row["volume"],
                    ])
                logger.info(
                    "Loaded %d cached candles for %s from %s",
                    len(cached), tf, csv_path,
                )
            except Exception as e:
                logger.warning("Failed to load CSV cache for %s: %s", tf, e)
                cached = []

        need = self.history_bars
        if cached:
            since = cached[-1][0] + 1
            api_candles = await self._fetch_from_api(tf, need, since=since)
            all_candles = cached + api_candles
        else:
            all_candles = await self._fetch_from_api(tf, need)
            if not all_candles:
                logger.error(
                    "No data for %s: cache missing AND API failed.", tf,
                )

        # dedup + sort
        seen: dict[int, list] = {}
        for c in all_candles:
            seen[c[0]] = c
        all_candles = sorted(seen.values(), key=lambda x: x[0])
        if all_candles:
            self._save_csv(tf, all_candles)
        if len(all_candles) > self.history_bars:
            all_candles = all_candles[-self.history_bars :]

        logger.info("Backfilled %d candles for %s", len(all_candles), tf)
        return all_candles

    async def _fetch_from_api(
        self, tf: str, limit: int, since: int | None = None
    ) -> list:
        all_candles: list[list] = []
        fetched = 0
        batch_size = 100

        if since is None:
            tf_ms = TF_MS.get(tf, 3_600_000)
            since = int(time.time() * 1000) - (limit * tf_ms) - tf_ms

        while fetched < limit:
            try:
                batch = await self.exchange.fetch_ohlcv(
                    self.symbol, tf, since=since,
                    limit=min(batch_size, limit - fetched),
                )
            except Exception as e:
                logger.warning("API fetch error for %s: %s", tf, e)
                break
            if not batch:
                break
            all_candles.extend(batch)
            fetched += len(batch)
            since = batch[-1][0] + 1
            if len(batch) < batch_size:
                break
            await asyncio.sleep(0.3)
        return all_candles

    def _save_csv(self, tf: str, candles: list) -> None:
        try:
            os.makedirs(self.candle_dir, exist_ok=True)
            path = self._csv_path(tf)
            df = pd.DataFrame(
                candles,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            if os.path.exists(path):
                try:
                    old = pd.read_csv(
                        path, index_col="timestamp", parse_dates=["timestamp"]
                    )
                    old.index = pd.to_datetime(old.index, utc=True)
                    df = pd.concat([old, df])
                except Exception as e:
                    logger.warning("Failed to merge existing CSV %s: %s", path, e)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df.to_csv(path)
        except Exception as e:
            logger.warning("Failed to save CSV for %s: %s", tf, e)

    async def stream(self) -> None:
        """모든 활성 TF에 대해 WebSocket 캔들 구독."""
        self._running = True
        logger.info(
            "Starting WebSocket data feed for %s (timeframes=%s)",
            self.symbol, self.timeframes,
        )

        async def watch_tf(timeframe: str) -> None:
            while self._running:
                try:
                    # I-BL006 fix: watchdog timeout — watch_ohlcv가 hang하면 ccxt
                    # 내부에서 except 진입 못 함 → 봉 마감 미수신 stuck. wait_for로
                    # 강제 cancel 후 재시도 루프 진입.
                    ohlcv = await asyncio.wait_for(
                        self.exchange.watch_ohlcv(self.symbol, timeframe),
                        timeout=WEBSOCKET_WATCHDOG_TIMEOUT_SEC,
                    )
                    if ohlcv:
                        latest = ohlcv[-1]
                        candle = {
                            "timestamp": latest[0],
                            "open": latest[1],
                            "high": latest[2],
                            "low": latest[3],
                            "close": latest[4],
                            "volume": latest[5],
                        }
                        await self.event_bus.publish(
                            EventType.BAR_CLOSED.value,
                            {"timeframe": timeframe, "candle": candle},
                        )
                except asyncio.TimeoutError:
                    if not self._running:
                        break
                    logger.warning(
                        "WebSocket %s watchdog timeout (no update in %.0fs), reconnecting...",
                        timeframe, WEBSOCKET_WATCHDOG_TIMEOUT_SEC,
                    )
                    await asyncio.sleep(5)
                except Exception as e:
                    if not self._running:
                        break
                    logger.error(
                        "WebSocket error for %s: %s, reconnecting...", timeframe, e
                    )
                    await asyncio.sleep(5)

        await asyncio.gather(*(watch_tf(tf) for tf in self.timeframes))

    def save_all_csv(self, data_store) -> None:
        """현재 in-memory DataFrame들을 CSV에 머지·저장."""
        for tf in self.timeframes:
            df = data_store.get_df(tf)
            if df is None or df.empty:
                continue
            candles = []
            for ts, row in df.iterrows():
                candles.append([
                    int(ts.timestamp() * 1000),
                    row["open"], row["high"], row["low"],
                    row["close"], row["volume"],
                ])
            self._save_csv(tf, candles)
        logger.info("All candle CSVs saved")

    async def close(self) -> None:
        self._running = False
        await self.exchange.close()
        logger.info("DataFeed closed")
