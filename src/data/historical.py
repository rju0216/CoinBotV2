"""과거 캔들 백필 — 백테스트 / 라이브 시작 시 캐시·API 병합."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import ccxt.async_support as ccxt
import pandas as pd

logger = logging.getLogger(__name__)

TF_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


class HistoricalDataLoader:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.symbol = config["exchange"]["symbol"]
        self.candle_dir = config.get("data", {}).get("candle_dir", "data/candles")
        self.exchange: ccxt.okx | None = None

    async def _init_exchange(self) -> None:
        if self.exchange is None:
            self.exchange = ccxt.okx({"options": {"defaultType": "swap"}})
            await self.exchange.load_markets()

    async def close(self) -> None:
        if self.exchange:
            await self.exchange.close()
            self.exchange = None

    async def download(
        self, timeframe: str, limit: int = 300, since: int | None = None
    ) -> pd.DataFrame:
        await self._init_exchange()

        all_candles: list[list] = []
        fetched = 0
        batch_size = 100

        if since is None:
            tf_ms = TF_MS.get(timeframe, 3_600_000)
            since = int(time.time() * 1000) - (limit * tf_ms) - tf_ms

        current_since = since
        while fetched < limit:
            fetch_limit = min(batch_size, limit - fetched)
            extra = {"bar": "1Dutc"} if timeframe == "1d" else {}
            candles = await self.exchange.fetch_ohlcv(
                self.symbol, timeframe, since=current_since,
                limit=fetch_limit, params=extra,
            )
            if not candles:
                break
            all_candles.extend(candles)
            fetched += len(candles)
            current_since = candles[-1][0] + 1
            if len(candles) < fetch_limit:
                break
            await asyncio.sleep(0.3)

        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)
        logger.info("Downloaded %d candles for %s %s", len(df), self.symbol, timeframe)
        return df

    def _csv_path(self, timeframe: str) -> str:
        filename = (
            f"{self.symbol.replace('/', '_').replace(':', '_')}_{timeframe}.csv"
        )
        return os.path.join(self.candle_dir, filename)

    async def download_to_csv(self, timeframe: str, limit: int = 300) -> str:
        df = await self.download(timeframe, limit)
        os.makedirs(self.candle_dir, exist_ok=True)
        path = self._csv_path(timeframe)
        df.to_csv(path)
        logger.info("Saved %d candles to %s", len(df), path)
        return path

    async def download_range_merged(
        self, timeframe: str, start_ms: int, end_ms: int
    ) -> pd.DataFrame:
        """[start_ms, end_ms] 범위를 캐시·API 병합으로 보장.

        - 기존 CSV 로드 → 부족한 앞/뒤 구간만 API에서 추가
        - 합집합을 dedupe·정렬·저장
        """
        os.makedirs(self.candle_dir, exist_ok=True)
        path = self._csv_path(timeframe)
        tf_ms = TF_MS.get(timeframe, 60_000)

        existing: pd.DataFrame | None = None
        if os.path.exists(path):
            try:
                existing = self.load_from_csv(path)
                existing.index = pd.to_datetime(existing.index, utc=True)
                existing = existing[~existing.index.duplicated(keep="last")].sort_index()
            except Exception as e:
                logger.warning("Failed to read existing CSV %s: %s", path, e)
                existing = None

        new_chunks: list[pd.DataFrame] = []
        if existing is None or existing.empty:
            ranges_to_fetch = [(start_ms, end_ms)]
        else:
            existing_first_ms = int(existing.index.min().timestamp() * 1000)
            existing_last_ms = int(existing.index.max().timestamp() * 1000)
            ranges_to_fetch = []
            if start_ms < existing_first_ms:
                ranges_to_fetch.append((start_ms, existing_first_ms - tf_ms))
            if end_ms > existing_last_ms:
                ranges_to_fetch.append((existing_last_ms + tf_ms, end_ms))

        for r_start, r_end in ranges_to_fetch:
            if r_end < r_start:
                continue
            n_bars = max(1, int((r_end - r_start) / tf_ms) + 1)
            chunk = await self.download(timeframe, limit=n_bars, since=r_start)
            if not chunk.empty:
                new_chunks.append(chunk)

        frames: list[pd.DataFrame] = []
        if existing is not None and not existing.empty:
            frames.append(existing)
        frames.extend(new_chunks)
        if not frames:
            return pd.DataFrame()

        merged = pd.concat(frames)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()

        if new_chunks or existing is None:
            merged.to_csv(path)
            logger.info(
                "Saved merged %s CSV: %d rows (%s ~ %s)",
                timeframe, len(merged), merged.index.min(), merged.index.max(),
            )

        # 캐시 CSV는 전체 보존하되, 반환값은 호출자가 요청한 [start_ms, end_ms]로 슬라이스 (I-B006)
        start_ts = pd.Timestamp(start_ms, unit="ms", tz="UTC")
        end_ts = pd.Timestamp(end_ms, unit="ms", tz="UTC")
        merged = merged[(merged.index >= start_ts) & (merged.index <= end_ts)]
        return merged

    @staticmethod
    def load_from_csv(filepath: str) -> pd.DataFrame:
        return pd.read_csv(filepath, index_col="timestamp", parse_dates=True)
