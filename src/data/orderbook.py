"""호가창 인프라 (BL-2-2).

OKX 호가창 snapshot 수집 (BAR_CLOSED 시점, depth=20) + parquet 저장.
PaperExecutor가 사이즈 침투 VWAP 계산에 사용.

설계 원칙:
- ccxt fetch_order_book 패턴 그대로 (bids/asks list of [price, amount])
- snapshot 1개 = parquet 1 row (wide format: bid_price_0..19, bid_amount_0..19, ask_*)
- 일별 파일 분리 (`data/orderbook/<symbol>_<YYYY-MM-DD>.parquet`)
- 매 fetch 시 즉시 read+concat+write (15m 주기 + ~100 row/일이라 부담 작음)
- 라이브 거래소(OKX)는 자동 호가창 침투 → LiveExecutor는 호가창 사용 안 함
- Fallback (사안 CC''' 가): orderbook None / depth 부족 → fill_price 그대로 (silent)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.enums import PositionSide

logger = logging.getLogger(__name__)


def compute_market_impact(
    orderbook: dict[str, Any],
    side: PositionSide,
    size: float,
) -> float | None:
    """호가창 침투 VWAP 계산.

    LONG: ask side (낮은 가격부터) 침투
    SHORT: bid side (높은 가격부터) 침투

    Args:
        orderbook: ccxt fetch_order_book 형식 {"bids": [[p, a], ...], "asks": [[p, a], ...]}
        side: PositionSide.LONG / SHORT
        size: 진입 size (BTC 단위)

    Returns:
        VWAP 평균 체결가. depth 부족 또는 잘못된 입력 시 None (호출자가 fallback).
    """
    if size <= 0:
        return None
    if side == PositionSide.LONG:
        levels = orderbook.get("asks", [])
    elif side == PositionSide.SHORT:
        levels = orderbook.get("bids", [])
    else:
        return None
    if not levels:
        return None

    remaining = float(size)
    cost = 0.0
    filled = 0.0
    for level in levels:
        if len(level) < 2:
            continue
        price = float(level[0])
        amount = float(level[1])
        if price <= 0 or amount <= 0:
            continue
        take = min(remaining, amount)
        cost += take * price
        filled += take
        remaining -= take
        if remaining <= 1e-12:
            break

    if filled <= 0:
        return None
    if remaining > 1e-9:
        # depth 부족 — 사용자에게 알림 (warning) 후 부분 VWAP만 반환
        logger.warning(
            "compute_market_impact: depth 부족. requested=%.6f filled=%.6f remaining=%.6f. "
            "VWAP은 침투 가능한 부분만 반영 — 잔여는 마지막 호가 가격으로 가정",
            size, filled, remaining,
        )
        # 잔여를 마지막 호가 가격으로 채워 VWAP 산정 (실 거래에서 호가창 끝 도달 시 마켓 메이커 추가 가격으로 체결될 수 있음을 약식 모델링)
        last_price = float(levels[-1][0])
        cost += remaining * last_price
        filled = float(size)

    return cost / filled


class OrderBookCollector:
    """OKX 호가창 snapshot 수집 + parquet 저장 (BAR_CLOSED 시점).

    사용:
        collector = OrderBookCollector(config, exchange_client)
        snapshot = await collector.fetch_and_save()  # 매 BAR_CLOSED 호출
        # snapshot은 ccxt 형식 dict (PaperExecutor에 전달 가능)
    """

    def __init__(
        self,
        config: dict[str, Any],
        exchange_client: Any,
    ) -> None:
        self.config = config
        self.symbol = config["exchange"]["symbol"]
        ob_cfg = (config.get("live", {}) or {}).get("orderbook", {}) or {}
        self.enabled = bool(ob_cfg.get("enabled", False))
        self.depth = int(ob_cfg.get("depth", 20))
        save_dir = ob_cfg.get("save_dir", "data/orderbook")
        self.save_dir = Path(save_dir)
        if self.enabled:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.exchange = exchange_client

    async def fetch_and_save(self) -> dict[str, Any] | None:
        """호가창 snapshot fetch + parquet 저장. 반환: ccxt 형식 dict 또는 None."""
        if not self.enabled:
            return None
        try:
            ob = await self.exchange.fetch_order_book(self.symbol, limit=self.depth)
        except Exception as e:
            logger.warning("OrderBook fetch failed: %s", e)
            return None
        try:
            self._append_to_parquet(ob)
        except Exception as e:
            logger.warning("OrderBook parquet save failed: %s", e)
        return ob

    def _append_to_parquet(self, ob: dict[str, Any]) -> None:
        ts_ms = ob.get("timestamp")
        if ts_ms is None:
            ts = datetime.now(timezone.utc)
        else:
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

        row = self._snapshot_to_row(ts, ob)
        new_df = pd.DataFrame([row])
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], utc=True)

        path = self._daily_path(ts)
        if path.exists():
            try:
                existing = pd.read_parquet(path)
                combined = pd.concat([existing, new_df], ignore_index=True)
            except Exception as e:
                logger.warning("기존 parquet read 실패, overwrite: %s", e)
                combined = new_df
        else:
            combined = new_df
        combined.to_parquet(path, index=False)

    def _daily_path(self, ts: datetime) -> Path:
        date_str = ts.strftime("%Y-%m-%d")
        symbol_safe = self.symbol.replace("/", "_").replace(":", "_")
        return self.save_dir / f"{symbol_safe}_{date_str}.parquet"

    def _snapshot_to_row(
        self, ts: datetime, ob: dict[str, Any]
    ) -> dict[str, Any]:
        """ccxt orderbook → wide-format dict (bid_price_0..N, bid_amount_0..N, ask_*)."""
        row: dict[str, Any] = {"timestamp": ts}
        for side_key in ("bids", "asks"):
            levels = ob.get(side_key, []) or []
            for i in range(self.depth):
                if i < len(levels) and len(levels[i]) >= 2:
                    row[f"{side_key[:-1]}_price_{i}"] = float(levels[i][0])
                    row[f"{side_key[:-1]}_amount_{i}"] = float(levels[i][1])
                else:
                    row[f"{side_key[:-1]}_price_{i}"] = float("nan")
                    row[f"{side_key[:-1]}_amount_{i}"] = float("nan")
        return row


def row_to_ccxt(row: dict[str, Any], depth: int = 20) -> dict[str, Any]:
    """parquet row (wide) → ccxt 형식 dict 역변환 (분석/백테용)."""
    bids = []
    asks = []
    for i in range(depth):
        bp = row.get(f"bid_price_{i}")
        ba = row.get(f"bid_amount_{i}")
        if bp is not None and not pd.isna(bp):
            bids.append([float(bp), float(ba)])
        ap = row.get(f"ask_price_{i}")
        aa = row.get(f"ask_amount_{i}")
        if ap is not None and not pd.isna(ap):
            asks.append([float(ap), float(aa)])
    return {"bids": bids, "asks": asks, "timestamp": row.get("timestamp")}
