"""OKX 실거래 executor.

partial TP, exit_plan, owner 분기 제거. 거래소 SL/TP pending 주문(엔진 정책 (a))
지원은 그대로 유지하여 라이브 안전장치 제공.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

import ccxt.async_support as ccxt

from src.core.enums import OrderType, PositionSide

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 1.0


async def _retry_api(
    func, *args, retries: int = MAX_RETRIES, delay: float = RETRY_DELAY, **kwargs,
):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return await func(*args, **kwargs)
        except (
            ccxt.NetworkError,
            ccxt.RequestTimeout,
            ccxt.ExchangeNotAvailable,
        ) as e:
            last_err = e
            if attempt < retries:
                wait = delay * attempt
                logger.warning(
                    "API call %s failed (attempt %d/%d): %s — retry in %.1fs",
                    getattr(func, "__name__", str(func)),
                    attempt, retries, e, wait,
                )
                await asyncio.sleep(wait)
            else:
                logger.error(
                    "API call %s failed after %d attempts: %s",
                    getattr(func, "__name__", str(func)), retries, e,
                )
    raise last_err


def _open_order_side(position_side: PositionSide) -> str:
    return "buy" if position_side == PositionSide.LONG else "sell"


def _close_order_side(position_side: PositionSide) -> str:
    return "sell" if position_side == PositionSide.LONG else "buy"


class LiveExecutor:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.symbol = config["exchange"]["symbol"]
        # 거래소 단일 leverage 설정 — 전략별 max_leverage는 사이징 클램프 용도.
        self.leverage = int(config["exchange"].get("leverage", 5))
        self.exchange = ccxt.okx({
            "apiKey": config["exchange"]["api_key"],
            "secret": config["exchange"]["secret"],
            "password": config["exchange"]["passphrase"],
            "options": {"defaultType": "swap"},
        })
        self.balance = 0.0
        self._wallet_balance = 0.0
        self.contract_size = 1.0
        if config["exchange"].get("sandbox"):
            self.exchange.set_sandbox_mode(True)

    async def initialize(self) -> None:
        await self.exchange.load_markets()
        market = self.exchange.market(self.symbol)
        self.contract_size = float(market.get("contractSize", 1.0) or 1.0)
        await _retry_api(self.exchange.set_leverage, self.leverage, self.symbol)
        try:
            await _retry_api(self.exchange.set_margin_mode, "cross", self.symbol)
        except Exception as e:
            logger.warning("set_margin_mode: %s (may already be set)", e)
        logger.info(
            "LiveExecutor initialized: %s, leverage=%dx, contract_size=%s",
            self.symbol, self.leverage, self.contract_size,
        )

    def _btc_to_contracts(self, btc_amount: float) -> float:
        if self.contract_size <= 0:
            return btc_amount
        contracts = btc_amount / self.contract_size
        try:
            return float(self.exchange.amount_to_precision(self.symbol, contracts))
        except Exception:
            return contracts

    def _contracts_to_btc(self, contracts: float) -> float:
        return contracts * self.contract_size

    async def get_balance(self) -> float:
        raw = await _retry_api(self.exchange.fetch_balance)
        usdt = raw.get("USDT", {})
        self.balance = float(usdt.get("total", 0))
        self._wallet_balance = float(usdt.get("free", 0))
        return self.balance

    async def get_wallet_balance(self) -> float:
        raw = await _retry_api(self.exchange.fetch_balance)
        usdt = raw.get("USDT", {})
        self._wallet_balance = float(usdt.get("free", 0))
        self.balance = float(usdt.get("total", 0))
        return self._wallet_balance

    async def get_position(self) -> dict | None:
        positions = await _retry_api(
            self.exchange.fetch_positions, [self.symbol]
        )
        for pos in positions:
            contracts = float(pos.get("contracts", 0) or 0)
            if contracts > 0:
                side_str = pos.get("side", "")
                pos_side = (
                    PositionSide.LONG if side_str == "long" else PositionSide.SHORT
                )
                entry = float(pos.get("entryPrice", 0) or 0)
                btc_size = self._contracts_to_btc(contracts)
                return {
                    "side": pos_side,
                    "size": btc_size,
                    "entry_price": entry,
                }
        return None

    async def open_position(
        self,
        side: PositionSide,
        size: float,
        fill_price: float | None = None,
        order_type: OrderType = OrderType.MARKET,
    ) -> dict:
        order_side = _open_order_side(side)
        contracts = self._btc_to_contracts(size)
        params = {"tdMode": "cross"}
        if order_type == OrderType.MARKET:
            order = await _retry_api(
                self.exchange.create_order,
                self.symbol, "market", order_side, contracts, params=params,
            )
        else:
            if fill_price is None:
                raise ValueError("fill_price required for limit order")
            order = await _retry_api(
                self.exchange.create_order,
                self.symbol, "limit", order_side, contracts, fill_price,
                params=params,
            )
        logger.info(
            "Opened %s %.4f BTC (%.4f contracts) %s",
            side.value, size, contracts, order_type.value,
        )
        return order

    async def close_position(
        self,
        side: PositionSide,
        size: float,
        fill_price: float | None = None,
        order_type: OrderType = OrderType.MARKET,
    ) -> dict:
        close_side = _close_order_side(side)
        contracts = self._btc_to_contracts(size)
        params = {"tdMode": "cross", "reduceOnly": True}
        order = await _retry_api(
            self.exchange.create_order,
            self.symbol, order_type.value, close_side, contracts, params=params,
        )
        logger.info(
            "Closed: %s %.4f BTC (%.4f contracts)",
            close_side, size, contracts,
        )
        return order

    async def place_stop_loss(
        self, side: PositionSide, trigger_price: float, size: float
    ) -> dict:
        close_side = _close_order_side(side)
        contracts = self._btc_to_contracts(size)
        order = await _retry_api(
            self.exchange.create_order,
            self.symbol, "market", close_side, contracts,
            params={
                "tdMode": "cross",
                "reduceOnly": True,
                "stopLossPrice": trigger_price,
            },
        )
        logger.info(
            "SL set: %s %.4f contracts @ %.2f",
            close_side, contracts, trigger_price,
        )
        return order

    async def place_take_profit(
        self, side: PositionSide, trigger_price: float, size: float
    ) -> dict:
        close_side = _close_order_side(side)
        contracts = self._btc_to_contracts(size)
        order = await _retry_api(
            self.exchange.create_order,
            self.symbol, "market", close_side, contracts,
            params={
                "tdMode": "cross",
                "reduceOnly": True,
                "takeProfitPrice": trigger_price,
            },
        )
        logger.info(
            "TP set: %s %.4f contracts @ %.2f",
            close_side, contracts, trigger_price,
        )
        return order

    async def cancel_all_orders(self) -> None:
        orders = await _retry_api(self.exchange.fetch_open_orders, self.symbol)
        cancelled = 0
        for order in orders:
            try:
                await _retry_api(
                    self.exchange.cancel_order, order["id"], self.symbol
                )
                cancelled += 1
            except Exception as e:
                logger.error(
                    "Failed to cancel order %s: %s — continuing",
                    order["id"], e,
                )
        logger.info("Cancelled %d/%d open orders", cancelled, len(orders))

    async def fetch_funding_history(
        self, since: str | None = None, until: str | None = None
    ) -> list[dict]:
        try:
            since_ms = None
            if since:
                dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
                since_ms = int(dt.timestamp() * 1000)
            raw = await _retry_api(
                self.exchange.fetch_funding_history,
                self.symbol, since=since_ms, limit=100,
            )
            results = []
            for entry in raw:
                ts = entry.get("datetime") or entry.get("timestamp")
                amount = float(entry.get("amount", 0))
                if until and ts and ts > until:
                    continue
                results.append({
                    "timestamp": ts,
                    "amount": amount,
                    "symbol": entry.get("symbol", self.symbol),
                })
            return results
        except Exception as e:
            logger.error("fetch_funding_history failed: %s", e)
            return []

    async def close(self) -> None:
        await self.exchange.close()
