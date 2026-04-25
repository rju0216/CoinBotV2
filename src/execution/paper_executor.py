"""페이퍼 모드 시뮬레이션 executor.

SL/TP 체결 판정은 엔진이 담당하므로 PaperExecutor는 주문 수령·포지션 보유·
잔액 정산만 한다. partial close, exit_plan, owner 분기는 모두 제거.
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.enums import OrderType, PositionSide

logger = logging.getLogger(__name__)


class PaperExecutor:
    def __init__(self, config: dict[str, Any]) -> None:
        self.symbol = config["exchange"]["symbol"]
        paper_cfg = config.get("paper", {}) or {}
        self.initial_balance = float(paper_cfg.get("initial_balance", 10000.0))
        self.balance = self.initial_balance
        # 보유 포지션 정보 (None = 슬롯 비어있음)
        self._position: dict | None = None
        self._order_id = 0

    async def initialize(self) -> None:
        logger.info("PaperExecutor initialized: balance=$%.2f", self.balance)

    async def restore_state(
        self, balance: float, open_trade: dict | None
    ) -> None:
        """DB에서 잔액·오픈 포지션 복원."""
        self.balance = balance
        if open_trade:
            self._position = {
                "side": PositionSide(open_trade["side"]),
                "size": float(open_trade["size"]),
                "entry_price": float(open_trade["entry_price"]),
                "trade_id": open_trade["id"],
            }
            logger.info(
                "Restored paper position: %s %.4f @ $%.2f",
                self._position["side"].value,
                self._position["size"],
                self._position["entry_price"],
            )

    async def get_balance(self) -> float:
        return self.balance

    async def get_wallet_balance(self) -> float:
        return self.balance

    async def get_position(self) -> dict | None:
        return dict(self._position) if self._position else None

    async def open_position(
        self,
        side: PositionSide,
        size: float,
        fill_price: float | None = None,
        order_type: OrderType = OrderType.MARKET,
    ) -> dict:
        if fill_price is None or fill_price <= 0:
            logger.error("Cannot open paper position without fill_price")
            return {}
        if self._position is not None:
            logger.warning(
                "Open while position exists; closing existing first @ %.2f",
                fill_price,
            )
            self._close_internal(fill_price)
        self._position = {
            "side": side,
            "size": float(size),
            "entry_price": float(fill_price),
        }
        self._order_id += 1
        logger.info(
            "Paper open: %s %.4f @ %.2f", side.value, size, fill_price
        )
        return {
            "id": str(self._order_id),
            "side": side.value,
            "size": size,
            "price": fill_price,
        }

    async def close_position(
        self,
        side: PositionSide,
        size: float,
        fill_price: float | None = None,
        order_type: OrderType = OrderType.MARKET,
    ) -> dict:
        if self._position is None or fill_price is None:
            return {}
        self._close_internal(fill_price)
        self._order_id += 1
        return {
            "id": str(self._order_id),
            "status": "closed",
            "price": fill_price,
        }

    def _close_internal(self, exit_price: float) -> None:
        if self._position is None:
            return
        side = self._position["side"]
        entry = self._position["entry_price"]
        size = self._position["size"]
        if side == PositionSide.LONG:
            pnl = (exit_price - entry) * size
        elif side == PositionSide.SHORT:
            pnl = (entry - exit_price) * size
        else:
            pnl = 0.0
        self.balance += pnl
        logger.info(
            "Paper close: %s %.4f @ %.2f, PnL=$%.2f, Balance=$%.2f",
            side.value, size, exit_price, pnl, self.balance,
        )
        self._position = None

    async def place_stop_loss(
        self, side: PositionSide, trigger_price: float, size: float
    ) -> dict:
        # SL/TP 판정은 엔진 책임. paper는 인터페이스 호환만 유지 (no-op).
        return {"type": "stop_loss", "side": side.value, "price": trigger_price}

    async def place_take_profit(
        self, side: PositionSide, trigger_price: float, size: float
    ) -> dict:
        return {"type": "take_profit", "side": side.value, "price": trigger_price}

    async def cancel_all_orders(self) -> None:
        return None

    async def fetch_funding_history(
        self, since: str | None = None, until: str | None = None
    ) -> list[dict]:
        return []

    async def close(self) -> None:
        logger.info(
            "PaperExecutor closed. Final balance: $%.2f", self.balance
        )
