"""거래소 인터페이스 facade. mode에 따라 Live/Paper executor 선택."""

from __future__ import annotations

import logging
from typing import Any

from src.core.enums import OrderType, PositionSide
from src.execution.live_executor import LiveExecutor
from src.execution.paper_executor import PaperExecutor

logger = logging.getLogger(__name__)


class Broker:
    def __init__(self, config: dict[str, Any], mode: str) -> None:
        """
        Args:
            mode: "live" | "paper" | "backtest". CLI subcommand에서 결정.
                  "live"는 LiveExecutor (OKX 실거래),
                  나머지는 PaperExecutor (시뮬레이션).
        """
        self.mode = mode
        if mode == "live":
            self._executor: LiveExecutor | PaperExecutor = LiveExecutor(config)
        else:
            self._executor = PaperExecutor(config)
        logger.info("Broker initialized in %s mode", mode)

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    @property
    def executor(self) -> LiveExecutor | PaperExecutor:
        return self._executor

    async def initialize(self) -> None:
        await self._executor.initialize()

    async def get_balance(self) -> float:
        return await self._executor.get_balance()

    async def get_wallet_balance(self) -> float:
        return await self._executor.get_wallet_balance()

    async def get_position(self) -> dict | None:
        return await self._executor.get_position()

    async def open_position(
        self,
        side: PositionSide,
        size: float,
        fill_price: float | None = None,
        order_type: OrderType = OrderType.MARKET,
        orderbook: dict | None = None,
    ) -> dict:
        # BL-2-2: orderbook은 PaperExecutor만 사용 (LiveExecutor는 거래소가 자동 처리).
        # LiveExecutor.open_position은 orderbook 인자 없으므로 mode 분기.
        if self.mode == "live":
            return await self._executor.open_position(
                side, size, fill_price, order_type
            )
        return await self._executor.open_position(
            side, size, fill_price, order_type, orderbook=orderbook,
        )

    async def close_position(
        self,
        side: PositionSide,
        size: float,
        fill_price: float | None = None,
        order_type: OrderType = OrderType.MARKET,
        orderbook: dict | None = None,
    ) -> dict:
        if self.mode == "live":
            return await self._executor.close_position(
                side, size, fill_price, order_type
            )
        return await self._executor.close_position(
            side, size, fill_price, order_type, orderbook=orderbook,
        )

    async def place_stop_loss(
        self, side: PositionSide, trigger_price: float, size: float
    ) -> dict:
        return await self._executor.place_stop_loss(side, trigger_price, size)

    async def place_take_profit(
        self, side: PositionSide, trigger_price: float, size: float
    ) -> dict:
        return await self._executor.place_take_profit(side, trigger_price, size)

    async def cancel_all_orders(self) -> None:
        await self._executor.cancel_all_orders()

    async def fetch_funding_history(
        self, since: str | None = None, until: str | None = None
    ) -> list[dict]:
        return await self._executor.fetch_funding_history(since=since, until=until)

    async def close(self) -> None:
        await self._executor.close()
