"""뼈대 프로토타입 공용 데이터 타입."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from src.core.enums import (
    ExitReason,
    OrderSide,
    OrderType,
    PositionSide,
    PositionStatus,
    SignalSide,
)


@dataclass
class Candle:
    timestamp: datetime
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Signal:
    side: SignalSide
    confidence: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.side in (SignalSide.LONG, SignalSide.SHORT)


@dataclass
class Order:
    side: OrderSide
    size: float
    order_type: OrderType
    price: float | None = None
    reduce_only: bool = False
    client_order_id: str | None = None


@dataclass
class Fill:
    timestamp: datetime
    side: OrderSide
    price: float
    size: float
    fee: float
    fee_currency: str = "USDT"


@dataclass
class Position:
    side: PositionSide
    size: float
    entry_price: float
    entry_time: datetime
    strategy_name: str
    stop_loss: float | None = None
    take_profit: float | None = None
    trade_id: int | None = None
    status: PositionStatus = PositionStatus.OPEN
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.status == PositionStatus.OPEN

    @property
    def is_orphan(self) -> bool:
        return self.status == PositionStatus.ORPHAN


@dataclass
class ExitDecision:
    reason: ExitReason = ExitReason.FORCE_EXIT
    note: str = ""


@dataclass
class StrategyContext:
    candles: dict[str, pd.DataFrame]
    current_price: float
    balance: float
    position: Position | None
    is_slot_occupied: bool
    params: dict[str, Any]
    now: datetime
