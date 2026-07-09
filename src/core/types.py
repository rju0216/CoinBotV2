"""뼈대 공용 데이터 타입."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from src.core.enums import ExitReason, PositionSide, PositionStatus, SignalSide


@dataclass
class Signal:
    side: SignalSide
    confidence: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.side in (SignalSide.LONG, SignalSide.SHORT)


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


@dataclass
class ExitDecision:
    reason: ExitReason = ExitReason.FORCE_EXIT
    note: str = ""


@dataclass
class AccountState:
    """진입 게이트·사이징이 참조하는 계좌 텔레메트리 (엔진 계측 → 모델).

    drawdown_pct = (peak_equity - equity) / peak_equity (>=0). 이 값들은 계측치일
    뿐이며 정책 판단(진입 차단·사이징)은 모델(StrategyModule)이 소유한다.
    """

    balance: float
    equity: float
    peak_equity: float
    daily_pnl: float
    initial_balance: float
    drawdown_pct: float


@dataclass
class StrategyContext:
    candles: dict[str, pd.DataFrame]
    current_price: float
    balance: float
    position: Position | None
    is_slot_occupied: bool
    params: dict[str, Any]
    now: datetime
    # 계좌 텔레메트리 (allow_entry / compute_position_size 의 리스크 판단 재료).
    # 엔진이 _build_ctx 에서 AccountTracker 상태로 채운다.
    account: AccountState | None = None
